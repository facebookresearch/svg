#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch

import argparse
import os
import sys
import pickle as pkl
import shutil
from omegaconf import OmegaConf
from collections import namedtuple
import dmc2gym

import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib import cm

from multiprocessing import Process

from svg.video import VideoRecorder
from svg import utils, dx


def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--n_vids', type=int, default=10)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--output_dir_tag', type=str, default='eval')
    parser.add_argument('--framerate', type=int, default=16)
    parser.add_argument('--mode', type=str,
                        default='mean', choices=['mean', 'sample', 'ctrl'])
    parser.add_argument(
        '--vid_mode', type=str, default='full', choices=['full', 'highlight'])
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--no_mp', action='store_true')
    parser.add_argument('--show_dones', action='store_true')
    args = parser.parse_args()

    ev = EvalVis(args)
    ev.run()


class EvalVis:
    def __init__(self, args):
        self.args = args
        self.eval_dir = f'{args.exp_root}/{args.output_dir_tag}.{args.pkl_tag}.{args.mode}.{args.vid_mode}'
        if os.path.exists(self.eval_dir):
            shutil.rmtree(self.eval_dir)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.exp = pkl.load(open(f'{args.exp_root}/{args.pkl_tag}.pkl', 'rb'))
        del self.exp.logger
        self.env = self.exp.env
        self.max_obs = torch.zeros(self.exp.agent.obs_dim)
        self.reward_bounds = [0., 1.1]
        self.dx = self.exp.agent.dx
        self.domain_name = self.exp.cfg.env_name

    def run(self):
        rews = []
        for i in range(self.args.n_episodes):
            create_vid = i < self.args.n_vids
            rews.append(self.run_episode(self.args.start_seed+i, create_vid))
            print(rews)

            f = open(f'{self.eval_dir}/rews.csv', 'w')
            f.write(','.join(map(str, rews)) + '\n')
            f.close()


    # TODO: Could clean up a bit
    def run_episode(self, seed, create_vid):
        if create_vid:
            episode_dir = f'{self.eval_dir}/{seed:02d}'
            os.makedirs(episode_dir, exist_ok=True)

        self.env.set_seed(seed)
        obs = self.env.reset()
        domain_name = self.domain_name
        horizon = self.exp.agent.horizon

        device = 'cuda'
        done = False
        total_reward = 0.
        reward = 0.

        args = self.args
        env = self.env
        exp = self.exp
        replay_buffer = self.exp.replay_buffer
        step = 0

        ps = []
        while not done:
            if create_vid:
                if 'quadruped' in domain_name:
                    camera_id = 2
                else:
                    camera_id = 0
                frame = env.render(
                    mode='rgb_array',
                    height=256,
                    width=256,
                    camera_id=camera_id,
                )
                env_fname = f'{episode_dir}/env_{step:04d}.png'
                plt.imsave(env_fname, frame)

            if self.exp.cfg.normalize_obs:
                mu, sigma = replay_buffer.get_obs_stats()
                obs = (obs - mu) / sigma
            obs = torch.FloatTensor(obs).to(device)

            if args.mode == 'mean':
                action_seq, _, _ = self.exp.agent.dx.unroll_policy(
                    obs.unsqueeze(0), exp.agent.actor,
                    sample=False, last_u=True)
            elif args.mode == 'sample':
                action_seq, _, _ = self.exp.agent.dx.unroll_policy(
                    obs.unsqueeze(0), exp.agent.actor,
                    sample=True, last_u=True)
            elif args.mode == 'ctrl':
                exp.agent.ctrl.num_samples = 100 # TODO
                action_seq = self.exp.agent.ctrl.forward(
                    exp.agent.dx,
                    exp.agent.actor,
                    obs,
                    exp.agent.critic,
                    return_seq=True,
                )
                action_seq = action_seq[:-1]
            else:
                assert False

            if action_seq.ndimension() == 3:
                action_seq = action_seq.squeeze(dim=1)

            action = action_seq[0]
            action = action.clamp(min=env.action_space.low.min(),
                                max=env.action_space.high.max())
            if action.ndimension() == 1:
                # TODO: This is messy, shouldn't be so sensitive to the dim here.
                action = action.unsqueeze(0)

            if create_vid:
                def get_nominal_states(obs, actions):
                    assert obs.ndimension() == 1
                    assert actions.ndimension() == 2
                    obs = obs.unsqueeze(0)
                    pred_obs = exp.agent.dx.unroll(obs, actions.unsqueeze(1)).squeeze(1)
                    pred_obs = torch.cat((obs, pred_obs), dim=0)
                    return pred_obs

                # if env._max_episode_steps - env._elapsed_steps > exp.agent.horizon:
                if env._max_episode_steps - step > exp.agent.horizon:
                    true_xs = [obs.cpu()]
                    true_rews = [reward]
                    if 'gym' in domain_name:
                        freeze = utils.freeze_mbbl_env
                    elif domain_name == 'Humanoid-v2' or 'mbpo' in domain_name:
                        freeze = utils.freeze_gym_env
                    else:
                        freeze = utils.freeze_env
                    with freeze(env):
                        for t in range(horizon):
                            xt, rt, done, _ = env.step(utils.to_np(action_seq[t]))
                            if self.exp.cfg.normalize_obs:
                                mu, sigma = replay_buffer.get_obs_stats()
                                xt = (xt - mu) / sigma
                            true_xs.append(xt)
                            true_rews.append(rt)
                    true_xs = np.stack(true_xs)
                    true_rews = np.stack(true_rews)

                    max_obs = torch.from_numpy(true_xs).abs().max(axis=0).values.float().detach()
                    I = max_obs > self.max_obs
                    self.max_obs[I] = 1.1*max_obs[I]
                    if true_rews.min() < self.reward_bounds[0]:
                        self.reward_bounds[0] = 1.1*true_rews.min().item()
                    if true_rews.max() > self.reward_bounds[1]:
                        self.reward_bounds[1] = 1.1*true_rews.max().item()
                else:
                    true_xs = true_rews = None

                n_sample = 1
                pred_xs = []
                pred_rews = []
                pred_dones = []
                for i in range(n_sample):
                    pred_x = get_nominal_states(obs.squeeze(), action_seq[:-1])
                    max_obs = pred_x.abs().max(axis=0).values.cpu().detach()
                    I = max_obs > self.max_obs
                    self.max_obs[I] = 1.1*max_obs[I]
                    xu = torch.cat((pred_x, action_seq), dim=-1)
                    # xu = pred_x
                    pred_rew = exp.agent.rew(xu)
                    if pred_rew.min() < self.reward_bounds[0]:
                        self.reward_bounds[0] = 1.1*pred_rew.min().item()
                    if pred_rew.max() > self.reward_bounds[1]:
                        self.reward_bounds[1] = 1.1*pred_rew.max().item()

                    pred_done = exp.agent.done(xu).sigmoid()

                    pred_xs.append(pred_x.squeeze())
                    pred_rews.append(pred_rew.squeeze())
                    pred_dones.append(pred_done.squeeze())

                pred_xs = [x.cpu() for x in pred_xs]
                pred_rews = [x.cpu() for x in pred_rews]
                pred_dones = [x.cpu() for x in pred_dones]
                action_seq = action_seq.cpu()

                def f():
                    preds_fname = os.path.join(episode_dir,
                                            f'preds_{step:04d}.png')
                    self.plot_obs_rew(
                        true_xs, pred_xs, true_rews, pred_rews, pred_dones, preds_fname)

                    ctrl_fname = f'{episode_dir}/ctrl_{step:04d}.png'
                    self.plot_ctrl(action_seq, fname=ctrl_fname)

                    fname = f'{episode_dir}/{step:04d}.png'
                    os.system(f'convert {preds_fname} -trim {preds_fname}')
                    os.system(f'convert {ctrl_fname} -trim {ctrl_fname}')
                    if self.args.vid_mode == 'highlight':
                        # os.system('convert -gravity west -append '
                        #         f'{preds_fname} {ctrl_fname} {fname}')
                        os.system(f'convert {preds_fname} -resize x300 {fname}')
                        os.system(f'convert {env_fname} -resize 300x300! {env_fname}')
                        os.system(f'convert +append {env_fname} {fname} -resize x300 {fname}')
                        # os.system(f'convert {fname} -resize 1328x150! {fname}')
                    elif 'pendulum' in domain_name:
                        os.system('convert -gravity center -append '
                                f'{preds_fname} {ctrl_fname} {fname}')
                        os.system(f'convert {fname} -resize x700 {fname}')
                        os.system(f'convert -gravity center {env_fname} -resize x700 {env_fname}')
                        os.system('convert -gravity center +append '
                                f'{env_fname} {fname} {fname}')
                    else:
                        os.system('convert -gravity center +append -resize x700 '
                                f'{env_fname} {preds_fname} {fname}')
                        os.system('convert -gravity center -append -resize 1200x '
                                f'{fname} {ctrl_fname} {fname}')

                if self.args.no_mp:
                    f()
                else:
                    p = Process(target=f)
                    p.start()
                    ps.append(p)

            obs, reward, done, _ = env.step(utils.to_np(action.squeeze(0)))
            total_reward += reward
            print(
                f'--- Step {step} -- Total Rew: {total_reward:.2f} -- Step Rew: {reward:.2f}'
            )
            step += 1
            if args.n_steps is not None and step > args.n_steps:
                done = True

        if create_vid:
            for p in ps:
                p.join()

            os.system(
                f'ffmpeg -y -framerate {self.args.framerate} -i {episode_dir}/%04d.png -q 3 {episode_dir}/vid.mp4'
            )

        return total_reward

    def plot_obs_rew(self, true_xs, pred_xs, true_rews, pred_rews, pred_dones, fname):
        domain_name = self.domain_name
        bounds = (-self.max_obs, self.max_obs)
        reward_bounds = self.reward_bounds

        gridspec_kw = {'wspace': 0, 'hspace': 0}
        if self.args.vid_mode == 'highlight':
            # fig, axs = plt.subplots(2, 3, figsize=(4, 3), gridspec_kw=gridspec_kw)
            fig, axs = plt.subplots(3, 4, figsize=(4, 3), gridspec_kw=gridspec_kw)
        elif 'cheetah' in domain_name.lower():
            fig, axs = plt.subplots(3, 6, figsize=(14, 10), gridspec_kw=gridspec_kw)
        elif 'walker' in domain_name:
            fig, axs = plt.subplots(5, 5, figsize=(14, 10), gridspec_kw=gridspec_kw)
        elif domain_name == 'mbpo_humanoid':
            fig, axs = plt.subplots(6, 8, figsize=(16, 10), gridspec_kw=gridspec_kw)
        elif domain_name == 'mbpo_ant':
            fig, axs = plt.subplots(5, 6, figsize=(14, 10), gridspec_kw=gridspec_kw)
        elif 'humanoid' in domain_name.lower():
            fig, axs = plt.subplots(8, 9, figsize=(16, 10), gridspec_kw=gridspec_kw)
        elif 'pendulum' in domain_name:
            fig, axs = plt.subplots(4, 1, figsize=(6, 10), gridspec_kw=gridspec_kw)
        elif 'hopper' in domain_name:
            fig, axs = plt.subplots(4, 3, figsize=(14, 10), gridspec_kw=gridspec_kw)
        elif 'swimmer' in domain_name:
            fig, axs = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw=gridspec_kw)
        else:
            fig, axs = plt.subplots(5, 5, figsize=(14, 10), gridspec_kw=gridspec_kw)

        axs = axs.ravel()
        if self.args.vid_mode != 'highlight':
            add_label(axs[0], 'States', fontsize=20)
        for ax in axs:
            # ax.axis('off')
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.patch.set_edgecolor('black')

        horizon_p1, state_dim = pred_xs[0].shape
        horizon = horizon_p1-1

        for i in range(state_dim):
            if i >= len(axs)-1:
                # print(f'Warning: Skipping state dim {i}')
                continue
            ax = axs[i]
            if true_xs is not None:
                ax.plot(true_xs[:, i], color='k', label='Ground Truth')

            color = None
            for j in range(len(pred_xs)):
                p, = ax.plot(utils.to_np(pred_xs[j][:, i]), alpha=1., color=color)
                color = p.get_color()
            ax.set_ylim(bounds[0][i], bounds[1][i])
            ax.set_xlim(0, horizon)

            if self.args.vid_mode != 'highlight':
                ax.axhline(color='k', linestyle='--', alpha=0.4)

        rew_ax = axs[-1]
        if true_rews is not None:
            rew_ax.plot(true_rews, alpha=0.5, color='k')

        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
        for j in range(len(pred_rews)):
            rew_ax.plot(utils.to_np(pred_rews[j]), alpha=1., color=color)
        rew_ax.set_ylim(*reward_bounds)
        rew_ax.set_xlim(0, horizon)

        rew_ax.get_xaxis().set_ticklabels([])
        rew_ax.get_yaxis().set_ticklabels([])
        if self.args.vid_mode != 'highlight':
            add_label(rew_ax, 'Rewards', fontsize=20)

        if self.args.show_dones:
            done_ax = rew_ax.twinx()
            for j in range(len(pred_dones)):
                done_ax.plot(utils.to_np(pred_dones[j]), alpha=1.)
            done_ax.set_ylim(-0.1, 1.1)
            done_ax.set_xlim(0, horizon)
            done_ax.get_xaxis().set_ticklabels([])
            done_ax.get_yaxis().set_ticklabels([])

        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)


    def plot_ctrl(self, plan_us, fname):
        assert plan_us.ndimension() == 2
        T, nctrl = plan_us.size()

        domain_name = self.domain_name
        gridspec_kw = {'wspace': 0, 'hspace': 0}
        if self.args.vid_mode == 'highlight':
            # fig, axs = plt.subplots(1, 8, figsize=(20, 1.5), gridspec_kw=gridspec_kw)
            fig, axs = plt.subplots(1, 8, figsize=(8, 1), gridspec_kw=gridspec_kw)
        elif domain_name in ['Humanoid-v2', 'mbpo_humanoid']:
            fig, axs = plt.subplots(3, 6, figsize=(16, 4), gridspec_kw=gridspec_kw)
        elif 'humanoid' in domain_name:
            fig, axs = plt.subplots(3, 7, figsize=(16, 4), gridspec_kw=gridspec_kw)
        elif 'pendulum' in domain_name:
            fig, axs = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw=gridspec_kw)
        else:
            fig, axs = plt.subplots(1, nctrl, figsize=(16, 2), gridspec_kw=gridspec_kw)

        if nctrl > 1:
            axs = axs.ravel()
        else:
            axs = [axs]
        # for ax in axs: ax.axis('off')

        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
        for i in range(nctrl):
            if i > len(axs)-1:
                # print(f'Warning: Skipping action dim {i}')
                continue
            ax = axs[i]
            ax.plot(utils.to_np(plan_us[:, i]), color=color)
            ax.set_ylim(-1., 1.)
            ax.set_xlim(0, plan_us.shape[0]-1)
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.axhline(color='k', linestyle='--', alpha=0.4)

        for i in range(nctrl, len(axs)):
            ax = axs[i]
            ax.set_axis_off()

        if 'pendulum' in domain_name or self.args.vid_mode == 'highlight':
            fontsize = 20
        else:
            fontsize = 14
        if self.args.vid_mode != 'highlight':
            add_label(axs[0], 'Actions', fontsize)

        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)


def add_label(ax, text, fontsize, xpos=0., ypos=1.):
    ax.text(
        xpos,
        ypos,
        text,
        ha='left',
        va='top',
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.',
            alpha=0.5,
        ),
    )


if __name__ == '__main__':
    main()
