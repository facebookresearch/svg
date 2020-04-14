#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import shutil
import time
import pickle as pkl

from setproctitle import setproctitle
setproctitle('mve')

import hydra

from mve.video import VideoRecorder
from mve import utils
from mve.logger import Logger
from mve.replay_buffer import ReplayBuffer

if os.isatty(sys.stdout.fileno()):
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(
        mode='Verbose', color_scheme='Linux', call_pdb=1)


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_freq,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_norm_env(cfg)
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.done = False

        cfg.obs_dim = int(self.env.observation_space.shape[0])
        cfg.action_dim = self.env.action_space.shape[0]
        cfg.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        if isinstance(cfg.replay_buffer_capacity, str):
            cfg.replay_buffer_capacity = int(eval(cfg.replay_buffer_capacity))

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
            normalize_obs=cfg.normalize_obs,
        )
        self.replay_dir = os.path.join(self.work_dir, 'replay')

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

        self.step = 0
        self.steps_since_eval = 0
        self.steps_since_save = 0
        self.best_eval_rew = None

    def evaluate(self):
        episode_rewards = []
        for episode in range(self.cfg.num_eval_episodes):
            if self.cfg.fixed_eval:
                self.env.set_seed(episode)
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.agent.act(obs_norm, sample=False)
                    else:
                        action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
            episode_rewards.append(episode_reward)

            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval/episode_reward', episode_reward, self.step)
        if self.cfg.fixed_eval:
            self.env.set_seed(None)
        self.logger.dump(self.step)
        return np.mean(episode_rewards)

    def run(self):
        assert not self.done
        assert self.episode_reward == 0.0
        assert self.episode_step == 0
        self.agent.reset()
        obs = self.env.reset()

        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if self.done:
                if self.step > 0:
                    self.logger.log(
                        'train/episode_reward', self.episode_reward, self.step)
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/episode', self.episode, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.steps_since_eval >= self.cfg.eval_freq:
                    self.logger.log('eval/episode', self.episode, self.step)
                    eval_rew = self.evaluate()
                    self.steps_since_eval = 0

                    if self.best_eval_rew is None or eval_rew > self.best_eval_rew:
                        self.save(tag='best')
                        self.best_eval_rew = eval_rew

                    self.replay_buffer.save_data(self.replay_dir)
                    self.save(tag='latest')


                if self.step > 0 and self.cfg.save_freq and \
                  self.steps_since_save >= self.cfg.save_freq:
                    tag = str(self.step).zfill(self.cfg.save_zfill)
                    self.save(tag=tag)
                    self.steps_since_save = 0

                if self.cfg.num_initial_states is not None:
                    self.env.set_seed(self.episode % self.cfg.num_initial_states)
                obs = self.env.reset()
                self.agent.reset()
                self.done = False
                self.episode_reward = 0
                self.episode_step = 0
                self.episode += 1


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.agent.act(obs_norm, sample=True)
                    else:
                        action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps-1:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, self.done, _ = self.env.step(action)

            # allow infinite bootstrap
            done_float = float(self.done)
            done_no_max = done_float if self.episode_step + 1 < self.env._max_episode_steps \
              else 0.
            self.episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done_float, done_no_max)

            obs = next_obs
            self.episode_step += 1
            self.step += 1
            self.steps_since_eval += 1
            self.steps_since_save += 1


        if self.steps_since_eval > 1:
            self.logger.log('eval/episode', self.episode, self.step)
            self.evaluate()

        if self.cfg.delete_replay_at_end:
            shutil.rmtree(self.replay_dir)

    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(work_dir, tag='latest'):
        path = os.path.join(work_dir, f'{tag}.pkl')
        with open(path, 'rb') as f:
            return pkl.load(f)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['logger'], d['env']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        # override work_dir
        self.work_dir = os.getcwd()
        self.logger = Logger(self.work_dir,
                             save_tb=self.cfg.log_save_tb,
                             log_frequency=self.cfg.log_freq,
                             agent=self.cfg.agent.name)
        self.env = utils.make_norm_env(self.cfg)
        if 'max_episode_steps' in self.cfg and self.cfg.max_episode_steps is not None:
            self.env._max_episode_steps = self.cfg.max_episode_steps
        self.episode_step = 0
        self.episode_reward = 0
        self.done = False

        if os.path.exists(self.replay_dir):
            self.replay_buffer.load_data(self.replay_dir)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    # this needs to be done for successful pickle
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    from train import Workspace as W
    fname = os.getcwd() + '/latest.pkl'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()
