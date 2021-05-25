# Copyright (c) Facebook, Inc. and its affiliates.
#
# This contains all of the messy and undocumented code that we
# used for analysis and plotting when developing SAC-SVG(H).
# We hope some pieces will be scavengeable.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from pprint import pprint
from glob import glob
import os
import seaborn as sns
import pickle as pkl
import textwrap

from omegaconf import OmegaConf

def plot_exp(root, print_cfg=False, print_overrides=True, Qmax=None,
             obsmax=None, suptitle=None, save=None,
             plot_rew=True, N_smooth=200, N_downsample=200,
             smooth_train_rew=True):
    config = OmegaConf.load(f'{root}/.hydra/config.yaml')
    df = pd.read_csv(f'{root}/train.csv')

    def get_smooth(key):
        # it, vae_loss = smooth(df.index, df.vae_loss, N)
        it, v = df.step, df[key]
        _it = np.linspace(it.min(), it.max(), num=N_downsample)
        _v = sp.interpolate.interp1d(it, v)(_it)
        return _it, _v

    nrow, ncol = 2, 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
    axs = axs.reshape(-1)

    ax = axs[0]
    ax.plot(*get_smooth('actor_loss'), label='Total')
    # ax.set_ylim(0, 0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_title('Actor Loss')

    if 'critic_Q_loss' in df:
        ax = axs[1]
        ax.plot(*get_smooth('critic_Q_loss'))
        ax.set_ylim(0, Qmax)
        # ax.set_xlabel('1k Interactions')
        ax.set_title('Critic Loss')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        if 'critic_recon_loss' in df:
            ax = ax.twinx()
            ax.plot(*get_smooth('critic_recon_loss'), color='red')
            ax.set_ylim(0, None)
            ax.set_ylabel('Recon Loss')
    elif 'critic_loss' in df:
        ax = axs[1]
        ax.plot(*get_smooth('critic_loss'))
        ax.set_ylim(0, Qmax)
        # ax.set_xlabel('1k Interactions')
        ax.set_title('Critic Loss')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    if 'model_obs_loss' in df:
        ax = axs[2]
        ax.plot(*get_smooth('model_obs_loss'), label='Obs Loss')
        ax.set_ylim(0, obsmax)
        ax.set_title('Obs Loss')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # ax.legend()

        if 'model_reward_loss' in df and plot_rew:
            ax = ax.twinx()
            ax.plot(*get_smooth('model_reward_loss'), label='Rew Loss', color='red')
            ax.set_ylabel('Rew Loss')
            ax.set_ylim(0, None)
            ax.legend()
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax = axs[3]
    ax.plot(*get_smooth('alpha_value'), label='alpha loss')
    ax.set_title('Alpha Value')
    ax.set_yscale('log')
    ax.set_xlabel('Interations')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax = axs[4]
    ax.plot(*get_smooth('actor_entropy'))
    ax.plot(*get_smooth('actor_target_entropy'))
    ax.set_title('Actor Entropy')
    ax.set_xlabel('Interactions')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax = axs[5]
    if smooth_train_rew:
        l, = ax.plot(*get_smooth('episode_reward'), alpha=0.4)
    else:
        l, = ax.plot(df.step, df.episode_reward, alpha=0.4)
    df = load_eval(root)
    if df is not None and len(df) > 0:
        if len(df) == 1:
            ax.scatter(df.step, df.episode_reward, color=l.get_color())
        else:
            ax.plot(df.step, df.episode_reward, color=l.get_color())
        if 'gym' not in config.env_name and 'mbpo' not in config.env_name \
          and config.env_name != 'Humanoid-v2' and 'pets' not in config.env_name:
            ax.set_ylim(0, 1000)
    ax.set_xlabel('Interactions')
    ax.set_title('Reward')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if print_cfg:
        pprint(config)
    if print_overrides:
        o = OmegaConf.load(f'{root}/.hydra/overrides.yaml')
        pprint(o)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=20)
    else:
        fig.suptitle(root + ': ' + config.env_name, fontsize=20)

    if save:
        fig.savefig(save, transparent=True)
        os.system(f'convert -trim {save} {save}')

    return fig, axs


def load_eval(root):
    eval_f = f'{root}/eval.csv'
    try:
        skiprows = int('episode' in next(open(eval_f, 'r')))
        names = ('episode', 'episode_reward', 'step')
        eval_df = pd.read_csv(eval_f, names=names, skiprows=skiprows)
        return eval_df
    except:
        return None


def sweep_summary(root):
    configs = {}
    all_summary = []
    for d in glob(f'{root}/*/'):
        eval_df = load_eval(d)
        if eval_df is None:
            continue
#         last_eval_rew = eval_df.episode_reward.values[-10:].mean()
        last_eval_rew = eval_df.episode_reward.values[-1]
        best_eval_rew = eval_df.episode_reward.values.max()
        fname = f'{d}/config.yaml'
        if not os.path.exists(fname):
            fname = f'{d}/.hydra/config.yaml'
            assert os.path.exists(fname)
        config = OmegaConf.load(fname)
        configs[d] = config
        fname = f'{d}/overrides.yaml'
        if not os.path.exists(fname):
            fname = f'{d}/.hydra/overrides.yaml'
            assert os.path.exists(fname)
        overrides = OmegaConf.load(fname)
        summary = dict(x.split('=') for x in overrides)
        summary['best_eval_rew'] = best_eval_rew
        summary['last_eval_rew'] = last_eval_rew
        summary['d'] = d
        summary['env_name'] = config.env_name
        all_summary.append(summary)

    if len(all_summary) == 0:
        print('No experiments with eval data found.')
        return [None]*4

    all_summary = pd.DataFrame(all_summary)
    for col in all_summary.columns:
        if col != 'env_name' and len(all_summary[col].unique()) == 1:
            all_summary.drop(col,inplace=True,axis=1)

    filt = ['env_name', 'seed']
    groups = [x.split('=')[0] for x in overrides]
    groups = [x for x in groups if x not in filt]
    groups = list(set(groups) & set(all_summary.columns))
    groups = ['env_name'] + groups
    groups = all_summary.groupby(groups)
    agg = groups.agg(['mean', 'std'])

    return all_summary, groups, agg, configs

def plot_rew(root, ax=None, label=None):
    eval_df = load_eval(root)
    if eval_df is None:
        return

    if ax is None:
        nrow, ncol = 1, 1
        fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
        ax.set_xlabel('1k Updates')

    l, = ax.plot(eval_df.step/1000, eval_df.episode_reward, label=label)

def plot_rew_list(ds, title=None, ax=None):
    nrow, ncol = 1, 1
    if ax is None:
        fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
    ax.set_xlabel('1k Updates')
#     ax.set_ylim(0, 1000)
    if title is not None:
        ax.set_title(title)
    for d in ds:
        label = d.split('/')[-2]
        plot_rew(d, ax=ax, label=label)
    ax.legend()

def plot_all_rew(root):
    nrow, ncol = 1, 1
    fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
    ax.set_xlabel('1k Updates')
    title = '/'.join(root.split('/')[-3:])
    ax.set_title(title)
#     ax.set_ylim(0, 1000)
    for d in glob(f'{root}/*/'):
        label = d.split('/')[-2]
        plot_rew(d, fig=fig, label=label)
    ax.legend()

def plot_agg(df, agg, ncol=4):
    nrow = int(np.ceil(len(agg)/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
    if nrow == 1 and ncol == 1:
        axs = [axs]
    else:
        axs = axs.ravel()
    for ax, (r, sub_df) in zip(axs, agg.iterrows()):
        if isinstance(r, str):
            r = [r]
        I = df.index == df.index
        for k, v in zip(agg.index.names, r):
            I = I & (df[k] == v)
        df_I = df[I]
        title = '.'.join([f'{k}={v}' for k,v in zip(agg.index.names, r)])
        title = title.replace('agent.params.', '').replace('model.params.', '')
        title = '\n'.join(textwrap.wrap(title, 45))
        plot_rew_list(df_I.d.values, title=title, ax=ax)
    fig.tight_layout()

def plot_ablation(
    groups, title, xmax=None,
    save=None, lw=3,
    xlabel='Interactions', ylabel='Reward',
    legend=False, only_include_valid=False,
    axhline=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(4.5,3))

    for group in groups:
        all_df = []
        min_step = None
        for root in group['roots']:
            df = load_eval(root)
            if df is None:
                continue
            t = max(df['step'])
            if min_step is None or (only_include_valid and t < min_step) or \
                    (not only_include_valid and t > min_step):
                min_step = t
            # df['f'] = eval_f
            all_df.append(df)

        step_interp = np.linspace(0, min_step, num=20)
        all_df_interp = []
        for df in all_df:
            rew_interp = np.interp(step_interp, df['step'], df['episode_reward'])
            df_interp = pd.DataFrame({'step': step_interp, 'rew': rew_interp})
            all_df_interp.append(df_interp)
        all_df_interp = pd.concat(all_df_interp)
        label = group['tag'] if legend else None
        if 'color' in group:
            sns.lineplot(x='step', y='rew', data=all_df_interp,
                         ax=ax, linewidth=lw, label=label, color=group['color'])
        else:
            sns.lineplot(x='step', y='rew', data=all_df_interp,
                         ax=ax, linewidth=lw, label=label)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if xmax is not None:
        ax.set_xlim(0, xmax)

    if axhline:
        ax.axhline(axhline, lw=lw, linestyle='--', color='k')

    ax.set_xlabel('Interactions')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    fig.tight_layout()

    if save is not None:
        d = os.path.dirname(save)
        if not os.path.exists(d):
            os.makedirs(d)
        fig.savefig(save, transparent=True)
        os.system(f'convert -trim {save} {save}')

def plot_comparison(
    ac_base, ac_is, mbpo_f, title, xmax=None,
    steve=None, cb=None, save=None, lw=3,
    xlabel='Interactions', ylabel='Reward', alive_bonus=None,
    n_interp=20, n_smooth=4, only_include_valid=True,
    sac_base=None, sac_is=None, sac_f=None, sac_scale=1.0
):
    fig, ax = plt.subplots(1, 1, figsize=(4.5,3))

    colors = sns.color_palette("deep")

    all_df = []
    min_step = None
    for i in ac_is:
        root = f'{ac_base}/{i}'
        df = load_eval(root)
        if df is None:
            continue
        t = max(df['step'])
        if min_step is None or (only_include_valid and t < min_step) or \
                (not only_include_valid and t > min_step):
            min_step = max(df['step'])
        # df['f'] = eval_f
        all_df.append(df)

    step_interp = np.linspace(0, min_step, num=n_interp)
    all_df_interp = []
    for df in all_df:
        step, rew = df['step'], df['episode_reward']
        step = step[n_smooth-1:]
        rew = np.convolve(rew, np.full(n_smooth, 1./n_smooth), mode='valid')
        rew_interp = np.interp(step_interp, step, rew)
        df_interp = pd.DataFrame({'step': step_interp, 'rew': rew_interp})
        all_df_interp.append(df_interp)
    all_df_interp = pd.concat(all_df_interp)
    sns.lineplot(x='step', y='rew', data=all_df_interp, ax=ax, linewidth=lw)

    # SAC
    if sac_is is not None:
        sac_dfs = []
        for i in sac_is:
            root = f'{sac_base}/{i}'
            df = load_eval(root)
            if df is None:
                continue
            t = max(df['step'])
            if min_step is None or (only_include_valid and t < min_step) or \
                    (not only_include_valid and t > min_step):
                min_step = max(df['step'])

            step, rew = df['step'], df['episode_reward']
            step = step[n_smooth-1:]
            rew = np.convolve(rew, np.full(n_smooth, 1./n_smooth), mode='valid')
            rew_interp = np.interp(step_interp, step, rew)
            df = pd.DataFrame({'step': step_interp, 'rew': rew_interp})
            sac_dfs.append(df)
        sac_dfs = pd.concat(sac_dfs)
        sns.lineplot(x='step', y='rew', data=sac_dfs, ax=ax, linewidth=lw)

    if sac_f is not None:
        sac_data = np.loadtxt(sac_f, delimiter=',').T
        init_cost = np.expand_dims(np.array([0., all_df_interp.iloc[0].rew]), 1)

        sac_data = np.hstack((init_cost, sac_data))
        sac_data[1] = sac_scale*sac_data[1] # Correcting the reconstruction of their data
        l, = ax.plot(1e6*sac_data[0], sac_data[1], linewidth=lw)
        # ax.axhline(sac_data[1][-3:].mean(), linestyle='--', color=l.get_color(), linewidth=lw)

    mbpo_data = pkl.load(open(mbpo_f, 'rb'))
#     mbpo_data['step'] = mbpo_data['x']*1e3
# #     ax.plot(mbpo_data['step'], mbpo_data['y'], linewidth=lw)
#     min_step = min(max(mbpo_data['step']), min_step)
#     step_interp = np.linspace(0, min_step, num=20)
#     mbpo_interp = np.interp(step_interp, mbpo_data['step'], mbpo_data['y'])
#     ax.plot(step_interp, mbpo_interp, linewidth=lw)
    ax.axhline(mbpo_data['y'][-1], linestyle='--', linewidth=lw, color=colors[4])

    if steve is not None:
        ax.axhline(steve, linestyle='--', color='g', linewidth=lw)

    if alive_bonus is not None:
        ax.axhline(alive_bonus, linestyle='dotted', color='k', linewidth=lw, alpha=0.3)

    if cb is not None:
        cb(ax)

    # ax.set_xscale('log')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if xmax is not None:
        ax.set_xlim(0, xmax)

    ax.set_xlabel('Interactions')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, transparent=True)
        os.system(f'convert -trim {save} {save}')
