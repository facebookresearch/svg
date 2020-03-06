import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from pprint import pprint

from omegaconf import OmegaConf

def plot_ac_exp(root, print_cfg=False, print_overrides=True):
    config = OmegaConf.load(f'{root}/.hydra/config.yaml')
    df = pd.read_csv(f'{root}/train.csv')
    N_smooth = 200
    N_downsample = 200

    def get_smooth(key):
        # it, vae_loss = smooth(df.index, df.vae_loss, N)
        it, v = df.step/1E3, df[key]
        _it = np.linspace(it.min(), it.max(), num=N_downsample)
        _v = sp.interpolate.interp1d(it, v)(_it)
        return _it, _v

    nrow, ncol = 2, 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
    axs = axs.reshape(-1)

    ax = axs[0]
    ax.plot(*get_smooth('actor_loss'), label='Total')
    # ax.set_ylim(0, 0.3)
    ax.set_xlabel('1k Iteration')
    ax.set_title('Actor Loss')

    if 'critic_loss' in df:
        ax = axs[1]
        ax.plot(*get_smooth('critic_loss'))
        ax.set_ylim(0, None)
        ax.set_xlabel('1k Iteration')
        ax.set_title('Critic Loss')

    if 'model_obs_loss' in df:
        ax = axs[2]
        ax.plot(*get_smooth('model_obs_loss'), label='Obs Loss')
        ax.set_ylim(0, None)
        ax.set_xlabel('1k Iteration')
        ax.set_ylabel('Obs Loss')
        ax.legend()

        if 'model_reward_loss' in df:
            ax = ax.twinx()
            ax.plot(*get_smooth('model_reward_loss'), label='Rew Loss', color='red')
            ax.set_xlabel('1k Iteration')
            ax.set_ylabel('Rew Loss')
            ax.set_ylim(0, None)
            ax.legend()

    ax = axs[3]
    ax.plot(*get_smooth('alpha_value'), label='alpha loss')
    ax.set_title('Alpha Value')
    ax.set_yscale('log')

    ax = axs[4]
    ax.plot(*get_smooth('actor_entropy'))
    ax.plot(*get_smooth('actor_target_entropy'))
    ax.set_title('Actor Entropy')

    ax = axs[5]
    l, = ax.plot(df.step/1E3, df.episode_reward, alpha=0.4)
    try:
        df = pd.read_csv(f'{root}/eval.csv')
    except:
        df = None
    if df is not None and len(df) > 0:
        if len(df) == 1:
            ax.scatter(df.step/1E3, df.episode_reward, color=l.get_color())
        else:
            ax.plot(df.step/1E3, df.episode_reward, color=l.get_color())
        if 'gym' not in config.env_name and 'mbpo' not in config.env_name \
          and config.env_name != 'Humanoid-v2' and 'pets' not in config.env_name:
            ax.set_ylim(0, 1000)
    ax.set_xlabel('1k Iteration')
    ax.set_title('Reward')

    if print_cfg:
        pprint(config)
    if print_overrides:
        o = OmegaConf.load(f'{root}/.hydra/overrides.yaml')
        pprint(o)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(root + ': ' + config.env_name, fontsize=20)
    return fig, axs
