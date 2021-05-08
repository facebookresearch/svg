import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F

import gym

import os
from collections import deque
import random
import math
import time

from .env import dmc

from gym import spaces


# https://github.com/openai/gym/blob/master/gym/wrappers/rescale_action.py
class RescaleAction(gym.ActionWrapper):
    def __init__(self, env, a, b):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        dtype = env.action_space.sample().dtype
        self.a = np.zeros(env.action_space.shape, dtype=dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=dtype) + b
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape)

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*((action - self.a)/(self.b - self.a))
        action = np.clip(action, low, high)
        return action

def make_norm_env(cfg):
    if 'gym' in cfg.env_name:
        from mbbl.env.env_register import make_env
        misc_info = {'reset_type': 'gym'}
        if 'gym_pets' in cfg.env_name:
            misc_info['pets'] = True
        env, meta = make_env(cfg.env_name, rand_seed=cfg.seed, misc_info=misc_info)

        env.metadata = env._env.metadata
        env.reward_range = env._env.reward_range
        env.spec = env._env.spec
        env.unwrapped = env._env.unwrapped
        # env._configured = env._env._configured
        env.close = env._env.close
        env = RescaleAction(env, -1., 1.)
        # assert np.all(env._env.action_space.high == env._env.action_space.high)
        assert not cfg.max_episode_steps

        # env.action_space = env._env.action_space
        if cfg.env_name == 'gym_fswimmer' or 'gym_pets' in cfg.env_name:
            env._max_episode_steps = env.env._env_info['max_length']
        else:
            env._max_episode_steps = env.env._env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env._env.render(mode='rgb_array')
            return frame

        env.render = render

        def set_seed(seed):
            if 'gym_pets' in cfg.env_name or cfg.env_name == 'gym_fswimmer':
                return env.env._env.seed(seed)
            else:
                return env.env._env.env.seed(seed)
    elif cfg.env_name == 'Humanoid-v2':
        env = gym.make('Humanoid-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'pets_cheetah':
        from svg.env import register_pets_environments
        register_pets_environments()
        env = gym.make('PetsCheetah-v0')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'pets_reacher':
        from svg.env import register_pets_environments
        register_pets_environments()
        env = gym.make('PetsReacher-v0')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'pets_pusher':
        from svg.env import register_pets_environments
        register_pets_environments()
        env = gym.make('PetsPusher-v0')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'mbpo_hopper':
        env = gym.make('Hopper-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'mbpo_walker2d':
        env = gym.make('Walker2d-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        # env.reset_old = env.reset
        # env.reset = lambda: env.reset_old()[0]
        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'mbpo_ant':
        from .env import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make('AntTruncatedObs-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'mbpo_cheetah':
        from svg.env import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make('HalfCheetah-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    elif cfg.env_name == 'mbpo_humanoid':
        from svg.env import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make('HumanoidTruncatedObs-v2')
        env = RescaleAction(env, -1., 1.)
        assert not cfg.max_episode_steps

        env._max_episode_steps = env.env._max_episode_steps

        def render(mode, height, width, camera_id):
            frame = env.env.render(mode='rgb_array')
            return frame
        env.render = render

        def set_seed(seed):
            return env.env.seed(seed)
    else:
        assert cfg.env_name.startswith('dmc_')
        env = dmc.make(cfg)

        if cfg.pixels:
            env = FrameStack(env, k=cfg.frame_stack)
            def set_seed(seed):
                return env.env.env._env.task.random.seed(seed)
        else:
            def set_seed(seed):
                return env.env._env.task.random.seed(seed)

    env.set_seed = set_seed

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class timer(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        print(f'{self.message}: {time.time() - self.start_time}')


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs_targets(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        if isinstance(output_mod, str):
            if output_mod == 'tanh':
                output_mod = torch.nn.Tanh()
            else:
                assert False
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def bottle(m, *inputs):
    seq_size, batch_size = inputs[0].size()[:2]
    inputs = [x.view(-1, *x.size()[2:]) for x in inputs]
    output = m(*inputs)
    return output.view(seq_size, batch_size, *output.size()[1:])


class FlatGaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.ndimension() == 2
        assert x.size(1) % 2 == 0
        n_batch = x.size(0)
        mu, sigma = x.chunk(2, dim=1)
        sigma = F.softplus(sigma)
        return pyd.Normal(mu, sigma)


class TanhTransform(pyd.transforms.Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class SquashedMultivariateNormal(
        pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale_tril):
        self.loc = loc
        self.scale_tril = scale_tril

        self.base_dist = pyd.MultivariateNormal(loc, scale_tril=scale_tril)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, value):
        assert ',' not in value # only single values
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


class freeze_env(object):
    def __init__(self, env):
        self._env = env

    def __enter__(self):
        self._init_state = self._env.env._env.physics.get_state().copy()
        self._elapsed_steps = self._env._elapsed_steps
        self._step_count = self._env.env._env._step_count

    def __exit__(self, *args):
        with self._env.env._env.physics.reset_context():
            self._env.env._env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self._env.env._env._step_count = self._step_count



class freeze_gym_env(object):
    def __init__(self, env):
        self._env = env
        self.time_env = self._env.env
        self.mj_env = self.time_env.env

    def __enter__(self):
        self._init_state = (
            self.mj_env.data.qpos.ravel().copy(), self.mj_env.data.qvel.ravel().copy()
        )
        self._elapsed_steps = self.time_env._elapsed_steps

    def __exit__(self, *args):
        self.mj_env.set_state(*self._init_state)
        self.time_env._elapsed_steps = self._elapsed_steps

class freeze_mbbl_env(object):
    def __init__(self, env):
        self._env = env.env
        self.env_name = self._env._env_name
        if 'gym_pets' in self.env_name or self.env_name == 'gym_fswimmer':
            self.mj_env = self._env._env
        else:
            self.time_env = self._env._env
            self.mj_env = self.time_env.env

    def __enter__(self):
        if self._env._env_name == 'gym_pendulum':
            self._init_state = self.mj_env.state.copy()
        else:
            self._init_state = (
                self.mj_env.data.qpos.ravel().copy(),
                self.mj_env.data.qvel.ravel().copy()
            )

        if 'gym_pets' not in self.env_name and self.env_name != 'gym_fswimmer':
            self._elapsed_steps = self.time_env._elapsed_steps

        self._current_step = self._env._current_step

    def __exit__(self, *args):
        if self._env._env_name == 'gym_pendulum':
            self.mj_env.state = self._init_state
        else:
            self.mj_env.set_state(*self._init_state)

        if 'gym_pets' not in self.env_name and self.env_name != 'gym_fswimmer':
            self.time_env._elapsed_steps = self._elapsed_steps
        self._env._current_step = self._current_step


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def accum_prod(x):
    assert x.dim() == 2
    x_accum = [x[0]]
    for i in range(x.size(0)-1):
        x_accum.append(x_accum[-1]*x[i])
    x_accum = torch.stack(x_accum, dim=0)
    return x_accum

import numpy as np

# https://pswww.slac.stanford.edu/svn-readonly/psdmrepo/RunSummary/trunk/src/welford.py
class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None):
        self._K = np.float64(0.)
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)

    def add_data(self, x):
        """Add data.
        """
        if x is None:
            return

        x = np.array(x)
        self.n += 1.
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)

        self._Ex += (x - self._K) / self.n
        self._Ex2 += (x - self._K) * (x - self._Ex)
        self._K = self._Ex

    def __call__(self, x):
        self.add_data(x)

    def max(self):
        """Max value for each element in array.
        """
        return self._max

    def min(self):
        """Min value for each element in array.
        """
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.

           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape)

        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

#    def __add__(self, val):
#        """Add two Welford objects.
#        """
#

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))

