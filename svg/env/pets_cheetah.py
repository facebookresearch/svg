# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PetsCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0] #- 0.0 * np.square(ob[2])
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        # Do the cosine transform directly in the environment rather than
        # as a pre-processing step:
        # https://github.com/kchua/handful-of-trials/blob/e1a62f217/dmbrl/config/halfcheetah.py#L43
        qpos = self.model.data.qpos.flat[1:]
        qpos_entry = np.concatenate(
            [qpos[0:1], np.sin(qpos[1:2]), np.cos(qpos[1:2]), qpos[2:]])
        return np.concatenate([
            (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            qpos_entry,
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
