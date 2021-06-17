#!/usr/bin/env python3
#
# This is a quick hacky replay buffer test, may be worth
# improving at some point

import numpy as np
import numpy.random as npr

import os
import sys
import shutil

sys.path.append('.')
from common.replay_buffer import ReplayBuffer

if os.isatty(sys.stdout.fileno()):
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(
        mode='Verbose', color_scheme='Linux', call_pdb=1)

scratch_dir = 't'
if os.path.exists(scratch_dir):
    shutil.rmtree(scratch_dir)

obs_shape = [10]
action_shape = [5]
capacity = 100
device = 'cpu'

r = ReplayBuffer(obs_shape, action_shape, capacity, device)

d = scratch_dir + '/1'

for i in range(250):
    obs = npr.randn(*obs_shape)
    obs[:] = i
    action = npr.randn(*action_shape)
    next_obs = npr.randn(*obs_shape)
    reward = npr.randn()
    done = 0
    done_no_max = 0
    r.add(obs, action, reward, next_obs, done, done_no_max)
    if (i+1) % 27 == 0:
        r.save(d)

r.save(d)

r1 = ReplayBuffer(obs_shape, action_shape, capacity, device)
r1.load(d)

N = 1
eps = 1e-5
# assert(np.linalg.norm(r.obses[:N] - r1.obses[:N]) <= eps)
assert(r.obses.sum() == r1.obses.sum())


for i in range(37):
    obs = npr.randn(*obs_shape)
    obs[:] = 1000+i
    action = npr.randn(*action_shape)
    next_obs = npr.randn(*obs_shape)
    reward = npr.randn()
    done = 0
    done_no_max = 0
    r.add(obs, action, reward, next_obs, done, done_no_max)
    r1.add(obs, action, reward, next_obs, done, done_no_max)

d = scratch_dir + '/1'
r.save(d)

r2 = ReplayBuffer(obs_shape, action_shape, capacity, device)
r2.load(d)

assert(r.obses.sum() == r1.obses.sum())
assert(r.obses.sum() == r2.obses.sum())
assert(r1.obses.sum() == r2.obses.sum())
