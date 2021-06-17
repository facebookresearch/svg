#!/usr/bin/env python3

import os
import numpy as np
from multiprocessing import Pool
from subprocess import Popen, DEVNULL
import submitit


r = 'exp/2020.05.18/1501_sac_mve_poplin_10/'
cmds = []
n_trials = 20

for i in os.listdir(r):
    d = f'{r}/{i}'
    if os.path.exists(f'{d}/eval.csv'):
        fname = f'{d}/eval.latest.mean.full/rews.csv'
        cmd = f'./eval-vis-model.py --n_steps 1000 --n_episodes {n_trials} --n_vids 0 --mode mean {d} --start_seed 0 --pkl_tag latest'.split(' ')
        if os.path.exists(fname):
            rews = np.loadtxt(fname, delimiter=',')
            if len(rews) < n_trials:
                cmds.append(cmd)
        else:
            cmds.append(cmd)

def f(cmd):
    p = Popen(cmd)
    p.communicate()
    return None

executor = submitit.AutoExecutor(folder="exp/2020.06.04/poplin_eval")
executor.update_parameters(
    timeout_min=1000, partition="priority", comment='NeurIPS', gpus_per_node=1)

# print(cmds)
jobs = []
for c in cmds:
    job = executor.submit(f, c)
    print(job.job_id)
