#!/usr/bin/env python3

import numpy as np
import numpy.random as npr

from datetime import datetime
import time
import os
import sys
import argparse
from subprocess import Popen, DEVNULL

from mve import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    if not args.dry:
        now = datetime.now()
        sweep_dir = './exp/'+now.strftime("%Y.%m.%d/%H%M") + '_' + args.experiment
        assert not os.path.exists(sweep_dir)
        os.makedirs(sweep_dir)

    # envs = ['poplin_ant', 'poplin_cheetah', 'poplin_pets_cheetah',
    #         'poplin_swimmer', 'poplin_walker2d', 'poplin_hopper']
    envs = ['poplin_walker2d', 'poplin_hopper']
    # envs = ['mbpo_cheetah', 'mbpo_hopper', 'mbpo_walker2d',
    #         'mbpo_humanoid', 'mbpo_ant']
    # envs = ['mbpo_walker2d']

    n_sample = 20
    n_seed = 10

    global_i = 1
    for sample in range(n_sample):
        for env in envs:
            # horizon = npr.choice([3, 5])
            horizon = npr.choice([3])
            init_targ_entr = npr.choice([1, 0, -1, -2])
            final_targ_entr_choices = list(range(init_targ_entr, -5, -1)) + \
              [-2**i for i in range(3,5+1)]
            final_targ_entr = npr.choice(final_targ_entr_choices)
            gamma_choices = list(reversed([2**(-i) for i in range(1,7)])) + \
                [2**i for i in range(0,7)]
            gamma = npr.choice(gamma_choices)

            overrides = utils.Overrides()
            overrides.add('env', [env])
            overrides.add('seed', list(range(1, n_seed+1)))
            overrides.add('experiment', [args.experiment])
            overrides.add('log_save_tb', ['false'])
            overrides.add('agent.params.horizon', [horizon])
            overrides.add('learn_temp.params.init_targ_entr', [init_targ_entr])
            overrides.add('learn_temp.params.final_targ_entr', [final_targ_entr])
            overrides.add('learn_temp.params.entr_decay_factor', [gamma])
            overrides.add('hydra.launcher.params.queue_parameters.slurm.partition',
                          ['scavenge'])

            overrides.add('hydra.sweep.dir', [sweep_dir])
            overrides.add('hydra.sweep.subdir', [str(global_i)+'.${seed}'])
            global_i += 1

            cmd = ['python3', 'train.py', '-m']
            cmd += overrides.cmd()

            print(' '.join(cmd))
            if not args.dry:
                env = os.environ.copy()
                p = Popen(cmd, env=env, stderr=DEVNULL, stdout=DEVNULL)
                time.sleep(5)


if __name__ == '__main__':
    main()
