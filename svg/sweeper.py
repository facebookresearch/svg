# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import numpy.random as npr

from dataclasses import dataclass

import hydra
from hydra.plugins.sweeper import Sweeper
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins

import random

from svg import utils

class SVGSweeper(Sweeper):
    def __init__(self):
        pass

    def setup(
        self,
        config,
        config_loader,
        task_function,
    ):
        self.job_idx = 0
        self.config = config
        self.config_loader = config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader,
            task_function=task_function
        )

    def sweep(self, arguments):
        overrides = []

        MBPO_ENVS = ['mbpo_cheetah', 'mbpo_hopper', 'mbpo_walker2d',
                'mbpo_humanoid', 'mbpo_ant']
        POPLIN_ENVS = ['poplin_ant', 'poplin_cheetah', 'poplin_pets_cheetah',
                'poplin_swimmer', 'poplin_walker2d', 'poplin_hopper']
        ENV_DEFAULTS = {
            'full_poplin_sweep': POPLIN_ENVS,
            'full_mbpo_sweep': MBPO_ENVS,
            'mbpo_sac_baseline': MBPO_ENVS,
        }
        HORIZON_DEFAULTS = {
            'full_poplin_sweep': [0, 2, 3, 4, 5, 6, 11],
            'full_mbpo_sweep': [0, 2, 3, 4],
            'mbpo_sac_baseline': [0],
        }

        assert self.config.experiment in ENV_DEFAULTS, \
          "experiment not recognized"
        envs = ENV_DEFAULTS[self.config.experiment]
        horizons = HORIZON_DEFAULTS[self.config.experiment]

        n_sample = self.config.sweep.n_sample
        n_seed = self.config.sweep.n_seed

        npr.seed(self.config.seed)
        overrides = []
        for env in envs:
            for sample in range(n_sample):
                horizon = npr.choice(horizons)
                init_targ_entr = npr.choice([1, 0, -1, -2])
                final_targ_entr_choices = list(
                    range(init_targ_entr, -5, -1)) + \
                    [-2**i for i in range(3,5+1)]
                final_targ_entr = npr.choice(final_targ_entr_choices)
                gamma_choices = list(reversed(
                    [2**(-i) for i in range(1,7)])) + \
                    [2**i for i in range(0,7)]
                gamma = npr.choice(gamma_choices)
                for seed in range(1, n_seed+1):
                    overrides_i = {
                        'experiment': self.config.experiment,
                        'env': env,
                        'seed': seed,
                        'agent.horizon': horizon,
                        'learn_temp.init_targ_entr': init_targ_entr,
                        'learn_temp.final_targ_entr': final_targ_entr,
                        'learn_temp.entr_decay_factor': gamma
                    }
                    overrides_i = [
                        f'{k}={v}' for k,v in overrides_i.items()
                    ]
                    overrides.append(overrides_i)
        random.shuffle(overrides)
        # self.validate_batch_is_legal(overrides) # Can take a long time
        returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(returns)


@dataclass
class SVGSweeperConf:
    _target_: str = "svg.sweeper.SVGSweeper"

# Hacks for a non-standard plugin
Plugins.is_in_toplevel_plugins_module = lambda x, y: True
Plugins.instance().class_name_to_class['svg.sweeper.SVGSweeper'] = SVGSweeper

ConfigStore.instance().store(
    group="hydra/sweeper",
    name="svg",
    node=SVGSweeperConf,
    provider="svg",
)
