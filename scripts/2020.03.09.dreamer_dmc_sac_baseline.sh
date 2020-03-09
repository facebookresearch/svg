#!/bin/bash

cd $(dirname $0)/..

ENVS=dmc_acrobot_swingup,dmc_cartpole_balance,dmc_cartpole_balance_sparse,dmc_cartpole_swingup,dmc_cartpole_swingup_sparse,dmc_cheetah_run,dmc_finger_spin,dmc_finger_turn_easy,dmc_finger_turn_hard,dmc_hopper_hop,dmc_hopper_stand,dmc_pendulum_swingup,dmc_quadruped_walk,dmc_quadruped_run,dmc_reacher_easy,dmc_reacher_hard,dmc_walker_stand,dmc_walker_walk,dmc_walker_run

./train.py -m experiment=dreamer_dmc_sac env=$ENVS obs_enc=proprio_identity agent.params.horizon=0 seed=1,2,3
