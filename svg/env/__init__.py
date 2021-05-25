# Copyright (c) Facebook, Inc. and its affiliates.

import gym

MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'svg.env.ant:AntTruncatedObsEnv'),
        'max_episode_steps': 1000,
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'svg.env.humanoid:HumanoidTruncatedObsEnv'),
        'max_episode_steps': 1000,
    },
)

PETS_ENVIRONMENT_SPECS = (
	{
        'id': 'PetsCheetah-v0',
        'entry_point': (f'svg.env.pets_cheetah:PetsCheetahEnv'),
        'max_episode_steps': 1000,
    },
	{
        'id': 'PetsReacher-v0',
        'entry_point': (f'svg.env.pets_reacher:PetsReacherEnv'),
        'max_episode_steps': 150,
    },
	{
        'id': 'PetsPusher-v0',
        'entry_point': (f'svg.env.pets_pusher:PetsPusherEnv'),
        'max_episode_steps': 150,
    },
)


def _register_environments(register, specs):
    for env in specs:
        register(**env)

    gym_ids = tuple(environment_spec['id'] for environment_spec in specs)
    return gym_ids

def register_mbpo_environments():
    _register_environments(gym.register, MBPO_ENVIRONMENT_SPECS)

def register_pets_environments():
    _register_environments(gym.envs.registration.register, PETS_ENVIRONMENT_SPECS)
