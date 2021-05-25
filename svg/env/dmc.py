# Copyright (c) Facebook, Inc. and its affiliates.

import dmc2gym


def make(cfg):
    """Helper function to create dm_control environment"""
    action_repeat = cfg.get('action_repeat', 1)

    if cfg.env_name == 'dmc_ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        toks = cfg.env_name.split('_')[1:]
        domain_name = toks[0]
        task_name = '_'.join(toks[1:])

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=cfg.seed,
        visualize_reward=False,
        frame_skip=action_repeat,
        from_pixels=cfg.pixels,
        height=64,
        width=64,
    )

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env
