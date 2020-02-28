import dmc2gym


def make(cfg):
    """Helper function to create dm_control environment"""
    action_repeat = cfg.get('action_repeat', 1)
    if 'name' not in cfg.env:
        if cfg.env == 'ball_in_cup_catch':
            domain_name = 'ball_in_cup'
            task_name = 'catch'
        else:
            domain_name = cfg.env.split('_')[0]
            task_name = '_'.join(cfg.env.split('_')[1:])

        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           seed=cfg.seed,
                           visualize_reward=True,
                           frame_skip=action_repeat)
    else:
        # For @bda's older models
        env = dmc2gym.make(seed=cfg.seed, **cfg.env.params)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env
