#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import pickle as pkl
import torch


def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', type=str)
    args = parser.parse_args()

    exp = pkl.load(open(args.pkl, 'rb'))
    exp.agent.actor.zero_grad(set_to_none=True)
    del exp.agent.actor_opt

    exp.agent.critic.zero_grad(set_to_none=True)
    del exp.agent.critic_opt

    del exp.agent.critic_target, exp.agent.critic_target_mve

    exp.agent.done.zero_grad(set_to_none=True)
    del exp.agent.done_opt

    exp.agent.dx.zero_grad(set_to_none=True)
    # del exp.agent.dx_opt

    pt_file = args.pkl[:-3] + 'pt'
    print(f'saving to {pt_file}')
    torch.save(exp, pt_file)


if __name__ == '__main__':
    main()
