# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

def eval_float_maybe(x):
    if isinstance(x, int):
        return float(x)
    elif isinstance(x, float):
        return x
    else:
        return float(eval(x))

class LearnTemp:
    def __init__(
        self,
        init_temp, max_steps,
        init_targ_entr, final_targ_entr,
        entr_decay_factor,
        only_decrease_alpha,
        lr, device
    ):
        self.device = device
        self.init_temp = init_temp
        self.max_steps = max_steps
        self.log_alpha = torch.tensor(np.log(init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.init_targ_entr = eval_float_maybe(init_targ_entr)
        self.final_targ_entr = eval_float_maybe(final_targ_entr)
        assert self.final_targ_entr <= self.init_targ_entr
        self.entr_decay_factor = entr_decay_factor
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.targ_entr = self.init_targ_entr
        self.only_decrease_alpha = only_decrease_alpha

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, first_log_p, logger, step):
        t = (1.-step/self.max_steps)**self.entr_decay_factor
        self.targ_entr = (self.init_targ_entr - self.final_targ_entr)*t + self.final_targ_entr

        self.log_alpha_opt.zero_grad()
        alpha_loss = (self.alpha * (-first_log_p - self.targ_entr).detach()).mean()
        alpha_loss.backward()
        if not self.only_decrease_alpha or self.log_alpha.grad.item() > 0.:
            self.log_alpha_opt.step()
        logger.log('train_actor/target_entropy', self.targ_entr, step)
        logger.log('train_alpha/loss', alpha_loss, step)


class ExpTemp:
    def __init__(self, init_temp, temp_decay, min_temp, max_steps, device):
        self.device = device
        self.init_temp = init_temp
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self.max_steps = max_steps
        self.alpha = torch.tensor(init_temp).to(self.device)

    def update(self, first_log_p, logger, step):
        alpha = (1.-step/self.max_steps)**self.temp_decay
        alpha = (self.init_temp - self.min_temp)*alpha +self.min_temp
        self.alpha = torch.tensor(alpha).to(self.device)

class StepTemp:
    def __init__(self, num_train_steps, intervals, temps, device):
        self.device = device
        self.intervals = intervals
        self.temps = temps

        if self.intervals[-1] != num_train_steps:
            self.intervals.append(num_train_steps)

        assert all(i < j for i, j in zip(intervals, intervals[1:]))
        assert len(self.intervals) == len(self.temps) + 1
        assert self.intervals[0] == 0
        assert self.intervals[-1] == num_train_steps

        self.ipos = 0
        self.alpha = torch.tensor(temps[self.ipos]).to(self.device)

    def update(self, first_log_p, logger, step):
        if step >= self.intervals[self.ipos+1]:
            self.ipos += 1
            self.alpha = torch.tensor(self.temps[self.ipos]).to(self.device)
