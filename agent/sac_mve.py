import numpy as np
import torch
from torch import distributions
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List

import hydra

from agent import actor, critic, Agent
from common import utils, dx

class SACMVEAgent(Agent):
    """SAC-MVE agent."""
    def __init__(
        self, env_name, obs_dim, latent_obs_dim, action_dim, action_range,
        device,
        dx_cfg,
        num_train_steps,
        train_with_policy_mean, train_action_noise,
        obs_encoder_cfg,
        temp_cfg,
        actor_cfg,
        actor_lr, actor_betas,
        actor_update_freq, actor_num_sample,
        actor_clip_grad_norm,
        actor_mve,
        actor_detach_rho,
        critic_cfg, critic_lr, critic_tau,
        critic_target_update_freq,
        critic_clip_grad_norm,
        critic_target_mve,
        discount,
        seq_batch_size, seq_train_length,
        step_batch_size,
        update_freq, model_update_freq,
        rew_obs_noise, done_obs_noise, done_obs_repeat,
        rew_hidden_dim, rew_hidden_depth, rew_lr,
        done_hidden_dim, done_hidden_depth, done_lr,
        done_ctrl_accum,
        model_update_repeat,
        model_free_update_repeat,
        horizon,
        act_with_horizon, warmup_steps,
        det_suffix,
    ):
        super().__init__()
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.num_train_steps = num_train_steps
        self.det_suffix = det_suffix

        self.train_with_policy_mean = train_with_policy_mean
        self.train_action_noise = train_action_noise

        self.discount = discount
        self.discount_horizon = torch.tensor(
            [discount**i for i in range(horizon)]).to(device)
        self.seq_batch_size = seq_batch_size

        # self.seq_train_length = eval(seq_train_length)
        self.seq_train_length = seq_train_length

        self.step_batch_size = step_batch_size
        self.update_freq = update_freq
        self.model_update_repeat = model_update_repeat
        self.model_update_freq = model_update_freq
        self.model_free_update_repeat = model_free_update_repeat

        self.rew_obs_noise = rew_obs_noise
        self.done_obs_noise = done_obs_noise
        self.done_obs_repeat = done_obs_repeat

        self.horizon = horizon
        self.act_with_horizon = act_with_horizon

        self.warmup_steps = warmup_steps

        if obs_encoder_cfg is not None:
            import ipdb; ipdb.set_trace()
            self.obs_encoder = TODO
        else:
            self.obs_encoder = None

        self.temp = hydra.utils.instantiate(temp_cfg)

        self.dx = hydra.utils.instantiate(dx_cfg).to(self.device)

        self.rew = utils.mlp(
            obs_dim+action_dim, rew_hidden_dim, 1, rew_hidden_depth
        ).to(self.device)
        self.rew_opt = torch.optim.Adam(self.rew.parameters(), lr=rew_lr)

        self.done = utils.mlp(
            obs_dim+action_dim, done_hidden_dim, 1, done_hidden_depth
        ).to(self.device)
        self.done_ctrl_accum = done_ctrl_accum
        self.done_opt = torch.optim.Adam(self.done.parameters(), lr=done_lr)

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        mods = [self.actor]
        if isinstance(self.actor, actor.NormalRecActor):
            mods.append(self.dx.rec)
        params = utils.get_params(mods)
        self.actor_opt = torch.optim.Adam(
            params, lr=actor_lr, betas=actor_betas)
        self.actor_update_freq = actor_update_freq
        self.actor_num_sample = actor_num_sample
        self.actor_clip_grad_norm = actor_clip_grad_norm
        self.actor_mve = actor_mve
        self.actor_detach_rho = actor_detach_rho

        # optional critic
        self.critic = None
        if critic_cfg is not None:
            self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
            self.critic_target = hydra.utils.instantiate(critic_cfg).to(
                self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.train()
            self.critic_opt = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr)
            self.critic_tau = critic_tau
            self.critic_target_update_freq = critic_target_update_freq
            self.critic_clip_grad_norm = critic_clip_grad_norm

        self.critic_target_mve = critic_target_mve
        if critic_target_mve:
            assert self.critic is not None

        self.train()
        self.last_step = 0


    def train(self, training=True):
        self.training = training
        self.dx.train(training)
        self.rew.train(training)
        self.done.train(training)
        self.actor.train(training)
        if self.critic is not None:
            self.critic.train(training)


    def reset(self):
        self.next_actions = []


    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(dim=0)

        if not sample or self.train_with_policy_mean:
            # Assume we never act with the horion if we aren't sampling
            # TODO: Could add an option for acting with the horizon and not sampling
            action, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        else:
            if not self.act_with_horizon or self.last_step < self.warmup_steps:
                with torch.no_grad():
                    _, action, _ = self.actor(obs, compute_log_pi=False)
            else:
                if len(self.next_actions) > 0:
                    action = self.next_actions.pop(0).squeeze(0)
                else:
                    with torch.no_grad():
                        actions, _, _ = self.dx.unroll_policy(
                            obs, self.actor, sample=True)
                    action = actions[0]
                    actions = list(actions.split(split_size=1))
                    self.next_actions = actions[1:]

        if self.train_action_noise > 0.0:
            action += self.train_action_noise*torch.randn_like(action)

        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])


    def expand_Q(self, xs, critic, sample=True, discount=False):
        assert xs.dim() == 2
        n_batch = xs.size(0)
        us, log_p_us, pred_obs = self.dx.unroll_policy(
            xs, self.actor, sample=sample, detach_xt=self.actor_detach_rho)

        all_obs = torch.cat((xs.unsqueeze(0), pred_obs), dim=0)
        xu = torch.cat((all_obs, us), dim=2)
        dones = self.done(xu).sigmoid().squeeze(dim=2)
        not_dones = 1. - dones
        not_dones = utils.accum_prod(not_dones)
        last_not_dones = not_dones[-1]

        rewards = not_dones * self.rew(xu).squeeze(2)
        if critic is not None:
            with utils.eval_mode(self.critic):
                q1, q2 = self.critic(all_obs[-1], us[-1])
            q = torch.min(q1, q2).reshape(n_batch)
            rewards[-1] = last_not_dones * q

        assert rewards.size() == (self.horizon, n_batch)
        assert log_p_us.size() == (self.horizon, n_batch)
        rewards -= self.temp.alpha.detach() * log_p_us

        if discount:
            rewards *= self.discount_horizon.unsqueeze(1)

        total_rewards = rewards.sum(dim=0)

        first_log_p = log_p_us[0]
        total_log_p_us = log_p_us.sum(dim=0).squeeze()
        return total_rewards, first_log_p, total_log_p_us

    # @profile
    def update_actor_and_alpha(self, xs, logger, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()

        if step < self.warmup_steps or self.horizon == 0 or not self.actor_mve:
            # Do vanilla SAC updates while the model warms up.
            # i.e., fit to just the Q function
            _, pi, first_log_p = self.actor(xs)
            actor_Q1, actor_Q2 = self.critic(xs, pi)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.temp.alpha.detach() * first_log_p - actor_Q).mean()
        else:
            # Switch to the model-based updates.
            # i.e., fit to the controller's sequence cost
            rewards, first_log_p, total_log_p_us = self.expand_Q(
                xs, self.critic, sample=True, discount=True)
            assert total_log_p_us.size() == rewards.size()
            alpha_det = self.temp.alpha.detach()
            actor_loss = ((alpha_det * total_log_p_us - rewards)/self.horizon).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/entropy', -first_log_p.mean(), step)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.actor_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_clip_grad_norm)
        self.actor_opt.step()

        self.actor.log(logger, step)
        self.temp.update(first_log_p, logger, step)

        logger.log('train_alpha/value', self.temp.alpha, step)


    # @profile
    def update_critic(self, xs, xps, us, rs, not_done, logger, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()
        rs = rs.squeeze()
        not_done = not_done.squeeze()

        with torch.no_grad():
            if not self.critic_target_mve or step < self.warmup_steps:
                mu, target_us, log_pi = self.actor.forward(
                    xps, compute_pi=True, compute_log_pi=True)
                log_pi = log_pi.squeeze(1)

                target_Q1, target_Q2 = [
                    Q.squeeze(1) for Q in self.critic_target(xps, target_us)]
                target_Q = torch.min(target_Q1, target_Q2) - self.temp.alpha.detach() * log_pi
                assert target_Q.size() == rs.size()
                assert target_Q.ndimension() == 1
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()
            else:
                target_Q, first_log_p, total_log_p_us = self.expand_Q(
                    xps, self.critic_target, sample=True, discount=True)
                target_Q = target_Q - self.temp.alpha.detach() * first_log_p
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()

        current_Q1, current_Q2 = [Q.squeeze(1) for Q in self.critic(xs, us)]
        assert current_Q1.size() == target_Q.size()
        assert current_Q2.size() == target_Q.size()
        Q_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        logger.log('train_critic/loss', Q_loss, step)
        current_Q = torch.min(current_Q1, current_Q2)
        logger.log('train_critic/value', current_Q.mean(), step)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        if self.critic_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.critic_clip_grad_norm)
        self.critic_opt.step()

        self.critic.log(logger, step)


    # @profile
    def update(self, replay_buffer, logger, step):
        self.last_step = step
        if step % self.update_freq != 0:
            return

        if (self.horizon > 1 or not self.critic) and \
              (step % self.model_update_freq == 0) and \
              (self.actor_mve or self.critic_target_mve):
            for i in range(self.model_update_repeat):
                obses, actions, rewards = replay_buffer.sample_multistep(
                    self.seq_batch_size, self.seq_train_length)
                assert obses.ndimension() == 3
                self.dx.update_step(obses, actions, rewards, logger, step)

        n_updates = 1 if step < self.warmup_steps else self.model_free_update_repeat
        for i in range(n_updates):
            obs, action, reward, next_obs, not_done, not_done_no_max = \
              replay_buffer.sample(self.step_batch_size)

            if self.critic is not None:
                self.update_critic(
                    obs, next_obs, action, reward, not_done_no_max, logger, step
                )

            if step % self.actor_update_freq == 0:
                if self.actor_num_sample > 1:
                    obs_actor = obs.repeat(1, self.actor_num_sample) \
                      .view(self.step_batch_size*self.actor_num_sample, -1)
                else:
                   obs_actor = obs
                self.update_actor_and_alpha(obs_actor, logger, step)

            if self.rew_opt is not None:
                self.update_rew_step(obs, action, reward, logger, step)

            self.update_done_step(obs, action, not_done_no_max, logger, step)

        if self.critic is not None and step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


    def update_rew_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

        if self.rew_obs_noise > 0.:
            obs += self.rew_obs_noise*torch.randn_like(obs)

        xu = torch.cat((obs, action), dim=1)
        pred_reward = self.rew(xu)
        assert pred_reward.size() == reward.size()
        reward_loss = F.mse_loss(pred_reward, reward, reduction='mean')

        self.rew_opt.zero_grad()
        reward_loss.backward()
        self.rew_opt.step()

        logger.log('train_model/reward_loss', reward_loss, step)

    def update_done_step(self, obs, action, not_done, logger, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

        done = 1.-not_done
        xu = torch.cat((obs, action), dim=1)

        if self.done_obs_noise > 0.:
            I = (done == 1.).squeeze()
            xu_aug = xu[I].repeat(self.done_obs_repeat,1)
            xu_aug += self.done_obs_noise*torch.randn_like(xu_aug)
            xu = torch.cat((xu, xu_aug), dim=0)
            done_aug = torch.ones(I.sum()*self.done_obs_repeat, 1, device=done.device)
            done = torch.cat((done, done_aug), dim=0)

        pred_logits = self.done(xu)
        n_done = torch.sum(done)
        if n_done > 0.:
            pos_weight = (batch_size - n_done) / n_done
        else:
            pos_weight = torch.tensor(1.)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_logits, done, pos_weight=pos_weight,
            reduction='mean')

        self.done_opt.zero_grad()
        done_loss.backward()
        self.done_opt.step()

        logger.log('train_model/done_loss', done_loss, step)

