import abc

import torch
import torch.nn.functional as F

import hydra

from . import utils

class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class SACMVEAgent(Agent):
    """SAC-MVE agent."""
    def __init__(
        self, env_name, obs_dim, action_dim, action_range,
        device,
        dx_cfg,
        num_train_steps,
        temp_cfg,
        actor_cfg,
        actor_lr, actor_betas,
        actor_update_freq,
        actor_mve,
        actor_detach_rho,
        actor_dx_threshold,
        actor_mve_update,
        critic_cfg, critic_lr, critic_tau,
        critic_target_update_freq,
        critic_target_mve,
        critic_mve_update,
        discount,
        seq_batch_size, seq_train_length,
        step_batch_size,
        update_freq, model_update_freq,
        rew_hidden_dim, rew_hidden_depth, rew_lr,
        done_hidden_dim, done_hidden_depth, done_lr,
        done_ctrl_accum,
        model_update_repeat,
        model_free_update_repeat,
        horizon,
        warmup_steps,
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

        self.horizon = horizon

        self.warmup_steps = warmup_steps

        if temp_cfg is not None:
            self.temp = hydra.utils.instantiate(temp_cfg)
        else:
            # TODO
            self.temp = get_best_temp(env_name, horizon)

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
        params = utils.get_params(mods)
        self.actor_opt = torch.optim.Adam(
            params, lr=actor_lr, betas=actor_betas)
        self.actor_update_freq = actor_update_freq
        self.actor_mve = actor_mve
        self.actor_detach_rho = actor_detach_rho
        self.actor_dx_threshold = actor_dx_threshold

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

        self.actor_mve_update = actor_mve_update
        self.critic_mve_update = critic_mve_update
        self.critic_target_mve = critic_target_mve

        self.train()
        self.last_step = 0
        self.rolling_dx_loss = None


    def __setstate__(self, d):
        self.__dict__ = d

        if 'full_target_mve' not in d:
            self.full_target_mve = False

        if 'actor_dx_threshold' not in d:
            self.actor_dx_threshold = None
            self.rolling_dx_loss = None

    def train(self, training=True):
        self.training = training
        self.dx.train(training)
        self.rew.train(training)
        self.done.train(training)
        self.actor.train(training)
        if self.critic is not None:
            self.critic.train(training)

    def reset(self):
        pass

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(dim=0)

        if not sample:
            action, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        else:
            with torch.no_grad():
                _, action, _ = self.actor(obs, compute_log_pi=False)

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
            with utils.eval_mode(critic):
                q1, q2 = critic(all_obs[-1], us[-1])
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

    def update_actor_and_alpha(self, xs, logger, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()

        do_model_free_update = step < self.warmup_steps or \
          self.horizon == 0 or not self.actor_mve or \
          (self.actor_dx_threshold is not None and \
           self.rolling_dx_loss is not None and
           self.rolling_dx_loss > self.actor_dx_threshold)

        if do_model_free_update:
            # Do vanilla SAC updates while the model warms up.
            # i.e., fit to just the Q function
            _, pi, first_log_p = self.actor(xs)
            actor_Q1, actor_Q2 = self.critic(xs, pi)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.temp.alpha.detach() * first_log_p - actor_Q).mean()
            logger.log('train_actor/q_avg', actor_Q.mean(), step)
        else:
            # Switch to the model-based updates.
            # i.e., fit to the controller's sequence cost
            rewards, first_log_p, total_log_p_us = self.expand_Q(
                xs, self.critic, sample=True, discount=True)
            assert total_log_p_us.size() == rewards.size()
            actor_loss = -(rewards/self.horizon).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/entropy', -first_log_p.mean(), step)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.actor.log(logger, step)
        self.temp.update(first_log_p, logger, step)

        logger.log('train_alpha/value', self.temp.alpha, step)

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
        logger.log('train_critic/target_avg', target_Q.mean(), step)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        self.critic_opt.step()

        self.critic.log(logger, step)

    def get_rollout(self, first_xs, first_us, first_rs, next_xs, first_not_dones):
        """ MVE critic loss from Feinberg et al (2015) """
        assert first_xs.dim() == 2
        assert first_us.dim() == 2
        assert first_rs.dim() == 2
        assert next_xs.dim() == 2
        assert first_not_dones.dim() == 2

        # unroll policy, concatenate obs and actions
        pred_us, log_p_us, pred_xs = self.dx.unroll_policy(
            next_xs, self.actor, sample=True, detach_xt=self.actor_detach_rho)
        all_obs = torch.cat((first_xs.unsqueeze(0), next_xs.unsqueeze(0), pred_xs))
        all_us = torch.cat([first_us.unsqueeze(0), pred_us])
        xu = torch.cat([all_obs, all_us], dim=2)
        horizon_len = all_obs.size(0) - 1  # H

        # get immediate rewards
        pred_rs = self.rew(xu[1:-1])  # t from 0 to H - 1
        rewards = torch.cat([first_rs.unsqueeze(0), pred_rs]).squeeze(2)
        rewards = rewards.unsqueeze(1).expand(-1, horizon_len, -1)

        # get not dones factor matrix, rows --> t, cols --> k
        first_not_dones = first_not_dones.unsqueeze(0)
        init_not_dones = torch.ones_like(first_not_dones)  # we know the first states are not terminal
        pred_not_dones = 1. - self.done(xu[2:]).sigmoid()  # t from 1 to H
        not_dones = torch.cat([init_not_dones, first_not_dones, pred_not_dones]).squeeze(2)
        not_dones = not_dones.unsqueeze(1).repeat(1, horizon_len, 1)
        triu_rows, triu_cols = torch.triu_indices(row=horizon_len + 1, col=horizon_len,
                                                  offset=1, device=not_dones.device)
        not_dones[triu_rows, triu_cols, :] = 1.
        not_dones = not_dones.cumprod(dim=0).detach()

        return all_obs, all_us, log_p_us, rewards, not_dones

    def _get_mve_discounts(self, horizon_len):
        # get lower-triangular reward discount factor matrix
        discount = torch.tensor(self.discount)
        discount_exps = torch.stack([torch.arange(-i, -i + horizon_len) for i in range(horizon_len)], dim=1)
        discount_matrix = discount ** discount_exps
        discount_matrix = discount_matrix.tril().unsqueeze(-1)
        return discount_matrix

    def _actor_mve_update(self, log_p_us, rewards, all_obs, all_us, logger, step):
        _, horizon_len, n_batch = rewards.shape

        final_q1, final_q2 = self.critic(all_obs[-1], all_us[-1])
        final_qs = torch.min(final_q1, final_q2)
        final_qs = final_qs.squeeze(-1).expand(1, horizon_len, -1)  # reshape for discounting

        discount_matrix = self._get_mve_discounts(horizon_len).to(final_qs.device)
        alpha = self.temp.alpha.detach()
        # soft_rewards = rewards[1:] - (alpha * self.discount * log_p_us[1:].unsqueeze(1))
        # import pdb; pdb.set_trace()
        action_value = (discount_matrix * torch.cat([rewards[1:], final_qs])).sum(0) - \
                       (discount_matrix[:-1] * (alpha * self.discount * log_p_us[1:].unsqueeze(1))).mean(0)

        assert log_p_us.shape == action_value.shape
        alpha = self.temp.alpha.detach()
        actor_loss = (alpha * log_p_us - action_value).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/entropy', -log_p_us.mean(), step)
        logger.log('train_actor/q_avg', action_value.mean(), step)

        self.actor.log(logger, step)

    def _critic_mve_update(self, all_obs, all_us, log_p_us, rewards, not_dones, logger, step):
        _, horizon_len, n_batch = rewards.shape

        discount_matrix = self._get_mve_discounts(horizon_len).to(rewards.device)
        alpha = self.temp.alpha.detach()
        entropy = -(self.discount * alpha * not_dones[1:] * log_p_us.unsqueeze(1).expand(-1, horizon_len, -1))
        # soft_rewards = () + entropy
        # entropy = -(self.discount * alpha * log_p_us.unsqueeze(1).expand(-1, horizon_len, -1))
        # soft_rewards = rewards + entropy
        discounted_rewards = (discount_matrix * not_dones[:-1] * rewards).sum(0) + \
                             (discount_matrix * entropy).sum(0)

        # get target q-values
        target_q1, target_q2 = self.critic_target(all_obs[-1], all_us[-1])
        target_qs = torch.min(target_q1, target_q2)
        target_qs = target_qs.squeeze(-1).expand(horizon_len, -1)
        q_discounts = (torch.tensor(self.discount) ** torch.arange(horizon_len, 0, step=-1)).to(target_qs.device)
        target_qs = target_qs * (not_dones[-1] * q_discounts.unsqueeze(-1))

        critic_targets = (discounted_rewards + target_qs).detach()

        # get predicted q-values
        with utils.eval_mode(self.critic):
            q1, q2 = self.critic(all_obs[:-1].flatten(end_dim=-2).detach(),
                                 all_us[:-1].flatten(end_dim=-2).detach())
            q1, q2 = q1.reshape(horizon_len, n_batch), q2.reshape(horizon_len, n_batch)
        assert q1.size() == critic_targets.size()
        assert q2.size() == critic_targets.size()

        # update critics
        # q1_loss = (not_dones[:-1, 0].detach() * (q1 - critic_targets).pow(2)).mean()
        # q2_loss = (not_dones[:-1, 0].detach() * (q2 - critic_targets).pow(2)).mean()
        q1_loss = ((q1 - critic_targets).pow(2)).mean()
        q2_loss = ((q2 - critic_targets).pow(2)).mean()
        Q_loss = q1_loss + q2_loss

        current_Q = torch.min(q1, q2)
        logger.log('train_critic/value', current_Q.mean(), step)
        logger.log('train_critic/target_avg', critic_targets.mean(), step)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        logger.log('train_critic/loss', Q_loss, step)
        self.critic_opt.step()

        self.critic.log(logger, step)

    # @profile
    def update(self, replay_buffer, logger, step):
        self.last_step = step
        if step % self.update_freq != 0:
            return

        if (self.horizon >= 1 or not self.critic) and \
              (step % self.model_update_freq == 0) and \
              (self.critic_mve_update or self.actor_mve_update):
            for i in range(self.model_update_repeat):
                obses, actions, rewards = replay_buffer.sample_multistep(
                    self.seq_batch_size, self.seq_train_length)
                assert obses.ndimension() == 3
                dx_loss = self.dx.update_step(obses, actions, rewards, logger, step)
                if self.actor_dx_threshold is not None:
                    if self.rolling_dx_loss is None:
                        self.rolling_dx_loss = dx_loss
                    else:
                        factor = 0.9
                        self.rolling_dx_loss = factor*self.rolling_dx_loss + \
                          (1.-factor)*dx_loss

        n_updates = 1 if step < self.warmup_steps else self.model_free_update_repeat
        for i in range(n_updates):
            obs, action, reward, next_obs, not_done, not_done_no_max = \
              replay_buffer.sample(self.step_batch_size)

            # update reward and term_fn models
            if self.rew_opt is not None:
                self.update_rew_step(obs, action, reward, logger, step)
            self.update_done_step(obs, action, not_done_no_max, logger, step)

            # do dx model rollout
            if step > self.warmup_steps and (self.critic_mve_update or self.actor_mve_update):
                rollout_args = obs, action, reward, next_obs, not_done_no_max
                all_obs, all_us, log_p_us, rewards, not_dones = self.get_rollout(*rollout_args)

            # critic update
            assert self.critic is not None
            if self.critic_mve_update and step > self.warmup_steps:
                self._critic_mve_update(all_obs, all_us, log_p_us, rewards, not_dones, logger, step)
            else:
                assert self.critic_target_mve is False
                self.update_critic(obs, next_obs, action, reward, not_done_no_max, logger, step)

            # actor update
            if step % self.actor_update_freq != 0:
                pass
            elif self.actor_mve_update and step > self.warmup_steps:
                actor_update_args = log_p_us, rewards, all_obs, all_us
                self._actor_mve_update(*actor_update_args, logger, step)
                self.temp.update(log_p_us, logger, step)
                logger.log('train_alpha/value', self.temp.alpha, step)
            else:
                self.update_actor_and_alpha(obs, logger, step)

            if self.critic is not None and step % self.critic_target_update_freq == 0:
                utils.soft_update_params(
                    self.critic, self.critic_target, self.critic_tau)

    def update_rew_step(self, obs, action, reward, logger, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

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
