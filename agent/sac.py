import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import hydra

from agent import actor
from agent import critic
from agent import Agent
from common import utils


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, planet_cfg,
                 critic_cfg, actor_cfg, discount, init_temperature, alpha_lr,
                 alpha_betas, actor_lr, actor_betas, actor_update_frequency,
                 critic_lr, critic_betas, critic_tau,
                 critic_target_update_frequency, batch_size):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.planet = None  # TODO: instantiate planet models if the actor needs them
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        if isinstance(self.actor, actor.OptActor):
            self.actor.opt.critic = self.critic  # TODO: Could clean up

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        if sample:
            _, action, _ = self.actor(obs, compute_log_pi=False)
        else:
            action, _, _ = self.actor(obs,
                                         compute_pi=False,
                                         compute_log_pi=False)
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        _, policy_action, log_pi = self.actor(next_obs)
        target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        kwargs = {}
        if isinstance(self.actor, actor.OptActor):
            kwargs['update_decoder'] = True
        _, pi, log_pi = self.actor(obs, **kwargs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
