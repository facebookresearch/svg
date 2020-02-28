from typing import Optional, List

import torch
from torch import distributions, jit, nn
import torch.nn.functional as F

import hydra

from common import utils

class PlaNet(nn.Module):
    def __init__(self, obs_dim, action_dim, action_range, device,
                 obs_encoder_cfg, obs_model_cfg, reward_model_cfg,
                 transition_model_cfg, hidden_dim, belief_dim,
                 embed_dim, free_nats, state_dim, model_lr,
                 model_eps, jit):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.state_dim = state_dim

        self.device = device

        self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
        self.obs_model = hydra.utils.instantiate(obs_model_cfg).to(self.device)
        self.reward_model = hydra.utils.instantiate(reward_model_cfg).to(self.device)
        self.transition_model = hydra.utils.instantiate(transition_model_cfg).to(self.device)

        if jit:
            import ipdb; ipdb.set_trace()

        self.free_nats = torch.full((1,), free_nats, device=self.device)

        trainable_model_params = utils.get_params([
            self.obs_encoder, self.obs_model, self.transition_model,
            self.reward_model
        ])
        self.model_opt = torch.optim.Adam(trainable_model_params,
                                          lr=model_lr,
                                          eps=model_eps)

    def train(self, training=True):
        self.obs_encoder.train(training)
        self.obs_model.train(training)
        self.reward_model.train(training)
        self.transition_model.train(training)

    def update_step(self, obses, actions, rewards, logger, step):
        T, batch_size, _ = obses.shape

        init_belief = torch.zeros(batch_size,
                                  self.belief_dim,
                                  device=self.device)
        # TODO: In the proprio setting I think we can set the initial
        # state her to the initial observation?
        init_state = torch.zeros(batch_size,
                                 self.state_dim,
                                 device=self.device)

        # TODO: verify if we need to have nonterms or not
        beliefs, prior_states, prior_means, prior_stds, posterior_states, posterior_means, posterior_stds = self.transition_model(
            init_state, actions[:-1], init_belief,
            utils.bottle(self.obs_encoder, obses[1:]))

        obs_loss = F.mse_loss(utils.bottle(self.obs_model, beliefs,
                                           posterior_states),
                              obses[1:],
                              reduction='none')
        obs_loss = obs_loss.sum(dim=2).mean(dim=(0, 1))

        pred_rewards = utils.bottle(self.reward_model, beliefs,
                                    posterior_states)
        reward_loss = F.mse_loss(pred_rewards, rewards[:-1], reduction='mean')

        kl_loss = distributions.kl_divergence(
            distributions.Normal(posterior_means, posterior_stds),
            distributions.Normal(prior_means, prior_stds)).sum(dim=2)
        kl_loss = torch.max(kl_loss, self.free_nats)
        kl_loss = kl_loss.mean(dim=(0, 1))

        loss = obs_loss + reward_loss + kl_loss

        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        logger.log('train_model/obs_loss', obs_loss, step)
        logger.log('train_model/reward_loss', reward_loss, step)
        logger.log('train_model/kl_loss', kl_loss, step)
        logger.log('train_model/loss', loss, step)


class ProprioObservationEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, embed_dim, hidden_depth):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim, embed_dim),
                                   nn.ReLU(inplace=True))

        #self.trunk = utils.mlp(obs_dim,
        #                       hidden_dim,
        #                       embed_dim,
        #                       hidden_depth,
        #                       output_mod=nn.ReLU(inplace=True))

        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.trunk(obs)


class ProprioObservationModel(nn.Module):
    def __init__(self, belief_dim, state_dim, obs_dim, hidden_dim,
                 hidden_depth):
        super().__init__()

        #self.trunk = utils.mlp(belief_dim + state_dim, hidden_dim, obs_dim,
        #                       hidden_depth)
        self.trunk = nn.Linear(belief_dim + state_dim, obs_dim)

        self.apply(utils.weight_init)

    def forward(self, belief, state):
        h = torch.cat([belief, state], dim=1)
        return self.trunk(h)


class RewardModel(nn.Module):
    def __init__(self, belief_dim, state_dim, hidden_dim, hidden_depth,
                 squash):
        super().__init__()

        self.trunk = utils.mlp(belief_dim + state_dim, hidden_dim, 1,
                               hidden_depth)
        self.squash = squash

        self.apply(utils.weight_init)

    def forward(self, belief, state):
        h = torch.cat([belief, state], dim=1)
        reward = self.trunk(h)
        reward = reward.squeeze(dim=1)
        if self.squash:
            reward = torch.sigmoid(reward)
        return reward


class TransitionModel(nn.Module):
    def __init__(self,
                 belief_dim,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 embed_dim,
                 min_std,
                 learn_state_delta,
                 use_mean_state=False):
        super().__init__()
        self.min_std = min_std
        self.learn_state_delta = learn_state_delta
        self.use_mean_state = use_mean_state

        self.state_action_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True))

        self.rnn = nn.GRUCell(hidden_dim, belief_dim)

        self.prior = nn.Sequential(nn.Linear(belief_dim, hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, 2 * state_dim))

        self.posterior = nn.Sequential(
            nn.Linear(belief_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2 * state_dim))

        self.apply(utils.weight_init)

    def forward_distribution(self, mean, std):
        std = F.softplus(std) + self.min_std
        sample = mean + std * torch.randn_like(mean)
        return sample, mean, std

    def forward_prior_distribution(self, x):
        mean, std = torch.chunk(self.prior(x), 2, dim=1)
        return self.forward_distribution(mean, std)

    def forward_posterior_distribution(self, x):
        mean, std = torch.chunk(self.posterior(x), 2, dim=1)
        return self.forward_distribution(mean, std)

    def forward(self,
                prev_state,
                actions,
                prev_belief,
                obses: Optional[torch.Tensor] = None,
                nonterms: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        #self.outputs.clear()
        # T x B x E
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_stds = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_stds = [torch.empty(0)] * T

        # init
        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        for t in range(T - 1):
            #if obses is None:
            #    if self.use_mean_state and t > 0:
            #        state = prior_means[t]
            #    else:
            #        state = prior_states[t]
            #else:
            #    if self.use_mean_state and t > 0:
            #        state = posterior_means[t]
            #    else:
            #        state = posterior_states[t]
            state = prior_states[t] if obses is None else posterior_states[t]
            state = state if nonterms is None else state * nonterms[t]

            state_action = self.state_action_encoder(
                torch.cat([state, actions[t]], dim=1))
            beliefs[t + 1] = self.rnn(state_action, beliefs[t])

            #if self.learn_state_delta:
            #    mean_offset = prev_state.detach()
            #else:
            #    mean_offset = None

            prior_states[t + 1], prior_means[t + 1], prior_stds[t + 1] = \
                self.forward_prior_distribution(beliefs[t + 1])

            if obses is not None:
                posterior_states[t + 1], posterior_means[t + 1], \
                    posterior_stds[t + 1] = self.forward_posterior_distribution(
                        torch.cat([beliefs[t + 1], obses[t]], dim=1))

        beliefs = [torch.stack(beliefs[1:], dim=0)]
        priors = [
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_stds[1:], dim=0)
        ]

        posteriors = []

        if obses is not None:
            posteriors = [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_stds[1:], dim=0)
            ]

        return beliefs + priors + posteriors
