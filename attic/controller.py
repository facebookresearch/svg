import torch

from common import utils


class CEMCtrl(object):
    def __init__(self, action_dim, action_range, discount, horizon, num_iters,
                 num_samples, num_elite):
        super().__init__()
        self.action_dim = action_dim
        self.action_range = action_range
        self.discount = discount

        self.horizon = horizon
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.num_elite = num_elite

    def forward(self,
                transition_model,
                reward_model,
                belief,
                state,
                critic=None,
                return_seq=False):
        B, H = belief.size()
        Z = state.size(1)
        device = state.device
        assert B == 1

        belief = belief.unsqueeze(dim=1).expand(B, self.num_samples, H)
        belief = belief.reshape(-1, H)

        state = state.unsqueeze(dim=1).expand(B, self.num_samples, Z)
        state = state.reshape(-1, Z)

        actions_mean = torch.zeros(self.horizon,
                                   B,
                                   1,
                                   self.action_dim,
                                   device=device)
        actions_std = torch.ones(self.horizon,
                                 B,
                                 1,
                                 self.action_dim,
                                 device=device)

        discounts = torch.logspace(0,
                                   self.horizon - 1,
                                   steps=self.horizon,
                                   base=self.discount,
                                   device=device)
        discounts = discounts.view(self.horizon, 1, 1)

        for i in range(self.num_iters):
            noise = torch.randn(self.horizon,
                                B,
                                self.num_samples,
                                self.action_dim,
                                device=device)
            actions = actions_mean + noise * actions_std
            actions = actions.view(self.horizon, B * self.num_samples,
                                   self.action_dim)
            # TODO: no clamping here migth improve exploration
            #actions = actions.clamp(*self.action_range)

            with utils.eval_mode(transition_model, reward_model):
                with torch.no_grad():
                    beliefs, states, means, _ = transition_model(
                        state, actions, belief)
                    #if transition_model.use_mean_state:
                    #    states = means
                    rewards = reward_model(beliefs.view(-1, H),
                                           states.view(-1, Z))
                    rewards = rewards.view(self.horizon, B, self.num_samples)

            if critic is not None:
                # replace last entry of rewards with a Q-function bootstrap
                with utils.eval_mode(critic):
                    with torch.no_grad():
                        # compute q values for s_{T-2}, a_{T-1}
                        q1, q2 = critic(states[-2], actions[-1])
                        q = torch.min(q1, q2)
                        q = q.view(1, 1, self.num_samples)
                        rewards = torch.cat([rewards[:-1, :, :], q], dim=0)

            # compute discounted sums
            rewards *= discounts
            returns = rewards.sum(dim=0)

            _, num_elite = returns.topk(self.num_elite,
                                        dim=1,
                                        largest=True,
                                        sorted=False)
            num_elite += self.num_samples * torch.arange(
                0, B, device=device).unsqueeze(dim=1)

            best_actions = actions[:, num_elite.view(-1)]
            best_actions = best_actions.reshape(self.horizon, B,
                                                self.num_elite,
                                                self.action_dim)

            actions_mean = best_actions.mean(dim=2, keepdim=True)
            actions_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

        if return_seq:
            return actions_mean
        else:
            return actions_mean[0].squeeze(dim=1)

class SampleCtrl(object):
    def __init__(self, obs_dim, action_dim, action_range, num_samples):
        assert action_range == [-1, 1]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.num_samples = num_samples

    # # @profile
    def forward(self,
                transition_model,
                reward_model,
                obs_model,
                sampler,
                belief,
                state,
                obs,
                critic=None,
                return_seq=False):
        device = belief.device
        batch_mode = belief.ndimension() == 2
        if not batch_mode:
            belief = belief.unsqueeze(0)
            state = state.unsqueeze(0)
        assert belief.ndimension() == 2
        assert obs.ndimension() == 2
        n_batch, belief_dim = belief.size()
        state_dim = state.size(1)

        # TODO: This currently uses the observation, but could alternatively
        # use the state and/or the belief.
        # TODO: Decide what obs to use
        # pred_obs = obs_model(belief, state)
        us = sampler(obs, self.num_samples)
        horizon = us.size(0)
        belief = belief.unsqueeze(1).repeat(
            1, self.num_samples, 1).reshape(self.num_samples * n_batch, -1)
        state = state.unsqueeze(1).repeat(
            1, self.num_samples, 1).reshape(self.num_samples * n_batch, -1)
        obs = obs.unsqueeze(1).repeat(
            1, self.num_samples, 1).reshape(self.num_samples * n_batch, -1)

        rewards = 0
        if horizon > 1 or not critic:
            with utils.eval_mode(transition_model, reward_model):
                with torch.no_grad():
                    belief, state, means, _ = transition_model(state, us, belief)
                    rewards = reward_model(
                        belief.view(-1, belief_dim),
                        state.view(-1, state_dim))

            rewards = rewards.reshape(self.num_samples, n_batch, horizon)

        if critic is None:
            rewards = rewards.sum(dim=-1)
        else:
            # compute q values for s_{T-2}, a_{T-1}
            if horizon > 1:
                rewards = rewards[:, :, :-1].sum(dim=-1)
                last_obs = obs_model(belief[-2], state[-2])

            with utils.eval_mode(critic):
                with torch.no_grad():
                    q1, q2 = critic(last_obs, us[-1])
            q = torch.min(q1, q2).reshape(self.num_samples, n_batch)
            rewards += q

        us = us.transpose(0, 1)
        us = us.reshape(self.num_samples, n_batch, horizon, self.action_dim)

        I = rewards.max(dim=0).indices
        # TODO: Could be done with the right gather
        best_us = []
        for i in range(n_batch):
            best_idx = I[i]
            best_us.append(us[best_idx, i])
        best_us = torch.stack(best_us).transpose(0, 1)

        if not batch_mode:
            best_us = best_us.squeeze(1)

        if not return_seq:
            best_us = best_us[0]

        return best_us

class SampleCtrlModel(object):
    def __init__(self, obs_dim, action_dim, action_range, num_samples):
        assert action_range == [-1, 1]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.num_samples = num_samples

    # # @profile
    def forward(self,
                model,
                policy,
                obs,
                critic=None,
                return_seq=False):
        device = obs.device
        batch_mode = obs.ndimension() == 2
        if not batch_mode:
            obs = obs.unsqueeze(0)
        assert obs.ndimension() == 2
        n_batch = obs.size(0)

        horizon = model.horizon
        obs = obs.unsqueeze(1).repeat(
            1, self.num_samples, 1).reshape(self.num_samples * n_batch, -1)

        rewards = 0
        if horizon > 1 or not critic:
            us, _, _, pred_obs = model.unroll_policy(obs, policy, sample=True, last_u=True)
            rewards = model.rew(pred_obs)
            rewards = rewards.reshape(horizon, self.num_samples, n_batch).permute(1,2,0)

        if critic is None:
            rewards = rewards.sum(dim=-1)
        else:
            if horizon > 1:
                rewards = rewards[:, :, :-1].sum(dim=-1)

            with utils.eval_mode(critic):
                with torch.no_grad():
                    q1, q2 = critic(pred_obs[-1], us[-1])
            q = torch.min(q1, q2).reshape(self.num_samples, n_batch)
            rewards += q

        us = us.transpose(0, 1)
        us = us.reshape(self.num_samples, n_batch, horizon+1, self.action_dim)

        I = rewards.max(dim=0).indices
        # TODO: Could be done with the right gather
        best_us = []
        for i in range(n_batch):
            best_idx = I[i]
            best_us.append(us[best_idx, i])
        best_us = torch.stack(best_us).transpose(0, 1)

        if not batch_mode:
            best_us = best_us.squeeze(1)

        if not return_seq:
            best_us = best_us[0]

        return best_us
