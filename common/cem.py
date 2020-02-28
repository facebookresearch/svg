import torch
import utils


class CEMPlanner(object):
    def __init__(self,
                 transition_model,
                 reward_model,
                 plan_horizon,
                 num_iters,
                 num_candidates,
                 topk,
                 action_dim,
                 lb=-1.,
                 ub=1.,
                 **kwargs):
        super().__init__()

        self.transition_model = transition_model
        self.reward_model = reward_model
        self.lb = lb
        self.ub = ub

        self.plan_horizon = plan_horizon
        self.num_iters = num_iters
        self.num_candidates = num_candidates
        self.topk = topk

        self.action_dim = action_dim

    def forward(self,
                belief,
                state,
                return_nominal=False,
                use_mean_state=False):
        B, H = belief.size()
        Z = state.size(1)
        device = state.device
        assert B == 1

        belief = belief.unsqueeze(dim=1).expand(B, self.num_candidates, H)
        belief = belief.reshape(-1, H)

        state = state.unsqueeze(dim=1).expand(B, self.num_candidates, Z)
        state = state.reshape(-1, Z)

        actions_mean = torch.zeros(self.plan_horizon,
                                   B,
                                   1,
                                   self.action_dim,
                                   device=device)
        actions_std = torch.ones(self.plan_horizon,
                                 B,
                                 1,
                                 self.action_dim,
                                 device=device)

        for i in range(self.num_iters):
            noise = torch.randn(self.plan_horizon,
                                B,
                                self.num_candidates,
                                self.action_dim,
                                device=device)
            actions = actions_mean + noise * actions_std
            actions = actions.view(self.plan_horizon, B * self.num_candidates,
                                   self.action_dim)
            # actions = torch.clamp(actions, self.lb, self.ub)
            actions = torch.clamp(actions, -1., 1.)  # TODO

            with utils.eval_mode(self.transition_model, self.reward_model):
                with torch.no_grad():
                    beliefs, (states, means, _), _ = self.transition_model(
                        state, actions, belief, use_mean_state=use_mean_state)
                    if use_mean_state:
                        states = means
                    rewards = self.reward_model(beliefs.view(-1, H),
                                                states.view(-1, Z))

            returns = rewards.view(self.plan_horizon, -1).sum(dim=0)
            returns = returns.reshape(B, self.num_candidates)

            _, topk = returns.topk(self.topk,
                                   dim=1,
                                   largest=True,
                                   sorted=False)
            topk += self.num_candidates * torch.arange(
                0, B, device=device).unsqueeze(dim=1)

            best_actions = actions[:, topk.view(-1)]
            best_actions = best_actions.reshape(self.plan_horizon, B,
                                                self.topk, self.action_dim)

            actions_mean = best_actions.mean(dim=2, keepdim=True)
            actions_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

        if return_nominal:
            return actions_mean
        else:
            return actions_mean[0].squeeze(dim=1)
