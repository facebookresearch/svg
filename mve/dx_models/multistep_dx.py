import torch
from torch.nn.functional import softplus
from model_zoo.regression import MaxLikelihoodRegression


class MultistepDx(MaxLikelihoodRegression):
    def __init__(self, input_dim, target_dim, model_class, model_kwargs, mode='prob', obs_dim=None):
        super().__init__(input_dim, target_dim, model_class, model_kwargs, mode)
        self.obs_dim = obs_dim
        if obs_dim is None:
            raise RuntimeError('you must specify the observation dimension')

    def _step(self, obs, actions):
        inputs = torch.cat([obs, actions], dim=-1)
        inputs = (inputs - self.input_mean) / self.input_std
        output = self.model.forward(inputs)
        mean, logvar = output.chunk(2, dim=-1)
        logvar = self.max_logvar - softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + softplus(logvar - self.min_logvar)
        mean = mean * self.target_std + self.target_mean
        var = logvar.exp() * self.target_std ** 2

        if self._deterministic:
            obs_delta = mean[..., 1:]
        else:
            obs_delta = mean[..., 1:] + var[..., 1:].sqrt() * torch.randn_like(var[..., 1:])
        next_obs = (obs + obs_delta).detach()

        return next_obs, mean, var

    def forward(self, obs_actions):
        assert obs_actions.dim() == 3
        _, seq_len, _ = obs_actions.shape
        obses, actions = obs_actions[..., :self.obs_dim], obs_actions[..., self.obs_dim:]
        obs = obses[:, 0]

        means, vars = [], []
        for t in range(seq_len):
            obs, mean, var = self._step(obs, actions[:, t])
            means.append(mean)
            vars.append(mean)

        mean = torch.stack(means, dim=1)
        var = torch.stack(vars, dim=1)
        return mean, var
