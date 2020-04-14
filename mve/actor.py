import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import higher
import hydra

from . import utils


class VanillaActor(nn.Module):
    """Actor network."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)

        return mu, pi, log_pi

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class NormalActor(nn.Module):
    """torch.distributions implementation of an isotropic Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        policy = utils.SquashedNormal(mu, std)
        pi = policy.rsample() if compute_pi else None
        log_pi = policy.log_prob(pi).sum(
            -1, keepdim=True) if compute_log_pi else None

        return policy.mean, pi, log_pi

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class NormalRecActor(nn.Module):
    """torch.distributions implementation of an isotropic Gaussian policy."""
    def __init__(self, obs_dim, action_dim, rec_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        rec_dim = eval(rec_dim)

        self.log_std_bounds = log_std_bounds
        self.rec_dim = rec_dim
        self.trunk = utils.mlp(obs_dim+rec_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, xt, ht=None, compute_pi=True, compute_log_pi=True):
        assert xt.dim() == 2
        n_batch = xt.size(0)

        if ht is None:
            ht = torch.zeros(n_batch, self.rec_dim, device=xt.device)

        assert ht.dim() == 2
        xht = torch.cat((xt, ht), dim=1)

        mu, log_std = self.trunk(xht).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        policy = utils.SquashedNormal(mu, std)
        pi = policy.rsample() if compute_pi else None
        log_pi = policy.log_prob(pi).sum(
            -1, keepdim=True) if compute_log_pi else None

        return policy.mean, pi, log_pi, policy.entropy().sum(-1)

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class MultivariateNormalActor(nn.Module):
    """torch.distributions implementation of an multivariate Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.mean = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.variance = utils.mlp(obs_dim, hidden_dim, action_dim * action_dim,
                                  hidden_depth)
        self.action_dim = action_dim

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mean = self.mean(obs)

        variance = self.variance(obs)
        variance = variance.reshape(-1, self.action_dim, self.action_dim)

        tril = variance.tril(-1) + F.softplus(
            variance.diagonal(dim1=-2, dim2=-1)).diag_embed()

        policy = utils.SquashedMultivariateNormal(mean, scale_tril=tril)

        self.outputs['mu'] = mean
        self.outputs['std'] = tril.diagonal(dim1=-2, dim2=-1)

        pi = policy.rsample() if compute_pi else None
        log_pi = policy.log_prob(pi).unsqueeze(
            dim=1) if compute_log_pi else None

        return policy.mean, pi, log_pi

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.mean):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class OptActor(nn.Module):
    """Actor network."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds, device, horizon_length, opt_cfg):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        assert horizon_length > 0
        self.horizon_length = horizon_length

        if horizon_length > 1:
            raise NotImplementedError

        # TODO: could share trunk with init_net
        self.std_net = utils.mlp(obs_dim, hidden_dim, action_dim,
                                 hidden_depth - 1)

        self.outputs = dict()
        self.apply(utils.weight_init)

        self.opt = hydra.utils.instantiate(opt_cfg)

    def parameters(self):
        return list(self.opt.parameters()) + list(self.std_net.parameters())

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                update_decoder=False):
        mu = self.opt.solve(obs,
                            return_first=True,
                            update_decoder=update_decoder)

        # log_std \in [log_std_min, log_std_max]
        log_std = self.std_net(obs)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def log(self, logger, step):
        pass
        # for k, v in self.outputs.items():
        #     logger.log_histogram('train_actor/%s_hist' % k, v, step)

        # logger.log_param('train_actor/fc1', self.trunk[0], step)
        # logger.log_param('train_actor/fc2', self.trunk[2], step)
        # logger.log_param('train_actor/fc3', self.trunk[4], step)


class GDOpt(nn.Module):
    def __init__(self, n_iter, lr, init_net_type, obs_dim, hidden_dim,
                 action_dim, hidden_depth, n_latent):
        super().__init__()
        self.n_iter = n_iter
        self.lr = lr
        self.n_latent = n_latent

        if n_latent > 0:
            self.decoder = utils.mlp(n_latent,
                                     hidden_dim,
                                     action_dim,
                                     hidden_depth - 1,
                                     output_mod=nn.Tanh())
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),
                                                lr=1e-4)
            init_net_output_sz = n_latent
        else:
            init_net_output_sz = action_dim

        if init_net_type == 'mlp':
            # TODO: Add more config here
            self.init_net = utils.mlp(obs_dim, hidden_dim, init_net_output_sz,
                                      hidden_depth - 1)
        elif init_net_type == 'rnn':
            raise NotImplementedError
        else:
            assert False

    def parameters(self):
        # Intentionally leave decoder out so we can train it for a different objective.
        return self.init_net.parameters()

    def solve(self, obs, return_first=True, update_decoder=False):
        n_batch = obs.size(0)

        z = self.init_net(obs)
        inner_opt = higher.create_diff_optim(
            torch.optim.SGD,
            {'lr': self.lr},
            params=[z],
        )

        for i in range(self.n_iter):
            us = self.decoder(z) if self.n_latent > 0 else z
            Q1, Q2 = self.critic(obs, us)
            Q = torch.min(Q1, Q2)
            z, = inner_opt.step(-Q.sum(), params=[z])

        us = self.decoder(z) if self.n_latent > 0 else z
        if update_decoder:
            self.decoder_opt.zero_grad()
            Q1, Q2 = self.critic(obs, us)
            Q = torch.min(Q1, Q2)
            (-Q).sum().backward(retain_graph=True)
            self.decoder_opt.step()
        return us


class DCEMOpt(nn.Module):
    def __init__(self, n_iter, n_samples, n_elite, tau, init_net_type, obs_dim,
                 hidden_dim, action_dim, hidden_depth, normalize, n_latent):
        super().__init__()
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.tau = tau
        self.normalize = normalize

        self.n_latent = n_latent
        if n_latent > 0:
            self.decoder = utils.mlp(n_latent,
                                     hidden_dim,
                                     action_dim,
                                     hidden_depth - 1,
                                     output_mod=nn.Tanh())
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),
                                                lr=1e-4)
            init_net_output_sz = 2 * n_latent
        else:
            init_net_output_sz = 2 * action_dim

        if init_net_type == 'mlp':
            # TODO: Add more config here
            self.init_net = utils.mlp(obs_dim, hidden_dim, init_net_output_sz,
                                      hidden_depth - 1)
        elif init_net_type == 'rnn':
            raise NotImplementedError
        else:
            assert False

    def parameters(self):
        return self.init_net.parameters()

    def solve(self, obs, return_first=True, update_decoder=False):
        n_batch = obs.size(0)

        init_mu, init_sigma = self.init_net(obs).chunk(2, dim=-1)
        if self.n_latent == 0:
            init_mu = torch.tanh(init_mu)
        init_sigma = F.softplus(init_sigma) + 1e-4

        obs_batch = obs.unsqueeze(1).repeat(1, self.n_samples, 1)

        def f(z):
            us = self.decoder(z) if self.n_latent > 0 else z
            assert us.ndimension() == 3
            Q1, Q2 = self.critic(obs_batch, us)
            Q = torch.min(Q1, Q2)
            vals = -Q
            return vals

        if self.n_latent == 0:
            lb, ub = -1., 1.
        else:
            lb, ub = None, None

        final_z = dcem(
            f,
            nx=init_mu.size(1),
            n_batch=n_batch,
            n_sample=self.n_samples,
            n_elite=self.n_elite,
            n_iter=self.n_iter,
            temp=self.tau,
            normalize=self.normalize,
            lb=lb,
            ub=ub,
            init_mu=init_mu,
            init_sigma=init_sigma,
            device=obs.device,
        )
        final_us = self.decoder(final_z) if self.n_latent > 0 else z
        import sys
        sys.exit(-1)
        # g, = torch.autograd.grad(final_us.sum(), init_mu, retain_graph=True)
        # if torch.any(g != g):
        #     import ipdb; ipdb.set_trace()

        if update_decoder:
            self.decoder_opt.zero_grad()
            Q1, Q2 = self.critic(obs, final_us)
            Q = torch.min(Q1, Q2)
            (-Q).sum().backward(retain_graph=True)
            self.decoder_opt.step()

        return final_us

class SeqActor(nn.Module):
    """Actor network for sequences."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 horizon, log_std_bounds):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.log_std_bounds = log_std_bounds

        # TODO: Could use an RNN
        self.trunk = utils.mlp(
            obs_dim, hidden_dim, 2 * action_dim * horizon, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)


    def sample(self, obs, num_samples):
        assert obs.ndimension() == 2
        _, samples, _, _ = self.forward(obs, num_samples)
        return samples


    def forward(self, obs, num_samples=1, compute_pi=True, compute_log_pi=True):
        assert obs.ndimension() == 2
        n_batch = obs.size(0)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        mu = mu.reshape(n_batch, self.horizon, self.action_dim)
        log_std = log_std.reshape(n_batch, self.horizon, self.action_dim)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + \
          0.5 * (log_std_max - log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn([num_samples]+list(mu.size()), device=mu.device)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)

        mu = mu.transpose(0, 1)
        log_std = log_std.transpose(0, 1)
        if compute_pi:
            pi = pi.reshape(num_samples * n_batch, self.horizon,
                            self.action_dim).transpose(0, 1)
            if compute_log_pi:
                log_pi = log_pi.reshape(num_samples * n_batch,
                                        self.horizon, 1).transpose(0, 1)

        return mu, pi, log_pi, log_std

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class SeqActorLSTM(nn.Module):
    """Actor network for sequences, using an LSTM."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 latent_dim, horizon, log_std_bounds, share_std):
        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.log_std_bounds = log_std_bounds
        self.share_std = share_std

        self.x_enc = utils.mlp(obs_dim, hidden_dim, 2*latent_dim, hidden_depth)
        self.u_dec = utils.mlp(latent_dim, hidden_dim, 2*action_dim, hidden_depth)
        self.lstm = nn.LSTM(latent_dim, latent_dim)

        self.outputs = dict()

        self.x_enc.apply(utils.weight_init)
        self.u_dec.apply(utils.weight_init)



    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.lstm.flatten_parameters()

    def sample(self, obs, num_samples):
        assert obs.ndimension() == 2
        _, samples, _, _ = self.forward(obs, num_samples)
        return samples


    def forward(self, obs, num_samples=1, compute_pi=True, compute_log_pi=True):
        assert obs.ndimension() == 2
        n_batch = obs.size(0)

        # mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # mu = mu.reshape(n_batch, self.horizon, self.action_dim)
        # log_std = log_std.reshape(n_batch, self.horizon, self.action_dim)

        init_hidden = torch.chunk(self.x_enc(obs).unsqueeze(0), 2, dim=2)
        init_hidden = [h.contiguous() for h in init_hidden]
        inputs = torch.zeros(self.horizon, n_batch, self.latent_dim, device=obs.device)
        embs, _ = self.lstm(inputs, init_hidden)
        mu, log_std = self.u_dec(embs).transpose(0, 1).chunk(2, dim=-1)

        if self.share_std:
            # TODO: Is slightly wasteful but should be ok
            log_std[:,:,:] = log_std[:,0,:].unsqueeze(1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + \
          0.5 * (log_std_max - log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn([num_samples]+list(mu.size()), device=mu.device)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)

        mu = mu.transpose(0, 1)
        log_std = log_std.transpose(0, 1)
        if compute_pi:
            pi = pi.reshape(num_samples * n_batch, self.horizon,
                            self.action_dim).transpose(0, 1)
            if compute_log_pi:
                log_pi = log_pi.reshape(num_samples * n_batch,
                                        self.horizon, 1).transpose(0, 1)

        return mu, pi, log_pi, log_std

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        # for i, m in enumerate(self.trunk):
        #     if type(m) == nn.Linear:
        #         logger.log_param(f'train_actor/fc{i}', m, step)
