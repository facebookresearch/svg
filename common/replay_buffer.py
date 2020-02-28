import numpy as np
import torch
import os

from common import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.mean_obs_np = self.std_obs_np = None
        self.welford = utils.Welford()

    def __len__(self):
        return self.capacity if self.full else self.idx

    def normalize_obs(self):
        MIN_STD = 1e-1
        MAX_STD = 10
        # i = self.capacity if self.full else self.idx
        self.mean_obs_np = self.welford.mean()
        self.std_obs_np = self.welford.std()
        self.std_obs_np[self.std_obs_np < MIN_STD] = MIN_STD
        self.std_obs_np[self.std_obs_np > MAX_STD] = MAX_STD
        self.obses = (self.obses - self.mean_obs_np) / self.std_obs_np
        self.next_obses = (self.next_obses - self.mean_obs_np) / self.std_obs_np

    def renormalize_obs(self):
        if self.mean_obs_np is not None:
            self.obses = (self.obses * self.std_obs_np) + self.mean_obs_np
            self.next_obses = (self.next_obses * self.std_obs_np) + self.mean_obs_np
        self.normalize_obs()

    def get_obs_stats(self):
        if self.mean_obs_np is None:
            return None, None
        else:
            obs_mean = torch.from_numpy(self.mean_obs_np).to(self.device).float()
            obs_std = torch.from_numpy(self.std_obs_np).to(self.device).float()
            return obs_mean, obs_std

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.welford.add_data(obs)
        if self.mean_obs_np is not None:
            obs = (obs - self.mean_obs_np) / self.std_obs_np
            next_obs = (next_obs - self.mean_obs_np) / self.std_obs_np
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_multistep(self, batch_size, T):
        assert batch_size < self.idx or self.full

        # The sampling here could be improved, and may cause an infinite
        # loop if T is too large.
        last_idx = self.capacity if self.full else self.idx
        last_idx -= T
        idxs = []
        while len(idxs) < batch_size:
            i = np.random.randint(0, last_idx)
            if i in idxs:
                continue
            if np.all(self.not_dones[i:i + T] == 1.):
                idxs.append(i)
        idxs = np.array(idxs)

        obses, actions, rewards = [], [], []

        for t in range(T):
            obses.append(self.obses[idxs + t])
            actions.append(self.actions[idxs + t])
            rewards.append(self.rewards[idxs + t])

        obses = np.stack(obses)
        actions = np.stack(actions)
        rewards = np.stack(rewards).squeeze(2)

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)

        return obses, actions, rewards

    def save(self, save_dir):
        assert not self.full # Unimplemented
        if self.idx == self.last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

        self.last_save = self.idx
