import numpy as np
import torch
import os

import pickle as pkl

from common import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, normalize_obs):
        self.capacity = capacity
        self.device = device

        self.pixels = len(obs_shape) > 1
        obs_dtype = np.float32 if not self.pixels else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.global_idx = 0
        self.global_last_save = 0
        self.full = False
        self.payload = []

        self.normalize_obs = normalize_obs

        if normalize_obs:
            assert not self.pixels
            self.welford = utils.Welford()

    def __len__(self):
        return self.capacity if self.full else self.idx

    def get_obs_stats(self):
        assert not self.pixels
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std[std < MIN_STD] = MIN_STD
        std[std > MAX_STD] = MAX_STD
        return mean, std

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        # For saving
        self.payload.append((
            obs.copy(), next_obs.copy(),
            action.copy(), reward,
            done, done_no_max
        ))

        if self.normalize_obs:
            self.welford.add_data(obs)

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma
            next_obses = (next_obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device).float()
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

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)

        return obses, actions, rewards

    def save(self, save_dir):
        # TODO: The serialization code and logic can be significantly improved.

        if self.global_idx == self.global_last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(
            save_dir, f'{self.global_last_save:08d}_{self.global_idx:08d}.pt')

        payload = list(zip(*self.payload))
        payload = [np.vstack(x) for x in payload]
        self.global_last_save = self.global_idx
        torch.save(payload, path)
        self.payload = []

        path = os.path.join(save_dir, 'stats.pkl')
        if not self.pixels:
            payload = (self.welford)
            with open(path, 'wb') as f:
                pkl.dump(payload, f)

    def load(self, save_dir):
        def parse_chunk(chunk):
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            return (start, end)

        if not self.pixels:
            path = os.path.join(save_dir, 'stats.pkl')
            with open(path, 'rb') as f:
                self.welford = pkl.load(f)

        chunks = os.listdir(save_dir)
        chunks = filter(lambda fname: 'stats' not in fname, chunks)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))

        _, self.global_idx = parse_chunk(chunks[-1])
        self.full = self.global_idx > self.capacity
        global_beginning = self.global_idx - self.capacity if self.full else 0

        # Load and re-normalize.
        for chunk in chunks:
            global_start, global_end = parse_chunk(chunk)
            start = global_start - global_beginning
            end = global_end - global_beginning
            if end <= 0:
                continue

            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            if start < 0:
                payload = [x[-start:] for x in payload]
                start = 0
            assert self.idx == start

            obses = payload[0]
            next_obses = payload[1]

            self.obses[start:end] = obses
            self.next_obses[start:end] = next_obses
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.not_dones_no_max[start:end] = payload[5]
            self.idx = end

        self.last_save = self.idx

        if self.full:
            assert self.idx == self.capacity
            self.idx = 0

