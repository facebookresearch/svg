import os
import pickle as pkl
from pathlib import Path
from copy import deepcopy
import hydra

import numpy as np
import torch

import model_zoo
from model_zoo.regression import MaxLikelihoodRegression
from model_zoo.utils.data import SeqDataset, format_seqs


def split_buffer(replay_buffer, max_transitions):
    field_names = ['obses', 'actions', 'rewards', 'next_obses', 'not_dones', 'not_dones_no_max']
    buffer_data = [getattr(replay_buffer, field) for field in field_names]

    input_seq, target_seq = [], []
    train_seqs, test_seqs = [], []
    max_transitions = len(replay_buffer) if max_transitions is None else max_transitions
    n = min(len(replay_buffer), max_transitions)
    for i in range(n):
        transition = [array[i:i + 1] for array in buffer_data]
        not_done = transition[-2]
        input_seq.append(np.concatenate(transition[:2], axis=1))
        target_seq.append(np.concatenate([transition[2].reshape(1, 1), transition[3] - transition[0]], axis=1))

        if not_done == 0:
            p = np.random.rand()
            if p < 0.1:
                test_seqs.append((np.concatenate(input_seq), np.concatenate(target_seq)))
            else:
                train_seqs.append((np.concatenate(input_seq), np.concatenate(target_seq)))
            input_seq, target_seq = [], []

    print(f"{n} transitions")
    print(f"{len(train_seqs)} training sequences")
    print(f"{len(test_seqs)} test sequences")

    return train_seqs, test_seqs


def simulate_rl_run(model, dataset, fit_params, train_seqs, test_data, num_evals):
    eval_freq = len(train_seqs) // num_evals
    run_metrics = dict(holdout_loss=[], holdout_mse=[],
                       test_loss=[], test_mse=[], num_train=[])
    model_checkpoints = []
    num_train = 0
    for i, seq_pair in enumerate(train_seqs):
        dataset.add_seq(*seq_pair)
        num_train += seq_pair[0].shape[0]
        if (i + 1) % eval_freq == 0:
            fit_metrics = model.fit(dataset, fit_params)
            eval_metrics = model.validate(*test_data)
            run_metrics['holdout_loss'].append(fit_metrics['holdout_loss'][-1])
            run_metrics['holdout_mse'].append(fit_metrics['holdout_mse'])
            run_metrics['test_loss'].append(eval_metrics['val_loss'])
            run_metrics['test_mse'].append(eval_metrics['val_mse'])
            run_metrics['num_train'].append(num_train)
            model_checkpoints.append(deepcopy(model.state_dict()))

    return run_metrics, model_checkpoints


def experiment(ckpt, cfg):
    env = ckpt.env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = obs_dim + action_dim
    target_dim = 1 + obs_dim
    replay_buffer = ckpt.replay_buffer

    network_class = getattr(model_zoo.architecture, cfg.network.name)
    network_kwargs = dict(cfg.network.kwargs)
    det_model = (cfg.model_type == 'det')

    results = []
    for trial in range(cfg.num_seeds):
        print(f"--- TRIAL {trial + 1} ---")
        # format buffer data
        dataset = SeqDataset(train_seq_len=cfg.seq_len, holdout_ratio=0.1)
        train_seqs, test_seqs = split_buffer(replay_buffer, max_transitions=cfg.max_transitions)
        test_input_seqs = [seq_pair[0] for seq_pair in test_seqs]
        test_target_seqs = [seq_pair[1] for seq_pair in test_seqs]
        test_data = format_seqs(test_input_seqs, test_target_seqs, cfg.seq_len, 'sequential')

        # create model, run trial
        model = MaxLikelihoodRegression(input_dim, target_dim, network_class,
                                        network_kwargs, deterministic=det_model)
        trial_metrics, _ = simulate_rl_run(model, dataset, cfg.network.fit_params, train_seqs, test_data, cfg.num_evals)
        results.append(trial_metrics)
    results = {key: np.stack([d[key] for d in results]) for key in results[0].keys()}

    return results


@hydra.main(config_path='../config/sim_rl_run/main.yaml', strict=True)
def main(cfg):
    # set up checkpoint, log paths
    root_dir = Path(hydra.utils.get_original_cwd()) / 'exp'
    ckpt_dir = root_dir / cfg.ckpt_dir
    log_dir = root_dir / 'sim_rl_run' / cfg.log_dir
    if not os.path.exists(log_dir.as_posix()):
        os.makedirs(log_dir.as_posix())
    print(f"checkpoint dir: {ckpt_dir.as_posix()}")
    print(f"log dir: {log_dir.as_posix()}")

    # load checkpoint
    ckpt_files = list(ckpt_dir.rglob('*.pkl'))
    if len(ckpt_files) < 1:
        raise RuntimeError("no checkpoints found")
    else:
        print(f"{len(ckpt_files)} checkpoints found")
    with open(ckpt_files[-1].as_posix(), 'rb') as f:
        ckpt = pkl.load(f)

    results = experiment(ckpt, cfg)
    torch.save(results, (log_dir / 'results.dat').as_posix())
    torch.save(dict(cfg), log_dir / 'cfg.yaml')


if __name__ == '__main__':
    main()
