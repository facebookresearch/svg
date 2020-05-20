import os
import pickle as pkl
from pathlib import Path
from copy import deepcopy
import hydra

import numpy as np
import torch

from model_zoo.utils.data import SeqDataset, format_seqs


def split_buffer(replay_buffer, max_transitions, subsample_rate, min_seq_len=1, max_seq_len=None):
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

        if not_done == 0 or (max_seq_len and len(input_seq) == max_seq_len):
            p1, p2 = np.random.rand(), np.random.rand()
            if len(input_seq) < min_seq_len:
                pass
            elif p1 > subsample_rate:
                pass
            elif p2 < 0.1:
                test_seqs.append((np.concatenate(input_seq), np.concatenate(target_seq)))
            else:
                train_seqs.append((np.concatenate(input_seq), np.concatenate(target_seq)))
            input_seq, target_seq = [], []

    print(f"{n} transitions")
    print(f"{len(train_seqs)} training sequences")
    print(f"{len(test_seqs)} test sequences")

    return train_seqs, test_seqs


def simulate_rl_run(model, dataset, fit_params, train_seqs, test_data, sim_params):
    eval_freq = len(train_seqs) // sim_params['num_evals']
    run_metrics = dict(train_loss=[], num_train=[],
                       holdout_loss=[], holdout_mse=[],
                       validation_loss=[], validation_mse=[],
                       test_loss=[], test_mse=[])
    model_checkpoints = []
    num_train = 0
    for i, seq_pair in enumerate(train_seqs):
        dataset.add_seq(*seq_pair)
        num_train += seq_pair[0].shape[0]
        if (i + 1) % eval_freq == 0:
            run_metrics['num_train'].append(num_train)
            fit_metrics = model.fit(dataset, fit_params)

            validation_data = format_seqs(dataset.holdout_input_seqs, dataset.holdout_target_seqs,
                                       sim_params.test_seq_len, 'sequential')
            validation_metrics = model.validate(*validation_data)
            eval_metrics = model.validate(*test_data)

            run_metrics['train_loss'].append(fit_metrics['train_loss'][-1])
            run_metrics['holdout_loss'].append(fit_metrics['holdout_loss'][-1])
            run_metrics['holdout_mse'].append(fit_metrics['holdout_mse'])
            run_metrics['validation_loss'].append(validation_metrics['val_loss'])
            run_metrics['validation_mse'].append(validation_metrics['val_mse'])
            run_metrics['test_loss'].append(eval_metrics['val_loss'])
            run_metrics['test_mse'].append(eval_metrics['val_mse'])

            model_checkpoints.append(deepcopy(model.state_dict()))

    return run_metrics, model_checkpoints


def experiment(ckpt, cfg):
    min_seq_len = max(cfg.sim_params.train_seq_len, cfg.sim_params.test_seq_len)
    assert min_seq_len < cfg.sim_params.max_seq_len

    replay_buffer = ckpt.replay_buffer
    env = ckpt.env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    cfg.reg_model.params['input_dim'] = obs_dim + action_dim
    cfg.reg_model.params['target_dim'] = 1 + obs_dim

    results = []
    for trial in range(cfg.num_seeds):
        print(f"--- TRIAL {trial + 1} ---")
        # format buffer data
        dataset = SeqDataset(cfg.sim_params.train_seq_len, holdout_ratio=cfg.holdout_ratio)
        train_seqs, test_seqs = split_buffer(replay_buffer, cfg.max_transitions, cfg.subsample_rate,
                                             min_seq_len, cfg.sim_params.max_seq_len)
        test_input_seqs = [seq_pair[0] for seq_pair in test_seqs]
        test_target_seqs = [seq_pair[1] for seq_pair in test_seqs]
        # import pdb; pdb.set_trace()
        test_data = format_seqs(test_input_seqs, test_target_seqs, cfg.sim_params.test_seq_len, 'sequential')

        cfg.reg_model.params.obs_dim = obs_dim
        model = hydra.utils.instantiate(cfg.reg_model)

        trial_metrics, _ = simulate_rl_run(model, dataset, cfg.network.fit_params, train_seqs, test_data, cfg.sim_params)
        results.append(trial_metrics)
    results = {key: np.stack([d[key] for d in results]) for key in results[0].keys()}

    return results


@hydra.main(config_path='../config/sim_rl_run/main.yaml', strict=True)
def main(cfg):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # set up checkpoint, log paths
    root_dir = Path(hydra.utils.get_original_cwd()) / 'exp'
    ckpt_dir = root_dir / cfg.ckpt_dir
    log_dir = root_dir / 'sim_rl_run' / cfg.log_dir
    if not os.path.exists(log_dir.as_posix()):
        os.makedirs(log_dir.as_posix())
    print(f"checkpoint dir: {ckpt_dir.as_posix()}")
    print(f"log dir: {log_dir.as_posix()}")

    # load checkpoint
    ckpt_files = list(ckpt_dir.rglob('latest.pkl'))
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
