# On the model-based stochastic value gradient for continuous reinforcement learning

This repository is by
[Brandon Amos](http://bamos.github.io),
[Samuel Stanton](https://samuelstanton.github.io),
[Denis Yarats](https://cs.nyu.edu/~dy1042/),
and
[Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/)
and contains the PyTorch source code to reproduce the
experiments in our [L4DC 2021](https://l4dc.ethz.ch/) paper
[On model-based stochastic value gradient for continuous reinforcement learning](https://arxiv.org/abs/2008.12775).
Videos of our agents are available
[here](https://sites.google.com/view/2020-svg).

# Setup and dependencies
After cloning this repository and installing PyTorch
on your system, you can set up the code with:

```bash
python3 setup.py develop
```

# A basic run and analysis
You can start a single local run on the humanoid with:

```bash
./train.py env=mbpo_humanoid
```

This will create an experiment directory in
`exp/local/<date>/` with models and logging info.
Once that has saved out the first model,
you can plot a video of the agent with some diagnostic
information with the command:

```bash
./eval-vis-model.py exp/local/2021.05.07
```

# Reproducing our main experimental results
We have the default hyper-parameters in this repo set
to the best ones we found with a hyper-parameter search.
The following command reproduces our final results using 10
seeds with the optimal hyper-parameter:

```bash
./train.py -m experiment=mbpo_final env=mbpo_cheetah,mbpo_hopper,mbpo_walker2d,mbpo_humanoid,mbpo_ant seed=$(seq -s, 10)
```

The results from this experiment can be plotted with
our notebook [nbs/mbpo.ipynb](./nbs/mbpo.ipynb), which can
also serve as a starting point for analyzing and
developing further methods.

# Reproducing our sweeps and ablations
Our main hyper-parameter sweeps are run with hydra's
multi-tasking mode and can be launched with the following
command after uncommenting the `hydra/sweeper` line in
`config/train.yaml`:

```bash
./train.py -m experiment=full_poplin_sweep
```

The results from this experiment can be plotted with
our notebook [nbs/poplin.ipynb](./nbs/poplin.ipynb).

# Citations
If you find this repository helpful for your publications,
please consider citing our paper:

```
@inproceedings{amos2021svg,
  title={On the model-based stochastic value gradient for continuous reinforcement learning},
  author={Amos, Brandon and Stanton, Samuel and Yarats, Denis and Wilson, Andrew Gordon},
  booktitle={L4DC},
  year={2021}
}
```

# Licensing
This repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
