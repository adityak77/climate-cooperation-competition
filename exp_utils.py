import os
import sys
import warnings
warnings.filterwarnings('ignore')
_ROOT = os.getcwd()
sys.path.append(_ROOT+"/scripts")
sys.path = [os.path.join(_ROOT, "/scripts")] + sys.path

import fileinput
import numpy as np
import random
import torch
from collections import namedtuple
from filelock import FileLock

from train_with_warp_drive import trainer
from desired_outputs import desired_outputs
from opt_helper import save, load, plot_training_curve, plot_result

REGION_YAMLS_DIR = 'region_yamls'

Distribution = namedtuple('Distribution', ['mu', 'sigma'])

def seed_everything(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _initialize_params(defaults, overrides):
    params = defaults.copy()
    params.update(overrides)

    lock = FileLock(os.path.join(REGION_YAMLS_DIR, "trainer.lock"))
    lock.acquire()
    # bad practice, but we release it later in `rice.py`
    print("[INFO] File lock acquired...")

    for i, fname in enumerate(os.listdir(REGION_YAMLS_DIR)):
        if 'yml' not in fname:
            continue

        full_path = os.path.join(REGION_YAMLS_DIR, fname)
        for line in fileinput.input(full_path, inplace=True):
            found = False
            for param in params:
                if line.strip().startswith(param):
                    found = True
                    val = max(0, np.random.normal(params[param].mu, params[param].sigma))
                    print(f'  {param}: {val}')
            if not found:
                print(line.rstrip())

def sample_and_train(exp_name, tag, defaults, overrides={}, seed=None):
    seed_everything(seed)

    print("[INFO] Sampling new parameters...")
    print(f"Training {exp_name}/{tag}")
    print(overrides)
    _initialize_params(defaults, overrides)

    if not os.path.exists(os.path.join("experiments", exp_name)):
        os.makedirs(os.path.join("experiments", exp_name))

    print("[INFO] Starting training...")
    _, outputs_mean, outputs_std, submission_file = trainer(negotiation_on=0,
                   num_envs=500,
                   train_batch_size=10000,
                   num_episodes=500000,
                   lr=0.0005,
                   model_params_save_freq=5000,
                   desired_outputs=desired_outputs,
                   output_all_envs=False)

    print(f"[INFO] Max air temperature: {np.max(outputs_mean['global_temperature'][:, 0])}")

    print("[INFO] Saving results...")
    save({
        "overrides": overrides,
        "mean": outputs_mean,
        "std": outputs_std,
        "submission_file": submission_file,
    }, os.path.join("experiments", exp_name, f"{tag}.pkl"))

def grid_search(exp_name, defaults, param, mu_or_sigma, low, high, amt, overrides={}, seed=None):
    assert mu_or_sigma in ['mu', 'sigma']
    print(f"[INFO] Grid searching {param}'s {mu_or_sigma} from {low} to {high} with {amt} steps")

    steps = np.linspace(low, high, num=amt)
    overrides = overrides.copy()
    for step in steps:
        tag = f"{param}_{mu_or_sigma}_{step}"
        default = defaults[param]
        if mu_or_sigma == 'mu':
            overrides[param] = Distribution(mu=step, sigma=default.sigma)
        else:
            overrides[param] = Distribution(mu=default.mu, sigma=step)
        sample_and_train(exp_name, tag, defaults, overrides, seed=seed)

