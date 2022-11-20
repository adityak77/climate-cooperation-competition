import os
import sys

import warnings
warnings.filterwarnings('ignore')
_ROOT = os.getcwd()
sys.path.append(_ROOT+"/scripts")
sys.path = [os.path.join(_ROOT, "/scripts")] + sys.path

from desired_outputs import desired_outputs
from importlib import reload
from codecarbon import EmissionsTracker

import train_with_warp_drive as gpu_trainer

tracker = EmissionsTracker()
tracker.start()

gpu_trainer_off, gpu_nego_off_ts = gpu_trainer.trainer(
        negotiation_on=1, # no negotiation
        num_envs=100, 
        train_batch_size=1024, 
        num_episodes=3000, 
        lr=0.0005,
        model_params_save_freq=5000, 
        desired_outputs=desired_outputs, # a list of values that the simulator will output
        output_all_envs=False # output the mean of all "num_envs" results. Set to True for output all results
    )

tracker.stop()
