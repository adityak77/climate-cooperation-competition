import os
import sys
from glob import glob

import warnings
warnings.filterwarnings('ignore')
_ROOT = os.getcwd()
sys.path.append(_ROOT+"/scripts")
sys.path = [os.path.join(_ROOT, "/scripts")] + sys.path

from desired_outputs import desired_outputs
from importlib import reload
from codecarbon import EmissionsTracker
from opt_helper import save, load, plot_training_curve, plot_result


# tracker = EmissionsTracker()
# tracker.start()

device = 'cpu' # 'gpu'

if device == 'gpu':
    import train_with_warp_drive as gpu_trainer
elif device == 'cpu':
    os.chdir(_ROOT+"/scripts")
    import train_with_rllib as cpu_trainer

nego_off_ts, nego_on_ts = None, None

print('Beginning no negotiation...')
if device == 'gpu':
    trainer_off, nego_off_ts = gpu_trainer.trainer(
            negotiation_on=0, # no negotiation
            num_envs=100, 
            train_batch_size=1024, 
            num_episodes=3000, 
            lr=0.0005,
            model_params_save_freq=5000, 
            desired_outputs=desired_outputs, # a list of values that the simulator will output
            output_all_envs=False # output the mean of all "num_envs" results. Set to True for output all results
        )

# elif device == 'cpu':
#     cpu_trainer = reload(cpu_trainer)
#     trainer_off, nego_off_ts = cpu_trainer.trainer(negotiation_on=0,  # no negotiation
#             num_envs=22, 
#             train_batch_size=1024, 
#             num_episodes=300, 
#             lr=0.0005, 
#             model_params_save_freq=5000, 
#             desired_outputs=desired_outputs, # a list of values that the simulator will output
#             num_workers=22
#         )

print('Beginning negotiation...')
if device == 'gpu':
    trainer_on, nego_on_ts = gpu_trainer.trainer(
            negotiation_on=1, # no negotiation
            num_envs=100, 
            train_batch_size=1024, 
            num_episodes=3000, 
            lr=0.0005,
            model_params_save_freq=5000, 
            desired_outputs=desired_outputs, # a list of values that the simulator will output
            output_all_envs=False # output the mean of all "num_envs" results. Set to True for output all results
        )

elif device == 'cpu':
    cpu_trainer = reload(cpu_trainer)
    trainer_on, nego_on_ts = cpu_trainer.trainer(negotiation_on=1, # with naive negotiation
        num_envs=22, 
        train_batch_size=1024, 
        num_episodes=300, 
        lr=0.0005, 
        model_params_save_freq=5000, 
        desired_outputs=desired_outputs, # a list of values that the simulator will output
        num_workers=22
    )


# saving and loading
# save_path = 'results.pkl'
# print(f'Saving results to {save_path}')
# save({"nego_off":nego_off_ts, "nego_on":nego_on_ts}, save_path)
# dict_ts = load("filename.pkl")
# nego_off_ts, nego_on_ts = dict_ts["nego_off"], dict_ts["nego_on"]

# plotting training results
# log_zip = sorted(glob(os.path.join(_ROOT,"Submissions/*.zip")))[-1]
# print(log_zip)
# plot_training_curve(None, 'Mean episodic reward', log_zip, 'training_results.png')

# to check the raw logging dictionary, uncomment below
# logs = get_training_curve(log_zip)

plot_result("global_temperature",
            "temperature_result_neg_off_modified.png",
            nego_off=None, # change it to cpu_nego_off_ts if using CPU
            nego_on=nego_on_ts, 
            k=0)

# tracker.stop()
