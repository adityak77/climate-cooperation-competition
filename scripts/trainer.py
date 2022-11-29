import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from gym.spaces import Discrete, MultiDiscrete
from torch import nn
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.models.fully_connected import FullyConnected
from warp_drive.training.trainer import Trainer
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.param_scheduler import ParamScheduler
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants

_ROOT_DIR = get_project_root()

_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_COMBINED = "combined"
_EPSILON = 1e-10  # small number to prevent indeterminate divisions


class PPODataset(Dataset):
    def __init__(self, actions, rewards, done_flags, probabilities, processed_obs):
        self.num_envs = actions.shape[1]
        self.actions = actions
        self.rewards = rewards
        self.done_flags = done_flags
        self.probs = probabilities
        self.processed_obs = processed_obs

    def __getitem__(self, idx):
        return {
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'done_flags': self.done_flags[idx],
            'probabilities': [prob[idx] for prob in self.probs],
            'processed_obs': self.processed_obs[idx],
        }

    def __len__(self):
        return self.num_envs


# sigh
class PatchedPPO:
    """
    The Proximal Policy Optimization Class
    https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        discount_factor_gamma=1.0,
        clip_param=0.1,
        normalize_advantage=False,
        normalize_return=False,
        vf_loss_coeff=0.01,
        entropy_coeff=0.01,
    ):
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma
        assert 0 <= clip_param <= 1
        self.clip_param = clip_param
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        # Create vf_loss and entropy coefficient schedules
        self.vf_loss_coeff_schedule = ParamScheduler(vf_loss_coeff)
        self.entropy_coeff_schedule = ParamScheduler(entropy_coeff)

    def compute_loss_and_metrics(
        self,
        timestep=None,
        actions_batch=None,
        rewards_batch=None,
        done_flags_batch=None,
        action_probabilities_batch=None,
        value_functions_batch=None,
        old_logprob=None,
        perform_logging=False,
    ):
        assert timestep is not None
        assert actions_batch is not None
        assert rewards_batch is not None
        assert done_flags_batch is not None
        assert action_probabilities_batch is not None
        assert value_functions_batch is not None
        assert old_logprob is not None

        # Detach value_functions_batch from the computation graph
        # for return and advantage computations.
        value_functions_batch_detached = value_functions_batch.detach()

        # Value objective.
        returns_batch = torch.zeros_like(rewards_batch)

        returns_batch[:, -1, :] = (
            done_flags_batch[:, -1, None] * rewards_batch[:, -1, :]
            + (1 - done_flags_batch[:, -1, None]) * value_functions_batch_detached[:, -1, :]
        )
        for step in range(-2, -returns_batch.shape[1] - 1, -1):
            future_return = (
                done_flags_batch[:, step, None] * torch.zeros_like(rewards_batch[:, step, :])
                + (1 - done_flags_batch[:, step, None])
                * self.discount_factor_gamma
                * returns_batch[:, step + 1, :]
            )
            returns_batch[:, step, :] = rewards_batch[:, step, :] + future_return

        # Normalize across the agents and env dimensions
        if self.normalize_return:
            normalized_returns_batch = (
                returns_batch - returns_batch.mean(dim=(0, 2), keepdim=True)
            ) / (returns_batch.std(dim=(0, 2), keepdim=True) + torch.tensor(_EPSILON))
        else:
            normalized_returns_batch = returns_batch

        vf_loss = nn.MSELoss()(normalized_returns_batch, value_functions_batch)

        # Policy objective
        advantages_batch = normalized_returns_batch - value_functions_batch_detached

        # Normalize across the agents and env dimensions
        if self.normalize_advantage:
            normalized_advantages_batch = (
                advantages_batch - advantages_batch.mean(dim=(0, 2), keepdim=True)
            ) / (
                advantages_batch.std(dim=(0, 2), keepdim=True) + torch.tensor(_EPSILON)
            )
        else:
            normalized_advantages_batch = advantages_batch

        log_prob = 0.0
        mean_entropy = 0.0
        for idx in range(actions_batch.shape[-1]):
            m = Categorical(action_probabilities_batch[idx])
            mean_entropy += m.entropy().mean()
            log_prob += m.log_prob(actions_batch[..., idx])

        ratio = torch.exp(log_prob - old_logprob)

        surr1 = ratio * normalized_advantages_batch
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * normalized_advantages_batch
        )
        policy_surr = torch.minimum(surr1, surr2)
        policy_loss = -1.0 * policy_surr.mean()

        # Total loss
        vf_loss_coeff_t = self.vf_loss_coeff_schedule.get_param_value(timestep)
        entropy_coeff_t = self.entropy_coeff_schedule.get_param_value(timestep)
        loss = policy_loss + vf_loss_coeff_t * vf_loss - entropy_coeff_t * mean_entropy

        variance_explained = max(
            torch.tensor(-1.0),
            (
                1
                - (
                    normalized_advantages_batch.detach().var()
                    / (normalized_returns_batch.detach().var() + torch.tensor(_EPSILON))
                )
            ),
        )

        # Approx KL divergence for early stopping
        approx_kl_div = torch.mean((ratio - 1) - torch.log(ratio)).item()

        if perform_logging:
            metrics = {
                "VF loss coefficient": vf_loss_coeff_t,
                "Entropy coefficient": entropy_coeff_t,
                "Total loss": loss.item(),
                "Policy loss": policy_loss.item(),
                "Value function loss": vf_loss.item(),
                "Mean rewards": rewards_batch.mean().item(),
                "Max. rewards": rewards_batch.max().item(),
                "Min. rewards": rewards_batch.min().item(),
                "Mean value function": value_functions_batch.mean().item(),
                "Mean advantages": advantages_batch.mean().item(),
                "Mean (norm.) advantages": normalized_advantages_batch.mean().item(),
                "Mean (discounted) returns": returns_batch.mean().item(),
                "Mean normalized returns": normalized_returns_batch.mean().item(),
                "Mean entropy": mean_entropy.item(),
                "Variance explained by the value function": variance_explained.item(),
            }
            # mean of the standard deviation of sampled actions
            std_over_agent_per_action = (
                actions_batch.float().std(axis=2).mean(axis=(0, 1))
            )
            std_over_time_per_action = (
                actions_batch.float().std(axis=1).mean(axis=(0, 1))
            )
            std_over_env_per_action = (
                actions_batch.float().std(axis=0).mean(axis=(0, 1))
            )
            for idx, _ in enumerate(std_over_agent_per_action):
                std_action = {
                    f"Std. of action_{idx} over agents": std_over_agent_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over envs": std_over_env_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over time": std_over_time_per_action[
                        idx
                    ].item(),
                }
                metrics.update(std_action)
        else:
            metrics = {}
        return loss, metrics, approx_kl_div


class PatchedTrainer(Trainer):
    def _initialize_policy_algorithm(self, policy):
        algorithm = self._get_config(["policy", policy, "algorithm"])
        assert algorithm == "PPO", "[PatchedTrainer] only works for PPO, use [Trainer] otherwise"

        entropy_coeff = self._get_config(["policy", policy, "entropy_coeff"])
        vf_loss_coeff = self._get_config(["policy", policy, "vf_loss_coeff"])
        self.clip_grad_norm[policy] = self._get_config(
            ["policy", policy, "clip_grad_norm"]
        )
        if self.clip_grad_norm[policy]:
            self.max_grad_norm[policy] = self._get_config(
                ["policy", policy, "max_grad_norm"]
            )
        normalize_advantage = self._get_config(
            ["policy", policy, "normalize_advantage"]
        )
        normalize_return = self._get_config(["policy", policy, "normalize_return"])
        gamma = self._get_config(["policy", policy, "gamma"])

        # Proximal Policy Optimization
        clip_param = self._get_config(["policy", policy, "clip_param"])
        self.trainers[policy] = PatchedPPO(
            discount_factor_gamma=gamma,
            clip_param=clip_param,
            normalize_advantage=normalize_advantage,
            normalize_return=normalize_return,
            vf_loss_coeff=vf_loss_coeff,
            entropy_coeff=entropy_coeff,
        )
        logging.info(f"Initializing the PPO trainer for policy {policy}")

    # I am sad that I have to write this
    def _update_model_params(self, iteration):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            iteration % self.config["saving"]["metrics_log_freq"] == 0
            or iteration == self.num_iters - 1
        )

        metrics_dict = {}

        # Fetch the actions and rewards batches for all agents
        if not self.create_separate_placeholders_for_each_policy:
            all_actions_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_ACTIONS}_batch"
                )
            )
            all_rewards_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_batch"
                )
            )
        done_flags_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            f"{_DONE_FLAGS}_batch"
        )
        # On the device, observations_batch, actions_batch,
        # rewards_batch are all shaped
        # (batch_size, num_envs, num_agents, *feature_dim).
        # done_flags_batch is shaped (batch_size, num_envs)
        # Perform training sequentially for each policy
        for policy in self.policies_to_train:
            # Get policy params
            num_sgd_iter = self._get_config(["policy", policy, "num_sgd_iter"])
            sgd_minibatch_size = self._get_config(["policy", policy, "sgd_minibatch_size"])
            kl_target = self._get_config(["policy", policy, "kl_target"])

            if self.create_separate_placeholders_for_each_policy:
                actions_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_ACTIONS}_batch_{policy}"
                    )
                )
                rewards_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_REWARDS}_batch_{policy}"
                    )
                )
            else:
                # Filter the actions and rewards only for the agents
                # corresponding to a particular policy
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                actions_batch = all_actions_batch[:, :, agent_ids_for_policy, :]
                rewards_batch = all_rewards_batch[:, :, agent_ids_for_policy]

            # Fetch the (processed) observations batch to pass through the model
            processed_obs_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_PROCESSED_OBSERVATIONS}_batch_{policy}"
                )
            )

            # Policy evaluation for the entire batch
            probabilities_batch, value_functions_batch = self.models[policy](
                obs=processed_obs_batch
            )

            # I am only slightly concerned about this
            actions_batch = actions_batch.transpose(0, 1)
            rewards_batch = rewards_batch.transpose(0, 1)
            done_flags_batch = done_flags_batch.transpose(0, 1)
            probabilities_batch = [probs.transpose(0, 1) for probs in probabilities_batch]
            processed_obs_batch = processed_obs_batch.transpose(0, 1)

            # Big brain time
            steps_per_env = actions_batch.shape[1]
            envs_per_batch = round(sgd_minibatch_size / steps_per_env)
            dataset = PPODataset(
                actions_batch,
                rewards_batch,
                done_flags_batch,
                probabilities_batch,
                processed_obs_batch,
            )
            dataloader = DataLoader(dataset, batch_size=envs_per_batch, shuffle=True)

            early_stopped = False
            print("trn")
            # self.current_timestep[policy] += self.training_batch_size
            for epoch in range(num_sgd_iter):
                for batch in dataloader:
                    actions_batch = batch['actions']
                    rewards_batch = batch['rewards']
                    done_flags_batch = batch['done_flags']
                    old_probabilities_batch = batch['probabilities']
                    processed_obs_batch = batch['processed_obs']

                    log_prob = 0.0
                    mean_entropy = 0.0
                    for idx in range(actions_batch.shape[-1]):
                        m = Categorical(old_probabilities_batch[idx])
                        log_prob += m.log_prob(actions_batch[..., idx])

                    old_logprob = log_prob.detach()

                    # HACK! We do this because action masks are tracked in the model and don't
                    # respond to minibatching attempts. Oof... Incidentally, this also calls into
                    # question the validity of training negotiation on with GPU
                    assert self.models[policy].action_mask is None or self.models[policy].action_mask.all()
                    self.models[policy].action_mask = None

                    # Feed minibatch into model
                    probabilities_batch, value_functions_batch = self.models[policy](
                        obs=processed_obs_batch
                    )

                    # Loss and metrics computation
                    loss, metrics, approx_kl_div = self.trainers[policy].compute_loss_and_metrics(
                        self.current_timestep[policy],
                        actions_batch,
                        rewards_batch,
                        done_flags_batch,
                        probabilities_batch,
                        value_functions_batch,
                        old_logprob,
                        perform_logging=logging_flag,
                    )

                    if approx_kl_div > 1.5 * kl_target:
                        early_stopped = True
                        break

                    # Compute the gradient norm
                    grad_norm = 0.0
                    for param in list(
                        filter(lambda p: p.grad is not None, self.models[policy].parameters())
                    ):
                        grad_norm += param.grad.data.norm(2).item()

                    # Update the timestep and learning rate based on the schedule
                    # self.current_timestep[policy] += self.training_batch_size
                    self.current_timestep[policy] += envs_per_batch * steps_per_env
                    lr = self.lr_schedules[policy].get_param_value(
                        self.current_timestep[policy]
                    )
                    for param_group in self.optimizers[policy].param_groups:
                        param_group["lr"] = lr

                    # Loss backpropagation and optimization step
                    self.optimizers[policy].zero_grad()
                    loss.backward()

                    if self.clip_grad_norm[policy]:
                        nn.utils.clip_grad_norm_(
                            self.models[policy].parameters(), self.max_grad_norm[policy]
                        )

                    self.optimizers[policy].step()

                if early_stopped:
                    break

            # Logging
            if logging_flag:
                metrics_dict[policy] = metrics
                # Update the metrics dictionary
                metrics_dict[policy].update(
                    {
                        "Current timestep": self.current_timestep[policy],
                        "Gradient norm": grad_norm,
                        "Learning rate": lr,
                        "Mean episodic reward": self.episodic_reward_sum[policy].item()
                        / (self.num_completed_episodes[policy] + _EPSILON),
                    }
                )

                # Reset sum and counter
                self.episodic_reward_sum[policy] = (
                    torch.tensor(0).type(torch.float32).cuda()
                )
                self.num_completed_episodes[policy] = 0

        print("done")

        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000
        return metrics_dict
