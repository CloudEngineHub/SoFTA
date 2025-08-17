import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.ppo.ppo import PPO

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

class PPOMultiActorCritic(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

        self.keys = self.env.config.robot.get('body_keys', ['lower_body', 'upper_body'])
        self.actor_deploy_freq = self.config.get('actor_deploy_freq', {
            "upper_body": 1,  
            "lower_body": 1   
        })
        self.actor_step_counters = {
            "upper_body": 0,
            "lower_body": 0
        }
        self.std_threshold = self.config.get('std_threshold', {
            "upper_body": 0.0,
            "lower_body": 0.0
        })
        self.previous_actions = {}
        self.previous_rewards = {}
        self.previous_dones = {}
        self.previous_obs_sict = {}
        
        for key in self.keys:
            self.previous_rewards[key] = torch.zeros((self.env.num_envs,  1), dtype=torch.float, device=self.device)
            self.previous_dones[key] = torch.zeros((self.env.num_envs, 1), dtype=torch.int, device=self.device)

    def _init_config(self):
        super()._init_config()
        self.algo_history_length_dict = self.env.config.obs.get('history_length', {})
        # Multi-critic reward related Config
        self.rw_groups = self.env.config.rewards.reward_groups
        self.rw_weights = self.env.config.rewards.reward_weights
        self.num_act_lower_body = self.env.config.robot.lower_body_actions_dim
        self.num_act_upper_body = self.env.config.robot.upper_body_actions_dim
        self.num_act = {'lower_body': self.num_act_lower_body, 'upper_body': self.num_act_upper_body}

    def _setup_models_and_optimizer(self):
        self.actors = {}
        self.critics = {}
        self.actor_learning_rates = {}
        self.critic_learning_rates = {}
        for key in self.keys:
            print(f"key: {key}")
            print("num_act[key]: ", self.num_act[key])
            self.actors[key] = PPOActor(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config_dict=getattr(self.config.module_dict, 'actor' + '_' + key),
                num_actions=self.num_act[key],
                init_noise_std=self.config.init_noise_std[key],
                std_threshold=self.std_threshold[key],
            ).to(self.device)
            self.critics[key] = PPOCritic(self.algo_obs_dim_dict,
                                          self.config.module_dict.critic).to(self.device)
            self.actor_learning_rates[key] = self.actor_learning_rate
            self.critic_learning_rates[key] = self.critic_learning_rate
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        for key in self.keys:
            self.actor_optimizers[key] = optim.Adam(self.actors[key].parameters(), lr=self.actor_learning_rates[key])
            self.critic_optimizers[key] = optim.Adam(self.critics[key].parameters(), lr=self.critic_learning_rates[key])

    def _setup_storage(self):
        self.multi_storage = {}
        for key in self.keys:
            self.multi_storage[key] = RolloutStorage(self.env.num_envs, (self.num_steps_per_env // self.actor_deploy_freq[key]), device=self.device)
            ## Register obs keys
            for obs_key, obs_dim in self.algo_obs_dim_dict.items():
                history_len = self.algo_history_length_dict.get(obs_key, 1)
                print(f"Registering key: {obs_key} with shape: {obs_dim * history_len}")
                self.multi_storage[key].register_key(obs_key, shape=(obs_dim * history_len,), dtype=torch.float)
        
            ## Register others
            self.multi_storage[key].register_key('actions', shape=(self.num_act[key],), dtype=torch.float)
            self.multi_storage[key].register_key('rewards', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('dones', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('returns', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('advantages', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('values', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('actions_log_prob', shape=(1,), dtype=torch.float)
            self.multi_storage[key].register_key('action_mean', shape=(self.num_act[key],), dtype=torch.float)
            self.multi_storage[key].register_key('action_sigma', shape=(self.num_act[key],), dtype=torch.float)

    def _eval_mode(self):
        for actor in self.actors.values():
            actor.eval()
        for critic in self.critics.values():
            critic.eval()

    def _train_mode(self):
        for actor in self.actors.values():
            actor.train()
        for critic in self.critics.values():
            critic.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            for key in self.keys:
                self.actors[key].load_state_dict(loaded_dict["actor_model_state_dict"][key])
                self.critics[key].load_state_dict(loaded_dict["critic_model_state_dict"][key])
            if self.load_optimizer:
                for key in self.keys:
                    self.actor_optimizers[key].load_state_dict(loaded_dict["actor_optimizer_state_dict"][key])
                    self.critic_optimizers[key].load_state_dict(loaded_dict["critic_optimizer_state_dict"][key])
                    self.actor_learning_rates[key] = loaded_dict['actor_optimizer_state_dict'][key]['param_groups'][0]['lr']
                    self.critic_learning_rates[key] = loaded_dict['critic_optimizer_state_dict'][key]['param_groups'][0]['lr']
                self.set_learning_rates(self.actor_learning_rates, self.critic_learning_rates)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rates}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rates}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def set_learning_rates(self, actor_learning_rates, critic_learning_rates):
        self.actor_learning_rates = actor_learning_rates
        self.critic_learning_rates = critic_learning_rates

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            "actor_model_state_dict": {key: self.actors[key].state_dict() for key in self.keys},
            'critic_model_state_dict': {key: self.critics[key].state_dict() for key in self.keys},
            'actor_optimizer_state_dict': {key: self.actor_optimizers[key].state_dict() for key in self.keys},
            'critic_optimizer_state_dict': {key: self.critic_optimizers[key].state_dict() for key in self.keys},
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def critic_evaluation(self, critic_obs, key):
        return self.critics[key].evaluate(critic_obs).detach()

    def _multi_actor_critic_rollout_step(self, obs_dict):
        actions_dict = {}
        for key in self.keys:
            freq = self.actor_deploy_freq[key]

            if self.actor_step_counters[key] % freq == 0:
                actions = self.actors[key].act(obs_dict["actor_obs"])
                actions_log_prob = self.actors[key].get_actions_log_prob(actions).detach().unsqueeze(1)
                action_mean = self.actors[key].action_mean.detach()
                action_sigma = self.actors[key].action_std.detach()
                values = self.critic_evaluation(obs_dict["critic_obs"], key)
                actor_obs = obs_dict["actor_obs"]
                critic_obs = obs_dict["critic_obs"]
                # import ipdb; ipdb.set_trace()
                self.multi_storage[key].update_key('actions', actions) 
                self.multi_storage[key].update_key('actions_log_prob', actions_log_prob)
                self.multi_storage[key].update_key('action_mean', action_mean)
                self.multi_storage[key].update_key('action_sigma', action_sigma)
                self.multi_storage[key].update_key('values', values)
                self.last_values = values
                # print("values: ", key, values)
                self.multi_storage[key].update_key('actor_obs', actor_obs)
                self.multi_storage[key].update_key('critic_obs', critic_obs)

                self.previous_actions[key] = actions
                # self.previous_obs_sict[key] = obs_dict
                self.previous_rewards[key] = torch.zeros((self.env.num_envs,  1), dtype=torch.float, device=self.device)
                self.previous_dones[key] = torch.zeros((self.env.num_envs, 1), dtype=torch.int, device=self.device)
            else:
                actions = self.previous_actions[key] 
                # obs_dict = self.previous_obs_sict[key]

            actions_dict[key] = actions

        return actions_dict

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                actions = self._multi_actor_critic_rollout_step(obs_dict)

                actor_state = {}
                actor_state["actions"] = torch.cat([actions[key] for key in self.keys], dim=1)
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for key in self.keys:
                    self.actor_step_counters[key] += 1
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                
                dones = dones.to(self.device)
                dones_stored = dones.clone().unsqueeze(1)
                rewards_stored = {}
                for rw_key, rw_value in rewards.items():
                    rewards_stored[rw_key] = rw_value.to(self.device).clone().unsqueeze(1)

                self.episode_env_tensors.add(infos["to_log"])

                self._process_env_step(rewards_stored, dones_stored, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    for rw_key, rw_value in rewards.items():
                        self.cur_reward_sum += rw_value
                    # self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training
            for key in self.keys:
                returns, advantages = self._compute_returns(
                    last_obs_dict=obs_dict,
                    policy_state_dict=dict(values=self.multi_storage[key].query_key('values'), 
                                           dones=self.multi_storage[key].query_key('dones'), 
                                           rewards=self.multi_storage[key].query_key('rewards')),
                    key=key
                )
                self.multi_storage[key].batch_update_data('returns', returns)
                self.multi_storage[key].batch_update_data('advantages', advantages)
                
        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        for key in self.keys:
            self.previous_rewards[key] += rewards[key]
            self.previous_dones[key] |= dones
            # lyt: asuming that only one time_out in deploy horizon
            if 'time_outs' in infos:
                # import ipdb; ipdb.set_trace()
                self.previous_rewards[key] += self.gamma * self.last_values * infos['time_outs'].unsqueeze(1).to(self.device)

            freq = self.actor_deploy_freq[key]
            if self.actor_step_counters[key] % freq == 0:
                self.multi_storage[key].update_key('rewards', self.previous_rewards[key])
                self.multi_storage[key].update_key('dones', self.previous_dones[key])
                # print("rewards: ", key, self.previous_rewards[key])
                # print("dones: ", key, self.previous_dones[key])
                self.multi_storage[key].increment_step()
                for actor in self.actors.values():
                    actor.reset(dones)
                for critic in self.critics.values():
                    critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict, key):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the 
        environment based on the provided observations and policy state. It uses 
        Generalized Advantage Estimation (GAE) to compute the advantages, which 
        helps in reducing the variance of the policy gradient estimates.
        Args:
            last_obs_dict (dict): The last observation dictionary containing the 
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state 
                          information, including 'values', 'dones', 
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        last_values = self.critic_evaluation(last_obs_dict["critic_obs"], key)
        advantage = 0
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        
        num_steps = returns.shape[0]
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
    
    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()
        for key in self.keys:
            generator = self.multi_storage[key].mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
            for policy_state_dict in generator:
                # Move everything to the device
                policy_state_dict_with_key = {}
                for policy_state_key in policy_state_dict.keys():
                    policy_state_dict_with_key[policy_state_key + '_' + key] = policy_state_dict[policy_state_key].to(self.device)

                # import ipdb; ipdb.set_trace()
                loss_dict = self._update_algo_step(policy_state_dict_with_key, loss_dict, key)
            
            self.multi_storage[key].clear()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        return loss_dict
        
    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        for key in self.keys:
            loss_dict['Value' + '_' + key] = 0
            loss_dict['Surrogate' + '_' + key] = 0
            loss_dict['Entropy' + '_' + key] = 0
            loss_dict["KL" + '_' + key] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict, key):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict, key)
        return loss_dict

    def _compute_ppo_loss(self, policy_state_dict, key):
        # print("policy_state_dict: ", policy_state_dict.keys())
        actions_batch = policy_state_dict['actions' + '_' + key]
        # target_values_batch = policy_state_dict['returns'] # This is wrong
        target_values_batch = policy_state_dict['values' + '_' + key] # This is correct
        advantages_batch = policy_state_dict['advantages' + '_' + key]
        returns_batch = policy_state_dict['returns' + '_' + key]
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob' + '_' + key]
        old_mu_batch = policy_state_dict['action_mean' + '_' + key]
        old_sigma_batch = policy_state_dict['action_sigma' + '_' + key]

        self.actors[key].act(policy_state_dict['actor_obs' + '_' + key])
        actions_log_prob_batch = self.actors[key].get_actions_log_prob(actions_batch)
        # value_batch = self.critic.evaluate(policy_state_dict["critic_obs"])
        value_batch = self.critics[key].evaluate(policy_state_dict['critic_obs' + '_' + key])
        mu_batch = self.actors[key].action_mean
        sigma_batch = self.actors[key].action_std
        entropy_batch = self.actors[key].entropy

        # KL
        # import ipdb; ipdb.set_trace()
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch) + 1.e-5) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rates[key] = max(1e-5, self.actor_learning_rates[key] / 1.5)
                    self.critic_learning_rates[key] = max(1e-5, self.critic_learning_rates[key] / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rates[key] = min(1e-2, self.actor_learning_rates[key] * 1.5)
                    self.critic_learning_rates[key] = min(1e-2, self.critic_learning_rates[key] * 1.5)

                for param_group in self.actor_optimizers[key].param_groups:
                    param_group['lr'] = self.actor_learning_rates[key]
                for param_group in self.critic_optimizers[key].param_groups:
                    param_group['lr'] = self.critic_learning_rates[key]

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss
        
        return actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean

    def _update_ppo(self, policy_state_dict, loss_dict, key):
        actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean = self._compute_ppo_loss(policy_state_dict, key)

        self.actor_optimizers[key].zero_grad()
        self.critic_optimizers[key].zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actors[key].parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critics[key].parameters(), self.max_grad_norm)

        self.actor_optimizers[key].step()
        self.critic_optimizers[key].step()

        loss_dict['Value' + '_' + key] += value_loss.item()
        loss_dict['Surrogate' + '_' + key] += surrogate_loss.item()
        loss_dict['Entropy' + '_' + key] += entropy_loss.item()
        loss_dict["KL" + '_' + key] += kl_mean.item()
        return loss_dict

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        mean_std = {}
        for key, actor in self.actors.items():
            mean_std[key] = actor.std.mean().item()
        train_log_dict['mean_std'] = mean_std

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
            for key in mean_std.keys():
                log_string += f"""{f'Mean action noise std ({key}):':>{pad}} {mean_std[key]:.2f}\n"""
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                          )
            for key in mean_std.keys():
                log_string += f"""{f'Mean action noise std ({key}):':>{pad}} {mean_std[key]:.2f}\n"""

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass
    
    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        # Logging Loss Dict
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(f'Loss/{loss_key}', loss_value, log_dict['it'])
        # self.writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
        # self.writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
        # self.writer.add_scalar('Policy/mean_noise_std', train_log_dict['mean_std'], log_dict['it'])
        for key in self.keys:
            self.writer.add_scalar(f'Loss/actor_learning_rate_{key}', self.actor_learning_rates[key], log_dict['it'])
            self.writer.add_scalar(f'Loss/critic_learning_rate_{key}', self.critic_learning_rates[key], log_dict['it'])
            self.writer.add_scalar(f'Policy/mean_noise_std_{key}', train_log_dict['mean_std'][key], log_dict['it'])
        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])
        if hasattr(self.env, 'action_scale_upper_body'):
            self.writer.add_scalar('Env/action_scale_upper_body', torch.mean(self.env.action_scale_upper_body).item(), log_dict['it'])
        if hasattr(self.env, 'apply_force_scale'):
            self.writer.add_scalar('Env/apply_force_scale', torch.mean(self.env.apply_force_scale).item(), log_dict['it'])
        if hasattr(self.env, 'command_height_scale'):
            self.writer.add_scalar('Env/command_height_scale', torch.mean(self.env.command_height_scale).item(), log_dict['it'])
        if hasattr(self.env, 'upper_body_tracking_sigma'):
            self.writer.add_scalar('Env/upper_body_tracking_sigma', torch.mean(self.env.upper_body_tracking_sigma).item(), log_dict['it'])
    
    @property
    def inference_model(self):
        return {
            "actors": self.actors,
            "critics": self.critics,
            "keys": self.keys,
        }

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################
    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act_lower_body + self.num_act_upper_body, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        for key in self.keys:
            self.actor_step_counters[key] = 0
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            for key in self.keys:
                self.actor_step_counters[key] += 1
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()
    
    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            # print(actions)
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    def act_inference(self, actor_obs):
        actions = {}
        for key in self.keys:
            freq = self.actor_deploy_freq[key]
            if self.actor_step_counters[key] % freq == 0:
                actions[key] = self.actors[key].act_inference(actor_obs)
                self.previous_actions[key] = actions[key]
            else:
                actions[key] = self.previous_actions[key]

        return torch.cat([actions[key] for key in self.keys], dim=1)

    def _get_inference_policy(self, device=None):
        for key in self.keys:
            self.actors[key].eval() # switch to evaluation mode (dropout for example)
            if device is not None:
                self.actors[key].to(device)
        return self.act_inference