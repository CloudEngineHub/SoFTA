from time import time
from warnings import WarningMessage
import numpy as np
import pinocchio as pin
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base_ma import LeggedRobotBase
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_inverse,
)
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from humanoidverse.envs.locomotion.locomotion_ma import LeggedRobotLocomotion

from loguru import logger

from scipy.stats import vonmises

from humanoidverse.utils.arm_ik import arm_ik

DEBUG = False
class LeggedRobotLocomotionStance(LeggedRobotLocomotion):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.stand_prob = self.config.stand_prob


    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Push the robots randomly
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
            
        # Update locomotion commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )

        self.commands[:, 0] *= self.commands[:, 4]
        self.commands[:, 1] *= self.commands[:, 4]
        self.commands[:, 2] *= self.commands[:, 4]
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled_lower_body = self.config.robot.control.action_scale * actions[:, self.lower_body_action_indices]
        actions_scaled_upper_body = actions[:, self.upper_body_action_indices]
        actions_scaled = torch.cat([actions_scaled_lower_body, actions_scaled_upper_body], dim=1)
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques

    def _resample_commands(self, env_ids):

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Sample the tapping or stand command with a probability
        self.commands[env_ids, 4] = (torch.rand(len(env_ids), device=self.device) > self.stand_prob).float()
        # Sample the tapping in place command with a probability
        self.commands[env_ids, 0] *= self.commands[env_ids, 4]
        self.commands[env_ids, 1] *= self.commands[env_ids, 4]
        self.commands[env_ids, 2] *= self.commands[env_ids, 4]
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands


    def _draw_debug_vis(self, debug=DEBUG):
        return
    
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Resets the environments with the given ids and optionally to the target states
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True

        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()
    
    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        ## Lower body dof reset
        if target_state is not None:
            self.simulator.dof_pos[env_ids.unsqueeze(1), self.lower_body_action_indices] = target_state[..., 0]
            self.simulator.dof_vel[env_ids.unsqueeze(1), self.lower_body_action_indices] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids.unsqueeze(1), self.lower_body_action_indices] = \
                self.default_dof_pos[:, self.lower_body_action_indices] * \
                torch_rand_float(0.5, 1.5, (len(env_ids), self.config.robot.lower_body_actions_dim), device=str(self.device))
            self.simulator.dof_vel[env_ids.unsqueeze(1), self.lower_body_action_indices] = 0.

        ## Upper body dof reset
        self.simulator.dof_pos[env_ids.unsqueeze(1), self.upper_dof_indices] = \
            self.default_dof_pos[:, self.upper_body_action_indices]
        self.simulator.dof_vel[env_ids.unsqueeze(1), self.upper_dof_indices] *= 0.0
    
    ########################### GAIT REWARDS ###########################
    
    ########################### FEET REWARDS ###########################
    
    ######################## LIMITS REWARDS #########################
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit (lower body only)
        out_of_limits = -(self.simulator.dof_pos[:, self.lower_body_action_indices] - \
                          self.simulator.dof_pos_limits[self.lower_body_action_indices, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos[:, self.lower_body_action_indices] - \
                          self.simulator.dof_pos_limits[self.lower_body_action_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit (lower body only)
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel[:, self.lower_body_action_indices]) - \
                          self.simulator.dof_vel_limits[self.lower_body_action_indices, 1]).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit (lower body only)
        return torch.sum((torch.abs(self.torques[:, self.lower_body_action_indices]) - \
                          self.torque_limits[self.lower_body_action_indices, 1]).clip(min=0.), dim=1)
    
    ######################### PENALTY REWARDS #########################
    def _reward_penalty_negative_knee_joint(self):
        # Penalize negative knee joint angles (lower body only)
        return torch.sum((self.simulator.dof_pos[:, self.knee_dof_indices] < 0.).float(), dim=1)
    
    def _reward_penalty_torques(self):
        # Penalize torques (lower body only)
        return torch.sum(torch.square(self.torques[:, self.lower_body_action_indices]), dim=1)
    
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities (lower body only)
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_body_action_indices]), dim=1)
    
    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations (lower body only)
        return torch.sum(torch.square((self.last_dof_vel[:, self.lower_body_action_indices] - \
                                       self.simulator.dof_vel[:, self.lower_body_action_indices]) / self.dt), dim=1)
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions (lower body only)
        return torch.sum(torch.square(self.last_actions[:, self.lower_body_action_indices] - \
                                      self.actions[:, self.lower_body_action_indices]), dim=1)
    
    def _reward_penalty_feet_swing_height(self):
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        # self.min_feet_height = torch.min(feet_height, self.min_feet_height)
        # print("min_feet_height: ", self.min_feet_height)
        # set to zero if not tappinging (standing)
        target_height = self.config.rewards.feet_height_target * self.commands[:, 4:5] + \
                        self.config.rewards.feet_height_stand * (1.0 - self.commands[:, 4:5])
        height_error = torch.square(feet_height - target_height) * ~contact
        return torch.sum(height_error, dim=(1))
    
    def _reward_penalty_torso_orientation(self):
        # Penalize non flat torso orientation
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        projected_gravity_torso = quat_rotate_inverse(torso_quat, self.gravity_vec)
        return torch.abs(projected_gravity_torso[:, 1]) * (1.0 - self.commands[:, 4]) * (1.0 - self.zero_fix_waist_roll) + \
               torch.square(projected_gravity_torso[:, 0]) * (1.0 - self.commands[:, 4]) * (1.0 - self.zero_fix_waist_pitch) + \
               torch.sum(torch.square(projected_gravity_torso[:, :2]), dim=1) * self.commands[:, 4] * self.apply_waist_roll_pitch_only_when_stance        
    
    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) * self.commands[:, 4] # target - 0.02 ~ target + 0.02 is acceptable, apply only when tapping

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2): # left and right feet
            is_stance = (self.leg_phase[:, i] < 0.55) | (self.commands[:, 4] == 0)
            contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
            contact_reward = ~(contact ^ is_stance)
            contact_penalty = contact ^ is_stance
            # res += contact_reward
            res += contact_reward.int() - contact_penalty.int()
        return res
    
    def _reward_penalty_contact(self):
        # Initialize the penalty reward tensor
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Check if the agent is in stance mode (commands[:, 4] == 0)
        is_stance = (self.commands[:, 4] == 0)
        # Determine foot contact (contact force in Z-axis > 1 is considered ground contact)
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1  
        # Count the number of feet in contact with the ground
        num_feet_on_ground = contact.sum(dim=1)
        # Penalize if any foot is off the ground when commands[:, 4] == 0 (stance mode)
        res[is_stance & (num_feet_on_ground < 2)] = 1.0  
        # Penalize if both feet are on the ground when commands[:, 4] == 1 (walking mode)
        res[~is_stance & ((num_feet_on_ground == 2) | (num_feet_on_ground == 0))] = 1.0  
        return res
    
    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1) * self.commands[:, 4] # only apply when walking

    
    def _reward_alive(self):
        # Reward for staying alive
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
    
    ######################### NOT USED REWARDS #########################
    def _reward_waist_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default (NOT used)
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.waist_dof_indices] - self.default_dof_pos[:,self.waist_dof_indices])
        return torch.sum(deviation, dim=1)
    
    def _reward_penalty_stance_dof(self):
        # Penalize the lower body dof velocity
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_body_action_indices]), dim=1) * (1.0 - self.commands[:, 4])

    def _reward_penalty_stance_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:, self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_tap_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:, self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        stance_tap = self.commands[:, 4] * (torch.abs(self.commands[:, 0]) > 0.0)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - stance_tap)
    
    def _reward_penalty_stance_root(self):
        # Penalize the root position
        feet_mid_pos = (self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] + self.simulator._rigid_body_pos[:, self.feet_indices[1], :3]) / 2
        root_pos = self.simulator._rigid_body_pos[:, self.pelvis_index, :3]
        root_feet_diff = root_pos - feet_mid_pos
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_root_feet_diff = quat_rotate_inverse(pelvis_quat, root_feet_diff)
        return torch.abs(projected_root_feet_diff[:, 1]) * (1.0 - self.commands[:, 4])


    def _reward_penalty_stand_still(self):
        # Penalize standing still
        no_contacts = torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] < 0.1, dim=1) > 0
        return no_contacts.float() * (1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_symmetry(self):
        # TODO: Hardcoded
        diff_lower_body_dof_po_no = self.simulator.dof_pos[:, self.lower_left_dofs_idx_no] - self.simulator.dof_pos[:, self.lower_right_dofs_idx_no]
        diff_lower_body_dof_pos_op = self.simulator.dof_pos[:, self.lower_left_dofs_idx_op] + self.simulator.dof_pos[:, self.lower_right_dofs_idx_op]
        return torch.sum(torch.abs(diff_lower_body_dof_po_no) + 
                         torch.abs(diff_lower_body_dof_pos_op), dim=1) * (1.0 - self.commands[:, 4])
    
    ######################### Phase Time #########################

    def _calc_phase_time(self):
        # Calculate the phase time
        episode_length_np = self.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
        phase_time *= self.commands[:, 4].cpu().numpy() # only apply when locomotion
        return phase_time

    ######################### Observations #########################

    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_ref_upper_dof_pos(self):
        return self.ref_upper_dof_pos
    
    def _get_obs_actions(self,):
        return self.actions[:, self.lower_body_action_indices]
    
    def _get_obs_command_stand(self):
        return self.commands[:, 4:5]
    
    def _get_obs_base_orientation(self):
        return self.base_quat[:, 0:4]