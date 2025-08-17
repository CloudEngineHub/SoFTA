import torch
from torch import nn
import os
import copy

def export_policy_as_jit(actor_critic, path, exported_policy_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')

        class PPOWrapper(nn.Module):
            def __init__(self, actor):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.actor = actor

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                return self.actor.act_inference(actor_obs)

        wrapper = PPOWrapper(actor)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_multi_agent_wbc_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        keys = inference_model['keys']
        actors = {key: copy.deepcopy(inference_model['actors'][key]).to('cpu') for key in keys}

        class PPOMAWBCWrapper(nn.Module):
            def __init__(self, actors, keys):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOMAWBCWrapper, self).__init__()
                self.actors = nn.ModuleDict(actors)
                self.keys = keys

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """

                for key in self.keys:
                     print(self.actors[key].act_inference(actor_obs).shape)
                return torch.concat([self.actors[key].act_inference(actor_obs) for key in self.keys], dim=-1)

        wrapper = PPOMAWBCWrapper(actors, keys)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )