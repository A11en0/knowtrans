# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
# from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
# from .config import MoELoraConfig

# from .layer import dispatch_default
# from .mmoelora import MoELoraLinear, MMOELoraLayer
from peft.tuners.lora import LoraLayer

from .layer import MoELoraLinear
from peft.tuners.lora import LoraModel


class MoELoraModel(LoraModel):
    # prefix: str = "lora_"

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)


    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f".*\.{key}$", current_key), pattern_keys), target_name)

        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        kwargs["bias"] = bias

        if isinstance(target, LoraLayer):
            target.update_layer(
                adapter_name,
                r,
                alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
                use_rslora=False
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)
    
        multiple_loras = getattr(lora_config, "multiple_loras", False)
        cluster = getattr(lora_config, "cluster", False)
        noise_std = getattr(lora_config, "noise_std", 0.1)
        kmeans_ckpt = getattr(lora_config, "kmeans_ckpt", None)
        total_tasks = getattr(lora_config, "total_tasks", 64)
        gates_tmp = getattr(lora_config, "gates_tmp", 1.0)
        g_enable = getattr(lora_config, "g_enable", False)
        topk = getattr(lora_config, "topk", 1)
        num_experts = getattr(lora_config, "num_experts", 4)
        embedding_dim = getattr(lora_config, "embedding_dim", 64)
        
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = MoELoraLinear(
            target, 
            adapter_name, 
            in_features=in_features, 
            out_features=out_features, 
            bias=bias, 
            embedding_dim=embedding_dim,
            multiple_loras=multiple_loras, 
            cluster=cluster, 
            noise_std=noise_std,
            kmeans_ckpt=kmeans_ckpt,
            total_tasks=total_tasks,
            gates_tmp=gates_tmp,
            g_enable=g_enable,
            topk=topk,
            num_experts=num_experts,
            **kwargs
        )

        return new_module
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        # for n, p in model.named_parameters():
        #     if self.prefix not in n:
        #         p.requires_grad = False

        for n, p in self.model.named_parameters():
            if ("lora_" not in n) and ("w_gate" not in n) and ("task_emb" not in n):
                p.requires_grad = False

    # def _create_and_replace(
    #     self,
    #     lora_config,
    #     adapter_name,
    #     target,
    #     target_name,
    #     parent,
    #     current_key,
    # ):
    #     if current_key is None:
    #         raise ValueError("Current Key shouldn't be `None`")

    #     # is_target_modules_in_base_model = False
    #     kwargs = {
    #         "r": lora_config.r,
    #         "lora_alpha": lora_config.lora_alpha,
    #         "lora_dropout": lora_config.lora_dropout,
    #         "fan_in_fan_out": lora_config.fan_in_fan_out,
    #         "init_lora_weights": lora_config.init_lora_weights,
    #         "task_num": lora_config.task_num,
    #         "task_embedding_dim": lora_config.task_embedding_dim,
    #         "expert_num": lora_config.expert_num,
    #     }

    #     if isinstance(target, LoraLayer):
    #         target.update_layer(
    #             adapter_name,
    #             lora_config.init_r,
    #             lora_config.lora_alpha,
    #             lora_config.lora_dropout,
    #             lora_config.init_lora_weights,
    #         )
    #     else:
    #         if isinstance(target, torch.nn.Linear):
    #             in_features, out_features = target.in_features, target.out_features
    #             if kwargs["fan_in_fan_out"]:
    #                 warnings.warn(
    #                     "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
    #                     "Setting fan_in_fan_out to False."
    #                 )
    #                 kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
    #         else:
    #             raise ValueError(
    #                 f"Target module {target} is not supported. "
    #                 f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
    #             )
            
    #         new_module = self._create_new_module(
    #             lora_config, 
    #             adapter_name, 
    #             target, 
    #             **kwargs
    #         )
    #         if adapter_name not in self.active_adapters:
    #             # adding an additional adapter: it is not automatically trainable
    #             new_module.requires_grad_(False)
    #         self._replace_module(parent, target_name, new_module, target)

    # @staticmethod
    # def _create_new_module(lora_config, adapter_name, target, **kwargs):

    #     if isinstance(target, BaseTunerLayer):
    #         target_base_layer = target.get_base_layer()
    #     else:
    #         target_base_layer = target

    #     if isinstance(target_base_layer, torch.nn.Linear):
    #         if kwargs["fan_in_fan_out"]:
    #             warnings.warn(
    #                 "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
    #                 "Setting fan_in_fan_out to False."
    #             )
    #             kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
    #     else:
    #         raise ValueError(
    #             f"Target module {target} is not supported. "
    #             f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
    #         )
    #     return MoELoraLinear(target, adapter_name, **kwargs) 

    # def _set_adapter_layers(self, enabled=True):
    #     for module in self.model.modules():
    #         if isinstance(module, LoraLayer):
    #             module.disable_adapters = False if enabled else True
    #         elif isinstance(module, ModulesToSaveWrapper):
    #             module.disable_adapters = False if enabled else True

    # def enable_adapter_layers(self):
    #     self._set_adapter_layers(enabled=True)    

    # def _get_active_adapters(self) -> List[str]:
    #     active_adapters = None
    #     for module in self.model.modules():
    #         if isinstance(module, LoraLayer):
    #             active_adapters = module.active_adapters

    #     if active_adapters is None:
    #         raise ValueError(
    #             "Something went wrong, no active adapter could be found, please report the issue on GitHub"
    #         )
    #     return active_adapters

    # def disable_adapter_layers(self):
    #     for active_adapter in self._get_active_adapters():
    #         val = self.peft_config[active_adapter].bias
    #         if val != "none":
    #             msg = (
    #                 f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
    #                 "output as the the base model would without adaption."
    #             )
    #             warnings.warn(msg)
    #     self._set_adapter_layers(enabled=False)

    # def set_adapter(self, adapter_name):
    #     for module in self.model.modules():
    #         if isinstance(module, LoraLayer):
    #             if module.merged:
    #                 warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
    #                 module.unmerge()
    #             module.active_adapter = adapter_name

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config 

    # def _mark_only_adapters_as_trainable(self, model: nn.Module):
    #     for active_adapter in self._get_active_adapters():
    #         bias = self.peft_config[active_adapter].bias

    #         for n, p in self.model.named_parameters():
    #             if ("lora_" not in n) and ("w_gate" not in n) and ("task_emb" not in n):
    #                 p.requires_grad = False
    #         if bias == "none":
    #             return
    #         elif bias == "all":
    #             for n, p in self.model.named_parameters():
    #                 if "bias" in n:
    #                     p.requires_grad = True
    #         elif bias == "lora_only":
    #             for m in self.model.modules():
    #                 if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
    #                     m.bias.requires_grad = True
    #         else:
    #             raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def _register_pre_hooks(self, task_ids):
        """Helper method to register pre hooks."""
        if task_ids is None:
            return []

        def pre_hook(_, args, kwargs):
            kwargs["task_ids"] = task_ids
            return args, kwargs

        handles = []

        for module in self.model.modules():
            if isinstance(module, MoELoraLinear):
                handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
                handles.append(handle)

        return handles
    
    @contextmanager
    def _manage_pre_hooks(self, task_ids):
        """Context manager to handle the lifecycle of pre hooks."""
        handles = self._register_pre_hooks(task_ids)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model(*args, **kwargs)
        
    def generate(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model.generate(*args, **kwargs)
        


