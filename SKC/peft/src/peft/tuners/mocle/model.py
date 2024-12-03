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

import warnings
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from typing import Any

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)

from peft.utils.integrations import gather_params_ctx
from .layer import MoELoraLayer, MoELoraLinear


class MoELoraModel(LoraModel):
   
    prefix: str = "moelora_"
    
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
        
    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        super()._check_new_adapter_config(config)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            # [modified]
            "multiple_loras": lora_config.multiple_loras,
            "cluster": lora_config.cluster,
            "noise_std": lora_config.noise_std,
            "kmeans_ckpt": lora_config.kmeans_ckpt,
            "total_tasks": lora_config.total_tasks,
            "gates_tmp": lora_config.gates_tmp,
            "g_enable": lora_config.g_enable,
            "topk": lora_config.topk,
            "num_experts": lora_config.num_experts,    
            "moe_type": lora_config.moe_type,
            "embedding_dim": lora_config.embedding_dim
        }

        # If it is not an MoELoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, MoELoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
                lora_config.use_rslora,
                lora_config.use_dora,
            )
    
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        new_module = MoELoraLinear(target, adapter_name, **kwargs)

        return new_module

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

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (MoELoraLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MoELoraLayer):
                module.set_adapter(adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_MOELORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _register_pre_hooks(self, task_types):
        """Helper method to register pre hooks."""
        if task_types is None:
            return []

        def pre_hook(_, args, kwargs):
            kwargs["task_types"] = task_types
            return args, kwargs

        handles = []

        for module in self.model.modules():
            if isinstance(module, MoELoraLinear):
                handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
                handles.append(handle)

        return handles
    
    @contextmanager
    def _manage_pre_hooks(self, task_types):
        """Context manager to handle the lifecycle of pre hooks."""
        handles = self._register_pre_hooks(task_types)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
    
    def forward(self, task_types=None, *args, **kwargs):
        with self._manage_pre_hooks(task_types):
            return self.model(*args, **kwargs)

    def generate(self, task_types=None, *args, **kwargs):
        with self._manage_pre_hooks(task_types):
            return self.model.generate(*args, **kwargs)        
