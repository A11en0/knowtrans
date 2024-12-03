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

from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class MoELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MoELora`].

    Args:

    """
    
    multiple_loras: bool = field(default=False, metadata={"help": "Whether to use multiple LORAs."})
    cluster: bool = field(default=False, metadata={"help": "Whether to use LORAs in a cluster."})
    noise_std: float = field(default=0.1, metadata={"help": "The std of gaussian noise for the KL loss."})
    kmeans_ckpt: Optional[str] = field(default = None, metadata={"help": "The checkpoint path for the kmeans model ."})
    total_tasks: int = field(default=64, metadata={"help": "The total number of tasks."})
    gates_tmp: float = field(default=1.0, metadata={"help": "The temperature for the gates."})
    g_enable: bool = field(default=False, metadata={"help": "Whether to use the gate mechanism."})
    topk: int = field(default=1, metadata={"help": "The number of topk experts to select."})
    num_experts: int = field(default=4, metadata={"help": "The number of experts."})
    embedding_dim: int = field(default=64, metadata={"help": "The embedding dimmension."})
    
    def __post_init__(self):
        super().__post_init__()

        self.peft_type = PeftType.MOELORA

        if self.use_dora:
            raise ValueError(f"{self.peft_type} does not support DoRA.")

        if self.loftq_config:
            raise ValueError(f"{self.peft_type} does not support LOFTQ.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
