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
from typing import List, Optional, Union


@dataclass
class LoraMoEConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MoELora`].

    Args:

    """

    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    blc_alpha: int = field(default=None, metadata={"help": "Alpha of blcloss"})
    blc_weight: int = field(default=None, metadata={"help": "Weight of blcloss"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    # merge_weights: bool = field(
    #     default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    # )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})

    def __post_init__(self):
        super().__post_init__()
        
        self.peft_type = PeftType.LORAMOE
        
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

