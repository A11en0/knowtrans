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
from typing import Any, List, Optional

import math
import warnings
from typing import Any, Optional, Union

import packaging
import torch
import transformers
from torch import nn

import torch.nn.functional as F

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config


class MoELoraLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")
    
    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora: bool = False, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)


    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class MoELoraLinear(nn.Module, MoELoraLayer):
    # MoELora implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        # [modified] 
        multiple_loras: bool = False,
        cluster: bool = False,
        noise_std: float = 0.1,
        kmeans_ckpt: str = None,
        total_tasks: int = 64,
        gates_tmp: float = 1.0,
        g_enable = False,
        topk = 1,
        num_experts = 4,
        moe_type = 0,
        embedding_dim = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        MoELoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        # [modified]
        self.multiple_loras = multiple_loras 
        self.cluster = cluster
        self.g_enable = g_enable
        self.moe_type = moe_type
        self.embedding_dim = embedding_dim
        
        if self.multiple_loras:
            if self.cluster:
                self.total_tasks = total_tasks 
                self.task_emb = nn.Embedding(
                    num_embeddings=self.total_tasks,embedding_dim=embedding_dim,
                )
                # self.kmeans = joblib.load(kmeans_ckpt)    
                self.num_experts = num_experts
                self.w_gate = nn.Linear(embedding_dim, self.num_experts)
                self.gates_tmp = gates_tmp
                self.topk = topk
                self.noise_std = noise_std
    
    # def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
    #     """
    #     Merge the active adapter weights into the base weights

    #     Args:
    #         safe_merge (`bool`, *optional*):
    #             If True, the merge operation will be performed in a copy of the original weights and check for NaNs
    #             before merging the weights. This is useful if you want to check if the merge operation will produce
    #             NaNs. Defaults to `False`.
    #         adapter_names (`List[str]`, *optional*):
    #             The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
    #             to `None`.
    #     """
    #     adapter_names = check_adapters_to_merge(self, adapter_names)
    #     if not adapter_names:
    #         # no adapter to merge
    #         return

    #     for active_adapter in adapter_names:
    #         base_layer = self.get_base_layer()
    #         if active_adapter in self.lora_A.keys():
    #             if safe_merge:
    #                 # Note that safe_merge will be slower than the normal merge
    #                 # because of the copy operation.
    #                 orig_weights = base_layer.weight.data.clone()
    #                 orig_weights += self.get_delta_weight(active_adapter)

    #                 if not torch.isfinite(orig_weights).all():
    #                     raise ValueError(
    #                         f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    #                     )

    #                 base_layer.weight.data = orig_weights
    #             else:
    #                 base_layer.weight.data += self.get_delta_weight(active_adapter)
    #             self.merged_adapters.append(active_adapter)

    # def unmerge(self) -> None:
    #     """
    #     This method unmerges all merged adapter layers from the base weights.
    #     """
    #     if not self.merged:
    #         warnings.warn("Already unmerged. Nothing to do.")
    #         return
    #     while len(self.merged_adapters) > 0:
    #         active_adapter = self.merged_adapters.pop()
    #         if active_adapter in self.lora_A.keys():
    #             self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     return (
    #         transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out)
    #         * self.scaling[adapter]
    #         / (self.ranknum[adapter] + 1e-5)
    #     )
    
    def forward(self, x: torch.Tensor, task_types: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x)
            torch_result_dtype = result.dtype

            if self.multiple_loras and self.cluster:
                    
                    # device = x.device
                    # [modified]
                    # task_id = torch.tensor(self.kmeans.predict(self.input_emb), device=device)
                    # task_types = kwargs['task_types']
                    # task_id = torch.tensor(task_types).to(device)
                    emb = self.task_emb(task_types)
                    
                    clean_logits = self.w_gate(emb)
                    raw_noise_stddev = self.noise_std
                    noise_stddev = raw_noise_stddev * self.training
                    noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                    logits = noisy_logits
                    logits = F.softmax(logits / self.gates_tmp, dim=1, dtype=torch.float16)             
                    top_logits, top_indices = logits.topk(min(self.topk+1, self.num_experts), dim=1)
                    top_k_logits = top_logits[:, :self.topk]
                    top_k_indices = top_indices[:, :self.topk]
                    top_k_gates = top_k_logits

                    zeros = torch.zeros_like(logits, requires_grad=True)
                    gates = zeros.scatter(1, top_k_indices, top_k_gates)
                    
                    gate_load = gates.gt(0).sum(0)
                    
                    dispatcher = SparseDispatcher(self.num_experts, gates)
                    expert_inputs = list(dispatcher.dispatch(x))
                    gates_ = dispatcher.expert_to_gates()

                    if self.g_enable:
                        expert_outputs = []
                        for i, adpt in enumerate(list(self.r.keys())[:-1]):
                            _x = x.to(self.lora_A[adpt].weight.dtype)
                            expert_inputs[i] = expert_inputs[i].to(self.lora_A[adpt].weight.dtype)
                            expert_outputs.append(
                                self.lora_B[adpt](
                                    self.lora_A[adpt](self.lora_dropout[adpt](expert_inputs[i]))
                                )
                                * self.scaling[adpt]
                            )
                        y_e = dispatcher.combine(expert_outputs)

                        lora_A = self.lora_A['g']
                        lora_B = self.lora_B['g']
                        dropout = self.lora_dropout['g']
                        scaling = self.scaling['g']
                        x = x.to(lora_A.weight.dtype)
                        y_g = lora_B(lora_A(dropout(x))) * scaling

                        w = gates.max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                        y = y_e * w + y_g * (1-w)
                    else:
                        expert_outputs = []
                        for i, adpt in enumerate(list(self.r.keys())):
                            _x = x.to(self.lora_A[adpt].weight.dtype)
                            expert_inputs[i] = expert_inputs[i].to(self.lora_A[adpt].weight.dtype)
                            expert_outputs.append(
                                self.lora_B[adpt](
                                    self.lora_A[adpt](self.lora_dropout[adpt](expert_inputs[i]))
                                )
                                * self.scaling[adpt]
                            )

                        y = dispatcher.combine(expert_outputs)

                    result += y                          
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_A.weight.dtype)

                    if not self.use_dora[active_adapter]:
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                    else:
                        x = dropout(x)
                        result = result + self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                        )

        result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "MoELora." + rep


# class RankAllocator:
#     """
#     The RankAllocator for MoELoRAModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

#     Args:
#         config ([`MoELoRAConfig`]): The configuration of the MoELoRA model.
#         model: the model that we apply MoELoRA to.

#     """

#     def __init__(self, model, peft_config, adapter_name):
#         self.peft_config = peft_config
#         self.adapter_name = adapter_name
#         self.beta1 = peft_config.beta1
#         self.beta2 = peft_config.beta2
#         assert self.beta1 > 0 and self.beta1 < 1
#         assert self.beta2 > 0 and self.beta2 < 1

#         self.reset_ipt()
#         self._set_budget_scheduler(model)

#     def set_total_step(self, total_step):
#         self.peft_config.total_step = total_step

#     def reset_ipt(self):
#         self.ipt = {}
#         self.exp_avg_ipt = {}
#         self.exp_avg_unc = {}

#     def _set_budget_scheduler(self, model):
#         self.init_bgt = 0
#         self.name_set = set()
#         for n, p in model.named_parameters():
#             if f"lora_A.{self.adapter_name}" in n:
#                 self.init_bgt += p.size(0)
#                 self.name_set.add(n.replace("lora_A", "%s"))
#         self.name_set = sorted(self.name_set)
#         # The total final rank budget
#         self.target_bgt = self.peft_config.target_r * len(self.name_set)

#     def budget_schedule(self, step: int):
#         tinit = self.peft_config.tinit
#         tfinal = self.peft_config.tfinal
#         total_step = self.peft_config.total_step
#         # Initial warmup
#         if step <= tinit:
#             budget = self.init_bgt
#             mask_ind = False
#         # Final fine-tuning
#         elif step > total_step - tfinal:
#             budget = self.target_bgt
#             mask_ind = True
#         else:
#             # Budget decreasing with a cubic scheduler
#             mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
#             budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
#             mask_ind = True if step % self.peft_config.deltaT == 0 else False
#         return budget, mask_ind

#     def update_ipt(self, model):
#         # Update the sensitivity and uncertainty for every weight
#         for n, p in model.named_parameters():
#             if "lora_" in n and self.adapter_name in n:
#                 if n not in self.ipt:
#                     self.ipt[n] = torch.zeros_like(p)
#                     self.exp_avg_ipt[n] = torch.zeros_like(p)
#                     self.exp_avg_unc[n] = torch.zeros_like(p)
#                 with torch.no_grad():
#                     if deepspeed_config() is not None:
#                         import deepspeed

#                         grad = deepspeed.utils.safe_get_full_grad(p)
#                         self.ipt[n] = (p * grad).abs().detach()
#                     else:
#                         self.ipt[n] = (p * p.grad).abs().detach()
#                     # Sensitivity smoothing
#                     self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
#                     # Uncertainty quantification
#                     self.exp_avg_unc[n] = (
#                         self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
#                     )

#     def _element_score(self, n):
#         return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

#     def _combine_ipt(self, ipt_E, ipt_AB):
#         ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
#         sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
#         return sum_ipt

#     def mask_to_budget(self, model, budget):
#         value_ipt = {}
#         vector_ipt = {}
#         triplet_ipt = {}
#         # Get the importance score for A, E, B
#         for n, p in model.named_parameters():
#             if f"lora_A.{self.adapter_name}" in n:
#                 entry_ipt = self._element_score(n)
#                 comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
#                 name_m = n.replace("lora_A", "%s")
#                 if name_m not in vector_ipt:
#                     vector_ipt[name_m] = [comb_ipt]
#                 else:
#                     vector_ipt[name_m].append(comb_ipt)
#             if f"lora_B.{self.adapter_name}" in n:
#                 entry_ipt = self._element_score(n)
#                 comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
#                 name_m = n.replace("lora_B", "%s")
#                 if name_m not in vector_ipt:
#                     vector_ipt[name_m] = [comb_ipt]
#                 else:
#                     vector_ipt[name_m].append(comb_ipt)
#             if f"lora_E.{self.adapter_name}" in n:
#                 entry_ipt = self._element_score(n)
#                 name_m = n.replace("lora_E", "%s")
#                 value_ipt[name_m] = entry_ipt

#         all_score = []
#         # Calculate the score for each triplet
#         for name_m in vector_ipt:
#             ipt_E = value_ipt[name_m]
#             ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
#             sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
#             name_E = name_m % "lora_E"
#             triplet_ipt[name_E] = sum_ipt.view(-1, 1)
#             all_score.append(sum_ipt.view(-1))

#         # Get the threshold by ranking ipt
#         mask_threshold = torch.kthvalue(
#             torch.cat(all_score),
#             k=self.init_bgt - budget,
#         )[0].item()

#         rank_pattern = {}
#         # Mask the unimportant triplets
#         with torch.no_grad():
#             for n, p in model.named_parameters():
#                 if f"lora_E.{self.adapter_name}" in n:
#                     p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
#                     rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
#         return rank_pattern

#     def update_and_allocate(self, model, global_step, force_mask=False):
#         # # Update the importance score and allocate the budget
#         if global_step < self.peft_config.total_step - self.peft_config.tfinal:
#             self.update_ipt(model)
#         budget, mask_ind = self.budget_schedule(global_step)
#         # Allocate the budget according to importance scores
#         if mask_ind or force_mask:
#             rank_pattern = self.mask_to_budget(model, budget)
#         else:
#             rank_pattern = None
#         return budget, rank_pattern

#     def mask_using_rank_pattern(self, model, rank_pattern):
#         # Mask the unimportant triplets
#         is_adapter_name_truncated = False
#         if self.adapter_name not in next(iter(rank_pattern.keys())):
#             is_adapter_name_truncated = True

#         with torch.no_grad():
#             for n, p in model.named_parameters():
#                 if f"lora_E.{self.adapter_name}" in n:
#                     key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
#                     mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
#                     p.masked_fill_(~mask.bool(), 0.0)


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, token=False):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        self.token = token
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        # inp_exp = inp[self._batch_index].squeeze(1)

        if self.token:
            inp_exp = inp[self._batch_index].squeeze(1)
        else:
            inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if self.token:
                stitched = stitched.mul(self._nonzero_gates)
            else:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), *expert_out[-1].shape[1:], requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts

        try:
            combined = zeros.index_add(0, self._batch_index, stitched.float())
        except RuntimeError:
            import ipdb
            ipdb.set_trace()
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        # return combined.log()
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)