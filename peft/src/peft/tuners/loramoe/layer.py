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
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
# from transformers.pytorch_utils import Conv1D
from torch.distributions.normal import Normal

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from peft.tuners.lora import LoraLayer


class LoraMoELinear(nn.Linear, LoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,        
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        LoraLayer.__init__(self, base_layer=base_layer, in_features=in_features, out_features=out_features)
        
        # Freezing the pre-trained weight matrix        
        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora=False)
        self._active_adapter = adapter_name

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        
        self.lora_alpha = self.lora_alpha[adapter_name]
        self.r = self.r[adapter_name]
        self.lora_dropout = self.lora_dropout[adapter_name]
        
        # Remove default loras
        # del self.lora_A
        # del self.lora_B

        # Actual trainable parameters
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, task_ids=None):

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(") 
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            if self.r > 0:
                route_weight = nn.functional.softmax(self.lora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)

                # for i in range(self.lora_num):
                #     result = result + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                #     BAx = getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                #     result = result + torch.unsqueeze(route_weight[:,:,i], -1) * BAx
                
                results = []
                for i in range(self.lora_num):
                    route_w = torch.unsqueeze(route_weight[:,:,i], -1)

                    Ax = getattr(self, f"lora_A{i}")(self.lora_dropout(x))
                    BAx = getattr(self, f"lora_B{i}")(Ax) * self.scaling

                    results.append(route_w * BAx)

                stacked_results = torch.stack(results)
                result = result + stacked_results.sum(dim=0)

        blcls = torch.zeros(1)[0].to(result)
        if task_ids != None:
            if self.blc_weight != 0:
                task_ids = task_ids.view(-1, 1)
                blcls = self.cv_squared((
                    route_weight.sum(dim=(1)) * torch.where(
                        torch.concat(
                            ((task_ids==1).repeat(1, self.lora_num//2), (task_ids==0).repeat(1, self.lora_num//2)), dim=-1
                            ), 1.0+self.blc_alpha, 1.0-self.blc_alpha
                        )
                    ).flatten()
                ) * self.blc_weight
                                
        # return result, blcls
        return result
        
#     def merge(self) -> None:
#         if self.merged:
#             warnings.warn(
#                 f"Already following adapters were merged {','.join(self.merged_adapters)}. "
#                 f"You are now additionally merging {','.join(self.active_adapters)}."
#             )
#         for active_adapter in self.active_adapters:
#             if active_adapter in self.lora_A.keys():
#                 self.weight.data += self.get_delta_weight(active_adapter)
#                 self.merged_adapters.append(active_adapter)
#                 self.merged = True

#     def unmerge(self) -> None:
#         if not self.merged:
#             warnings.warn("Already unmerged. Nothing to do.")
#             return
#         while len(self.merged_adapters) > 0:
#             active_adapter = self.merged_adapters.pop()
#             if active_adapter in self.lora_A.keys():
#                 self.weight.data -= self.get_delta_weight(active_adapter)
#                 self.merged = False

#     def get_delta_weight(self, adapter) -> torch.Tensor:
#         return (
#             transpose(
#                 self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
#                 self.fan_in_fan_out,
#             )
#             * self.scaling[adapter]
#         )

#     def _linear(self, input: torch.Tensor) -> torch.Tensor:
#         return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#     def _gates_to_load(self, gates):
#         """Compute the true load per expert, given the gates.
#         The load is the number of examples for which the corresponding gate is >0.
#         Args:
#         gates: a `Tensor` of shape [batch_size, n]
#         Returns:
#         a float32 `Tensor` of shape [n]
#         """
#         return (gates > 0).sum(0)

#     def _prob_in_top_k(
#             self, clean_values, noisy_values, noise_stddev, noisy_top_values
#     ):
#         """Helper function to NoisyTopKGating.
#         Computes the probability that value is in top k, given different random noise.
#         This gives us a way of backpropagating from a loss that balances the number
#         of times each expert is in the top k experts per example.
#         In the case of no noise, pass in None for noise_stddev, and the result will
#         not be differentiable.
#         Args:
#         clean_values: a `Tensor` of shape [batch, n].
#         noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
#           normally distributed noise with standard deviation noise_stddev.
#         noise_stddev: a `Tensor` of shape [batch, n], or None
#         noisy_top_values: a `Tensor` of shape [batch, m].
#            "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
#         Returns:
#         a `Tensor` of shape [batch, n].
#         """

#         batch = clean_values.size(0)
#         m = noisy_top_values.size(1) # (B*50, top_k+1)
#         top_values_flat = noisy_top_values.flatten()
#         threshold_positions_if_in = (
#                 torch.arange(batch, device=clean_values.device) * m + self.topk
#         )
#         threshold_if_in = torch.unsqueeze(
#             torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
#         ) # (B*50, 1)
#         is_in = torch.gt(noisy_values, threshold_if_in)  # (B*50, 4) similar to gate results.
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         threshold_if_out = torch.unsqueeze(
#             torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
#         )
#         # is each value currently in the top k.
#         normal = Normal(
#             torch.tensor([0.0], device=clean_values.device),
#             torch.tensor([1.0], device=clean_values.device),
#         )
#         prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
#         prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
#         prob = torch.where(is_in, prob_if_in, prob_if_out)
#         return prob

#     def cv_squared(self, x):
#         """The squared coefficient of variation of a sample.
#         Useful as a loss to encourage a positive distribution to be more uniform.
#         Epsilons added for numerical stability.
#         Returns 0 for an empty Tensor.
#         Args:
#         x: a `Tensor`.
#         Returns:
#         a `Scalar`.
#         """
#         eps = 1e-10
#         # if only num_expert = 1
#         if x.shape[0] == 1:
#             return torch.Tensor([0])
#         return x.float().var() / (x.float().mean() ** 2 + eps)

#     def forward(self, x: torch.Tensor, task_ids) -> torch.Tensor:
#         previous_dtype = x.dtype

#         if self.disable_adapters:
#             if self.merged:
#                 self.unmerge()
#             result = self._linear(x)
#         elif self.merged:
#             result = self._linear(x)
#         else:
#             result = self._linear(x)

#             if self.multiple_loras:
#                 if self.cluster:
#                     # device = x.device
                    
#                     # task_ids = torch.tensor(self.kmeans.predict(self.input_emb), device=device)
#                     emb = self.task_emb(task_ids)
                    
#                     clean_logits = self.w_gate(emb)
#                     raw_noise_stddev = self.noise_std
#                     noise_stddev = raw_noise_stddev * self.training
#                     noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
#                     logits = noisy_logits
#                     logits = F.softmax(logits / self.gates_tmp, dim=1, dtype=torch.float16)             
#                     top_logits, top_indices = logits.topk(min(self.topk+1, self.num_experts), dim=1)
#                     top_k_logits = top_logits[:, :self.topk]
#                     top_k_indices = top_indices[:, :self.topk]
#                     top_k_gates = top_k_logits

#                     zeros = torch.zeros_like(logits, requires_grad=True)
#                     gates = zeros.scatter(1, top_k_indices, top_k_gates)

#                     gate_load = gates.gt(0).sum(0)

#                     dispatcher = SparseDispatcher(self.num_experts, gates)
#                     expert_inputs = list(dispatcher.dispatch(x))
#                     gates_ = dispatcher.expert_to_gates()
                   
#                     if self.g_enable:
#                         expert_outputs = []
#                         for i, adpt in enumerate(list(self.r.keys())[:-1]):
#                             _x = x.to(self.lora_A[adpt].weight.dtype)
#                             expert_inputs[i] = expert_inputs[i].to(self.lora_A[adpt].weight.dtype)
#                             expert_outputs.append(
#                                 self.lora_B[adpt](
#                                     self.lora_A[adpt](self.lora_dropout[adpt](expert_inputs[i]))
#                                 )
#                                 * self.scaling[adpt]
#                             )
#                         y_e = dispatcher.combine(expert_outputs)
                        
#                         lora_A = self.lora_A['g']
#                         lora_B = self.lora_B['g']
#                         dropout = self.lora_dropout['g']
#                         scaling = self.scaling['g']
#                         x = x.to(lora_A.weight.dtype)
#                         y_g = lora_B(lora_A(dropout(x))) * scaling


#                         w = gates.max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
#                         y = y_e * w + y_g * (1-w)
#                     else:
#                         expert_outputs = []
#                         for i, adpt in enumerate(list(self.r.keys())):
#                             _x = x.to(self.lora_A[adpt].weight.dtype)
#                             expert_inputs[i] = expert_inputs[i].to(self.lora_A[adpt].weight.dtype)
#                             expert_outputs.append(
#                                 self.lora_B[adpt](
#                                     self.lora_A[adpt](self.lora_dropout[adpt](expert_inputs[i]))
#                                 )
#                                 * self.scaling[adpt]
#                             )
#                         y = dispatcher.combine(expert_outputs)
                    
#                     result += y 
#             else:
#                 for active_adapter in self.active_adapters:
#                     if active_adapter not in self.lora_A.keys():
#                         continue
#                     lora_A = self.lora_A[active_adapter]
#                     lora_B = self.lora_B[active_adapter]
#                     dropout = self.lora_dropout[active_adapter]
#                     scaling = self.scaling[active_adapter]
#                     x = x.to(lora_A.weight.dtype)
#                     result += lora_B(lora_A(dropout(x))) * scaling

#         result = result.to(previous_dtype)
#         return result


# class _MoELoraLinear(LoraLayer):

#     def __init__(self, 
#                  base_adapter: nn.Module,
#                  adapter_name: str, 
#                  r: int = 0, 
#                  lora_alpha: int = 1, 
#                  lora_dropout: float = 0, 
#                  fan_in_fan_out: bool = False, 
#                  **kwargs):

#         super().__init__(base_adapter, adapter_name, r, lora_alpha, lora_dropout, fan_in_fan_out, **kwargs)

#     def unmerge(self, expert_weight):
#         if self.active_adapter not in self.lora_A.keys():
#             return
#         if not self.merged:
#             warnings.warn("Already unmerged. Nothing to do.")
#             return
#         if self.r[self.active_adapter] > 0:
#             for i in range(self.expert_num):
#                 lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
#                 lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
#                 self.weight.data -= (
#                     transpose(
#                         lora_B_weights @ lora_A_weights,
#                         self.fan_in_fan_out,
#                     )
#                     * self.scaling[self.active_adapter]
#                     * expert_weight[..., i]
#                 )
#             self.merged = False

#     def forward(self, x: torch.Tensor, task_ids, **kwargs):
#         # expert_weight = kwargs["task_ids"]
#         expert_weight = task_ids
#         previous_dtype = x.dtype

#         if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
#             return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
#         if self.disable_adapters:   # No adapter
#             if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
#                 self.unmerge(expert_weight)
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
#         elif self.r[self.active_adapter] > 0 and not self.merged:   # general lora process
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
#             x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

#             for i in range(self.expert_num):
#                 result += ( # lora process
#                     self.lora_B[self.active_adapter].loraB[i](
#                         self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
#                     )
#                     * self.scaling[self.active_adapter]
#                     * expert_weight[..., i].unsqueeze(-1).unsqueeze(0)
#                 )
#         else:
#             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

#         result = result.to(previous_dtype)

#         return result


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


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: MoELoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

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
        kwargs.update(lora_config.loftq_config)
        new_module = MoELoraLinear(target, adapter_name, **kwargs)
        # new_module = MMOELoraLinear(target, adapter_name, **kwargs)
        
    return new_module
