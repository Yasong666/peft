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
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from peft.utils.other import transpose

class PloraLayer(BaseTunerLayer):
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_givens = nn.ParameterDict()
        self.W_1 = nn.ParameterDict()
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, r, lora_dropout, init_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        angles = torch.zeros(self.in_features // 2)#整除
        self.lora_givens[adapter_name] = nn.Parameter(angles)
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        for param in self.lora_A[adapter_name].parameters():
            param.requires_grad = True  # 设置为可训练
        if adapter_name not in self.W_1:
            self.W_1[adapter_name] = nn.Parameter(torch.empty((self.out_features, self.in_features), dtype=torch.float32))
        if init_weights:
            with gather_params_ctx(self.get_base_layer().weight):
                self.plora_init(adapter_name, init_weights)
        # else:
        #     self.reset_adapter(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def plora_init(self, adapter_name, init_weights):
        weight = self.get_base_layer().weight
        # dtype = weight.dtype
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
        # Vr = V[:, : self.r[adapter_name]]
        Sr = S[: self.r[adapter_name]]
        # Sr /= self.scaling[adapter_name]
        Uhr = Uh[: self.r[adapter_name]]
        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr

        W_0 = torch.diag(torch.sqrt(S[self.r[adapter_name]:])) @ Uh[self.r[adapter_name]:]
        W_0 = W_0.T @ W_0
        W_ = V@Uh
        self.lora_A[adapter_name].weight.data = lora_A
        with torch.no_grad():
            self.W_1[adapter_name].data.copy_(W_)
        self.get_base_layer().weight.data = W_0


class PloraLinear(nn.Module, PloraLayer):
    def __init__(
      self,
      base_layer,
      adapter_name:str,
      r:int = 0,
      lora_dropout:float=  0.0,
      fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
      init_weights:Union[bool, str] = True,
      **kwargs
    ):
        super().__init__()
        PloraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_dropout, init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return #no adapter to merge

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(adapter = active_adapter)
                    orig_weights += delta_weight
                    orig_weights = self.get_rotation_matric(self.lora_givens[active_adapter].data)@orig_weights
                    orig_weights = torch.mm(self.W_1[active_adapter].data,orig_weights)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights.contiguous()
                else:
                    # base_layer.weight.data += self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(adapter = active_adapter)
                    orig_weights += delta_weight
                    orig_weights = self.get_rotation_matric(self.lora_givens[active_adapter].data)@orig_weights
                    orig_weights = torch.mm(self.W_1[active_adapter].data,orig_weights)
                    base_layer.weight.data = orig_weights.contiguous()

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adpaters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                orig_weights = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                orig_weights = torch.mm(self.W_1[active_adapter].data.T,orig_weights)
                orig_weights = self.get_rotation_matric(self.lora_givens[active_adapter].data).T@orig_weights
                orig_weights.data -= delta_weight

                self.get_base_layer().weight.data = orig_weights.contiguous()

    def get_delta_weight(self, adapter) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        output_tensor = transpose(weight_A.T@weight_A, fan_in_fan_out=self.fan_in_fan_out)
        return output_tensor

    def get_rotation_matric(self, angle:torch.tensor):
            n = len(angle)
            rotation = torch.eye(2*n)
            for index in range(n):
                rotation[2*index,2*index] = torch.cos(angle[index])
                rotation[2*index,2*index+1] = torch.sin(angle[index])
                rotation[2*index+1,2*index] = -torch.sin(angle[index])
                rotation[2*index+1,2*index+1] = torch.cos(angle[index])
            return rotation

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)#这里以后要改。
            torch_result_dtype = result.dtype
            rotation = torch.eye(self.in_features, device=x.device, dtype=x.dtype)
            delta_w = torch.rand(self.in_features,self.in_features, device=x.device, dtype=x.dtype)
            origin_weight = self.get_base_layer().weight.data
            for active_adapter in self.active_adapters:
                # print("forward"*10)
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                rotation = rotation.to(device=lora_A.weight.device)
                x = x.to(lora_A.weight.dtype)
                # result = result + lora_A(dropout(x)) @ lora_A.weight.T @
                mat = self.get_rotation_matric(self.lora_givens[active_adapter].data).to(rotation.device)
                rotation = mat@rotation
                delta_w = rotation @self.get_delta_weight(active_adapter)
                new_weight = origin_weight + delta_w
                ro_weight = torch.mm(rotation, new_weight)
                new_weight = self.W_1[active_adapter].data@ro_weight
                result = torch.matmul(x, new_weight)
            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "plora." + rep