import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList, init
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union
from loguru import logger

class FuseLinear(nn.Module):
    r"""Applies a linear transformation to the 
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                 alpha: nn.Parameter = None,
                 alpha_scale: nn.Parameter = torch.tensor(1.0, requires_grad=False),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha # given by agent
        self.alpha_scale = alpha_scale # given by agnet
        self._bias = bias
        self.num_weights = num_weights # size of tau
        
        if self.num_weights > 0:
            # alpha need to match num_weights
            assert(self.alpha.shape[0] == self.num_weights)
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=True)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        
        # tau = {theta_0,theta_1,...theta_n}
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.empty((out_features, in_features)) for _ in range(num_weights)], dim=0), requires_grad=False)
        else:
            self.weights = None
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)

            # tau = {theta_0,theta_1,...theta_n}
            if self.num_weights > 0:
                self.biaes = Parameter(torch.stack([torch.empty(out_features) for _ in range(num_weights)], dim=0), requires_grad=False)
            else:
                self.biaes = None
        else:
            self.register_parameter('bias', None)
            if self.num_weights > 0:
                self.register_parameter('biases', None)
        self.reset_parameters()

    @torch.no_grad()
    def merge_weight(self):
        if self.num_weights <= 0:
            logger.debug("Not weights or alpha exists, return original weight")
            return
        # logger.debug(f"Merging FuseLinear: {self.weight.shape} + {self.weights.shape} * {self.alpha.shape}")
        alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
        # weight = self.weight.data # debug
        self.weight.data = self.weight.data + (alphas_normalized.view(-1, 1, 1) * self.weights.data).sum(dim = 0)
        if self._bias:
            self.bias.data = self.bias.data + (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)
        # logger.debug(weight == self.weight.data) # debug

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # alpha * tau {theta_0,theta_1,...theta_n} + base
        if self.alpha is not None:
            # logger.debug(f"Alpha is {self.alpha.data}, forward with alpha * tau")
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            weight = self.weight + (alphas_normalized.view(-1, 1, 1) * self.weights).sum(dim = 0)
            if self._bias:
                bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0)
            else:
                bias = None
        else:
            # logger.debug("Alpha is None, forward with base weight only")
            weight = self.weight
            if self._bias:
                bias = self.bias
                
        return F.linear(input, weight, bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, num_weights={self.num_weights}'   
        
    def set_base_and_vectors(self, base, vectors):
        # Set base weight
        if base is not None:
            # logger.debug(f"Setting base with tensor's shape = {base['weight'].shape}")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
        else:
            # logger.debug(f"Base is None, train base weight from scratch")
            return
            
        # Set vectors weight
        if vectors is not None: 
            # logger.debug(f"Setting vectors with tensor's shape = {vectors['weight'].shape}")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])

    def get_vectors(self, base = None):
        if base is None:
            base_weight = torch.zeros_like(self.weight)
            base_bias = torch.zeros_like(self.bias)
        else:
            base_weight = base['weight']
            base_bias = base['bias']
        new_weight = self.weight - base_weight
        new_bias = self.bias - base_bias
        
        if self.weights is not None:
            weights = torch.cat([new_weight.unsqueeze(0), self.weights], dim=0)
        else:
            weights = new_weight.unsqueeze(0)
        if self.biaes is not None:
            biaes = torch.cat([new_bias.unsqueeze(0), self.biaes], dim=0)
        else:
            biaes = new_bias.unsqueeze(0)
            
        return {"weight":weights, "bias":biaes}, weights.shape[0]
    
    def get_base(self):
        # return {"weight": self.weight.data, "bias": self.bias.data if self.bias is not None else None}
        return {"weight":self.weight, "bias":self.bias}

class FuseShared(nn.Module):
    def __init__(self, input_dim, 
                    layer_init=lambda x, **kwargs: x,
                    num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                    alpha: nn.Parameter = None,
                    alpha_scale: nn.Parameter = torch.tensor(1.0, requires_grad=False),
                    global_alpha: bool = True):
        super().__init__()
        self.fuse_layers = [0,2]
        if global_alpha or num_weights == 0:
            self.network = nn.Sequential(
                FuseLinear(input_dim, 256, 
                            num_weights=num_weights, 
                            alpha=alpha, alpha_scale=alpha_scale),
                nn.ReLU(),
                FuseLinear(256, 256,             
                            num_weights=num_weights, 
                            alpha=alpha, alpha_scale=alpha_scale),
                nn.ReLU(),
            )
        else:
            logger.debug("FuseShared using local alphas")
            self.alphas = ParameterList([Parameter(alpha.clone().detach().requires_grad_(alpha.requires_grad)) for _ in range(len(self.fuse_layers))])
            self.alpha_scales = ParameterList([Parameter(alpha_scale.clone().detach().requires_grad_(alpha_scale.requires_grad)) for _ in range(len(self.fuse_layers))])
            logger.debug(f"{self.alphas}")
            self.network = nn.Sequential(
                FuseLinear(input_dim, 256, num_weights=num_weights, 
                alpha=self.alphas[0], alpha_scale=self.alpha_scales[0]),
                nn.ReLU(),
                FuseLinear(256, 256, num_weights=num_weights, 
                                    alpha=self.alphas[1], alpha_scale=self.alpha_scales[1]),
                nn.ReLU(),
            )
        
    def load_base_and_vectors(self, base_dir, vector_dirs):
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/model.pt")
            base_state_dict = torch.load(f"{base_dir}/model.pt").state_dict()
            prefix = list(base_state_dict.keys())[0].split('0')[0][:-1]
            for i in self.fuse_layers:
                base.append({"weight":base_state_dict[f"{prefix}.{i}.weight"],"bias":base_state_dict[f"{prefix}.{i}.bias"]})
        else:
            return [None,None],[None,None]

        for idx,i in enumerate(self.fuse_layers):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                if idx == 0:
                    logger.debug(f"Loading vectors from {p}/model.pt")
                    ...
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/model.pt").state_dict()
                # get theta_i
                vector_weight.append(base[idx]['weight'] - vector_state_dict[f"{prefix}.{i}.weight"])
                vector_bias.append(base[idx]['bias'] - vector_state_dict[f"{prefix}.{i}.bias"])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
        num_weights += vectors[0]["weight"].shape[0] if vectors else 0
        return base, vectors
        
    def set_base_and_vectors(self, base_dir, prevs_paths):
        base, vectors = self.load_base_and_vectors(base_dir, prevs_paths)
        if base[0] is None:
            logger.warning("Not base or vectors exist")
            return 
        # logger.debug("Setting FuseActor's weight and vectors")
        for idx,i in enumerate(self.fuse_layers):
            self.network[i].set_base_and_vectors(base[idx],vectors[idx])
        
    def forward(self, x):
        return self.network(x)
    
    def merge_weight(self):
        for i in self.fuse_layers:
            self.network[i].merge_weight()
            
    def log_alphas(self):
        for i in self.fuse_layers:
            normalized_alpha = F.softmax(self.network[i].alpha * self.network[i].alpha_scale, dim=0)
            logger.info(f"Layer {i} alpha: {normalized_alpha}")

