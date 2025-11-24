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
                 alpha_scale: nn.Parameter = None,
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

class FuseConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
        alpha: nn.Parameter = None,
        alpha_scale: nn.Parameter = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.alpha = alpha # given by agent
        self.alpha_scale = alpha_scale # given by agnet
        self._bias = bias
        self.num_weights = num_weights # size of tau

        if self.num_weights > 0:
            # alpha need to match num_weights
            assert(self.alpha.shape[0] == self.num_weights)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        
        # tau = {theta_0,theta_1,...theta_n}
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.zeros_like((self.weight)) for _ in range(num_weights)], dim=0), requires_grad=False)
        else:
            self.weights = None
        
        if bias:
            # tau = {theta_0,theta_1,...theta_n}
            if self.num_weights > 0:
                self.biaes = Parameter(torch.stack([torch.zeros_like(self.bias) for _ in range(num_weights)], dim=0), requires_grad=False)
            else:
                self.biaes = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        # alpha * tau {theta_0,theta_1,...theta_n} + base
        if self.alpha is not None:
            # logger.debug(f"Alpha is {self.alpha.data}, forward with alpha * tau")
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            weight = self.weight + (alphas_normalized.view(-1, 1, 1, 1, 1) * self.weights).sum(dim = 0)
            if self._bias:
                bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0)
            else:
                bias = None
        else:
            # logger.debug("Alpha is None, forward with base weight only")
            weight = self.weight
            if self._bias:
                bias = self.bias
                
        return self._conv_forward(input, weight, bias)

    @torch.no_grad()
    def merge_weight(self):
        if self.num_weights <= 0:
            logger.debug("Not weights or alpha exists, return original weight")
            return
        # logger.debug(f"Merging FuseConv: {self.weight.shape} + {self.weights.shape} * {self.alpha.shape}")
        alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
        # weight = self.weight.data # debug
        self.weight.data = self.weight.data + (alphas_normalized.view(-1, 1, 1, 1, 1) * self.weights.data).sum(dim = 0)
        if self._bias:
            self.bias.data = self.bias.data + (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)
        # logger.debug(weight == self.weight.data) # debug

    def set_base_and_vectors(self, base, vectors):
        # Set base weight
        if base is not None:
            # logger.debug(f"Setting base with tensor's shape = {base['weight'].shape}")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
        else:
            logger.debug(f"Base is None, train base weight from scratch")
            
        # Set vectors weight
        if vectors is not None: 
            # logger.debug(f"Setting vectors with tensor's shape = {vectors['weight'].shape}")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])

class FuseEncoder(nn.Module):
    def __init__(self, hidden_dim=512, layer_init=lambda x, **kwargs: x,
                    num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                    alpha: nn.Parameter = None,
                    alpha_scale: nn.Parameter = None,
                    global_alpha: bool = True):
        super().__init__()
        self.fuse_layers = [0,2,4,7]
        if global_alpha or num_weights == 0:
            self.network = nn.Sequential(
                layer_init(FuseConv2d(4, 32, 8, stride=4, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 0
                nn.ReLU(),
                layer_init(FuseConv2d(32, 64, 4, stride=2, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 2
                nn.ReLU(),
                layer_init(FuseConv2d(64, 64, 3, stride=1, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 4
                nn.ReLU(),
                nn.Flatten(),
                layer_init(FuseLinear(64 * 7 * 7, hidden_dim,alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 7
                nn.ReLU(),
            )
        else:
            # logger.debug("FuseEncoder using local alphas")
            self.alphas = ParameterList([Parameter(alpha.clone().detach().requires_grad_(alpha.requires_grad)) for _ in range(len(self.fuse_layers))])
            self.alpha_scales = ParameterList([Parameter(alpha_scale.clone().detach().requires_grad_(alpha_scale.requires_grad)) for _ in range(len(self.fuse_layers))])
            # logger.debug(f"{self.alphas}")
            self.network = nn.Sequential(
                layer_init(FuseConv2d(4, 32, 8, stride=4, alpha=self.alphas[0], alpha_scale=self.alpha_scales[0],num_weights=num_weights)), # 0
                nn.ReLU(),
                layer_init(FuseConv2d(32, 64, 4, stride=2, alpha=self.alphas[1], alpha_scale=self.alpha_scales[1],num_weights=num_weights)), # 2
                nn.ReLU(),
                layer_init(FuseConv2d(64, 64, 3, stride=1, alpha=self.alphas[2], alpha_scale=self.alpha_scales[2],num_weights=num_weights)), # 4
                nn.ReLU(),
                nn.Flatten(),
                layer_init(FuseLinear(64 * 7 * 7, hidden_dim,alpha=self.alphas[3], alpha_scale=self.alpha_scales[3],num_weights=num_weights)), # 7
                nn.ReLU(),
            )
     
    def load_base_and_vectors(self, base_dir, vector_dirs):   
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/encoder.pt")
            base_state_dict = torch.load(f"{base_dir}/encoder.pt").state_dict()
            prefix = list(base_state_dict.keys())[0].split('.')[0]
            # logger.debug(prefix)
            for i in self.fuse_layers:
                base.append({"weight":base_state_dict[f"{prefix}.{i}.weight"],"bias":base_state_dict[f"{prefix}.{i}.bias"]})
        else:
            return [None,None],[None,None]

        for idx,i in enumerate(self.fuse_layers):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                if idx == 0:
                    # logger.debug(f"Loading vectors from {p}/encoder.pt")
                    ...
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/encoder.pt").state_dict()
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
        # logger.debug("Setting FuseEncoder's weight and vectors")
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

class FuseActor(nn.Module):
    def __init__(self, hidden_dim=512, n_actions=0, layer_init=lambda x, **kwargs: x,
                    num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                    alpha: nn.Parameter = None,
                    alpha_scale: nn.Parameter = None,
                    global_alpha: bool = True):
        super().__init__()
        self.fuse_layers = [0,2]
        if global_alpha or num_weights == 0:
            self.network = nn.Sequential(
                layer_init(FuseLinear(hidden_dim, hidden_dim, num_weights=num_weights, 
                                    alpha=alpha, alpha_scale=alpha_scale)),
                nn.ReLU(),
                layer_init(FuseLinear(hidden_dim, n_actions, num_weights=num_weights, 
                                    alpha=alpha, alpha_scale=alpha_scale),std=0.01),
            )
        else:
            logger.debug("FuseActor using local alphas")
            self.alphas = ParameterList([Parameter(alpha.clone().detach().requires_grad_(alpha.requires_grad)) for _ in range(len(self.fuse_layers))])
            self.alpha_scales = ParameterList([Parameter(alpha_scale.clone().detach().requires_grad_(alpha_scale.requires_grad)) for _ in range(len(self.fuse_layers))])
            logger.debug(f"{self.alphas}")
            self.network = nn.Sequential(
                layer_init(FuseLinear(hidden_dim, hidden_dim, num_weights=num_weights, 
                                    alpha=self.alphas[0], alpha_scale=self.alpha_scales[0])),
                nn.ReLU(),
                layer_init(FuseLinear(hidden_dim, n_actions, num_weights=num_weights, 
                                    alpha=self.alphas[1], alpha_scale=self.alpha_scales[1]),std=0.01),
            )
        
    def load_base_and_vectors(self, base_dir, vector_dirs):
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/actor.pt")
            base_state_dict = torch.load(f"{base_dir}/actor.pt").state_dict()
            prefix = list(base_state_dict.keys())[0].split('.')[0]
            for i in self.fuse_layers:
                base.append({"weight":base_state_dict[f"{prefix}.{i}.weight"],"bias":base_state_dict[f"{prefix}.{i}.bias"]})
        else:
            return [None,None],[None,None]

        for idx,i in enumerate(self.fuse_layers):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                if idx == 0:
                    # logger.debug(f"Loading vectors from {p}/actor.pt")
                    ...
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/actor.pt").state_dict()
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
            
    def get_vectors(self, base):
        vectors = []
        for idx, i in enumerate(self.fuse_layers):
            i_vectors, num_vectors = self.network[i].get_vectors(base=base[idx])
            vectors.append(i_vectors)
        return vectors, num_vectors
    
    def get_base(self):
        base = []
        for i in self.fuse_layers:
            base.append({"weight":self.network[i].weight.data,"bias":self.network[i].bias.data})
        return base

    def setup_fuse_layers(self, base, vectors):
        for idx,i in enumerate(self.fuse_layers):
            self.network[i].set_base_and_vectors(base[idx],vectors[idx])

class Actor(nn.Module):
    def __init__(self, hidden_dim=512, n_actions=0, layer_init=lambda x, **kwargs: x):
        super().__init__()
        self.network = nn.Sequential(
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, n_actions),std=0.01),
            )

    def forward(self, x):
        return self.network(x)
