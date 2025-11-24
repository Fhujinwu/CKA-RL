import torch
import torch.nn as nn
import os
from .fuse_module import FuseShared, FuseLinear
from .shared_arch import shared
from loguru import logger
import numpy as np

class CkaRlAgent(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 base_dir, 
                 latest_dir, 
                 pool_size = 2,
                 delta_theta_mode = "T", 
                 global_alpha = True, 
                 alpha_init = "Randn", 
                 alpha_major = 0.6, 
                 alpha_factor = 1e-3,
                 fix_alpha = False,
                 reset_heads = False,
                 encoder_from_base = False,
                 use_alpha_scale = True,
                 fuse_shared = False, 
                 fuse_heads = True,):
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.global_alpha = global_alpha
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.fuse_shared = fuse_shared
        self.fuse_heads = fuse_heads
        self.pool_size = pool_size

        assert(fuse_heads or fuse_shared)
        self.setup_vectors(base_dir, latest_dir)
        # Alpha Setting
        self.setup_alpha(num_vectors=self.num_vectors, 
                         fix_alpha=fix_alpha,alpha_init=alpha_init,
                         alpha_major=alpha_major,alpha_factor=alpha_factor,
                         use_alpha_scale=use_alpha_scale)
        
        self.log_alpha()
        
        if encoder_from_base and base_dir is not None:
            logger.info(f"Loading encoder from {base_dir}")
            self.fc = torch.load(f"{base_dir}/fc.pt")
        elif latest_dir is not None:
            logger.info(f"Loading latest shared from {latest_dir}")
            # self.fc = shared(input_dim=obs_dim)
            self.fc = torch.load(f"{latest_dir}/fc.pt")
        else:
            logger.info("Train shared from scratch")
            self.fc = shared(input_dim=obs_dim)
            
        self.setup_heads()

    def setup_heads(self):
        if self.fuse_heads:
            logger.debug("CKA-RL fuse heads")
            self.fc_mean = FuseLinear(256, 
                                    self.act_dim, alpha=self.alpha, 
                                    alpha_scale=self.alpha_scale, 
                                    num_weights=self.num_vectors)
            self.fc_logstd = FuseLinear(256, 
                                        self.act_dim, alpha=self.alpha, 
                                        alpha_scale=self.alpha_scale, 
                                        num_weights=self.num_vectors)
            if self.num_vectors > 0:
                logger.info("Set base and vectors for fc_mean")
                logger.info("Set base and vectors for fc_logstd")
                self.fc_mean.set_base_and_vectors(self.fc_mean_base, self.fc_mean_vectors)
                self.fc_logstd.set_base_and_vectors(self.fc_logstd_base, self.fc_logstd_vectors)
        else:
            self.fc_mean = nn.Linear(256, self.act_dim)
            self.fc_logstd = nn.Linear(256, self.act_dim)
        

    def load_base_and_vectors(self, base_dir, vector_dirs, module_name):
        num_weights = 0
        base = None
        vectors = None
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/model.pt")
            base_state_dict = torch.load(f"{base_dir}/model.pt").state_dict()
            base = {"weight":base_state_dict[f"{module_name}.weight"],"bias":base_state_dict[f"{module_name}.bias"]}
        else:
            return None, None

        vector_weight = []
        vector_bias = []
        for p in vector_dirs:
            logger.debug(f"Loading vectors from {p}/model.pt")
            # load theta_i + base weight from prevs
            vector_state_dict = torch.load(f"{p}/model.pt").state_dict()
            # get theta_i
            vector_weight.append(base['weight'] - vector_state_dict[f"{module_name}.weight"])
            vector_bias.append(base['bias'] - vector_state_dict[f"{module_name}.bias"])
        vectors = {"weight":torch.stack(vector_weight),
                        "bias":torch.stack(vector_bias)}
        num_weights += vectors["weight"].shape[0] if vectors else 0
        return base, vectors

    def heads_set_base_and_vectors(self, base_dir, prevs_paths):
        for module_name in ["fc_mean", "fc_logstd"]:
            base, vectors = self.load_base_and_vectors(base_dir, prevs_paths, module_name)
            if base is None:
                continue
            getattr(self, module_name).set_base_and_vectors(base, vectors)

    def set_base_and_vectors(self, base_dir, prevs_paths):
        if self.fuse_shared:
            self.fc.set_base_and_vectors(base_dir, prevs_paths)
        if self.fuse_heads:
            self.heads_set_base_and_vectors(base_dir, prevs_paths)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        # for actor, merge `theta + alpha * tau` to `theta` if delta_theta_mode  == 'TAT'
        if self.delta_theta_mode == "TAT":
            self.merge_weight()
        else:
            logger.info("save weight as theta")
        vectors, n = self.fc_mean.get_vectors()
        logger.info(f"vectors' shape {n}")
        torch.save(self.fc, f"{dirname}/fc.pt")
        torch.save(self.fc_mean, f"{dirname}/fc_mean.pt")
        torch.save(self.fc_logstd, f"{dirname}/fc_logstd.pt")

    def load(dirname,
             obs_dim, 
            act_dim, 
            map_location=None, reset_heads=False):
        model = CkaRlAgent(obs_dim,act_dim,None,None)
        model.fc = torch.load(f"{dirname}/fc.pt", map_location=map_location)
        model.fc_mean = torch.load(f"{dirname}/fc_mean.pt", map_location=map_location)
        model.fc_logstd = torch.load(f"{dirname}/fc_logstd.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model

    def merge_weight(self):
        if self.fuse_shared:
            self.fc.merge_weight()
        if self.fuse_heads:
            self.fc_mean.merge_weight()
            self.fc_logstd.merge_weight()
            
    def setup_vectors(self, base_dir, latest_dir):
        if base_dir == None:
            self.num_vectors = 0
        elif latest_dir == None:
            self.num_vectors = 1
        else:
            if self.fuse_heads:
                logger.debug("Setup head's vectors")
                # mean
                base_model = torch.load(f"{base_dir}/fc_mean.pt")
                self.fc_mean_base = base_model.get_base()
                latest_model = torch.load(f"{latest_dir}/fc_mean.pt", map_location=None)
                self.fc_mean_vectors, self.num_vectors = latest_model.get_vectors(self.fc_mean_base)
                # logstd
                base_model = torch.load(f"{base_dir}/fc_logstd.pt")
                self.fc_logstd_base = base_model.get_base()
                latest_model = torch.load(f"{latest_dir}/fc_logstd.pt", map_location=None)
                self.fc_logstd_vectors, self.num_vectors = latest_model.get_vectors(self.fc_logstd_base)
                logger.info(f"self.num_vectors:{self.num_vectors}")
                self.merge_vectors(self.fc_mean_vectors, self.fc_logstd_vectors)
                logger.debug(self.fc_mean_vectors['weight'].shape)
                logger.debug(self.fc_logstd_vectors['weight'].shape)
            
    def setup_alpha(self, num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor, use_alpha_scale):
        if num_vectors > 0:
            if fix_alpha: # Alpha is untrainable
                self.alpha = nn.Parameter(torch.zeros(self.num_vectors), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else: # Alpha is trainable
                logger.info(f"alpha_init, {alpha_init}")
                logger.info(f"alpha_major, {alpha_major}")
                if alpha_init == "Uniform" or self.num_vectors == 1:
                    self.alpha = nn.Parameter(torch.ones(self.num_vectors) * alpha_factor, requires_grad=True)
                elif alpha_init == "Randn":
                    self.alpha = nn.Parameter(torch.randn(self.num_vectors) / self.num_vectors, requires_grad=True)
                elif alpha_init == "Major" and self.num_vectors > 1:
                    alpha = [np.log((1-alpha_major)/(self.num_vectors-1)) for _ in range(self.num_vectors-1)]
                    alpha.append(np.log(alpha_major))
                    self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float), requires_grad=True)
                    logger.info(self.alpha)
                elif alpha_init not in ["Uniform", "Randn", "Major"]:
                    raise NotImplementedError
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
                logger.info("Train alpha")
            if not use_alpha_scale or fix_alpha:
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=False)
            self.log_alpha()
        else:
            self.alpha = None
            self.alpha_scale = None
            
    def log_alpha(self):
        logger.info(self.alpha)
        
        
    def merge_vectors(self, mean_vectors,logstd_vectors):
        def merge(vectors):
            for name, element in vectors.items():
                similarities = torch.ones((element.shape[0], element.shape[0])) * -1
                for i in range(element.shape[0]):
                    for j in range(i + 1, element.shape[0]):
                        similarities[i, j] = torch.cosine_similarity(element[i].flatten(), element[j].flatten(), dim=0)
                print(similarities)                
                max_sim_idx = torch.argmax(similarities)
                idx1, idx2 = divmod(max_sim_idx.item(), element.shape[0])
                logger.info(f"Merge vectors, name = {name}, idx1 = {idx1}, idx2 = {idx2}")
                new_element = (element[idx1] + element[idx2]) / 2
                element = torch.cat((element[:idx1], element[idx1+1:idx2], element[idx2+1:], new_element.unsqueeze(0)), dim=0)
                logger.info(element.shape)
                vectors[name] = element
        if self.num_vectors > self.pool_size:
            logger.info(f"Merge vectors, pool size = {self.pool_size}, current #vectors = {self.num_vectors}")
            merge(mean_vectors)
            merge(logstd_vectors)
            self.num_vectors = self.pool_size
