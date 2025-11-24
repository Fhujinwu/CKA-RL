import os
import torch
import numpy as np
import torch.nn as nn
from .cnn_encoder import CnnEncoder
from .fuse_modules import FuseActor, FuseEncoder, Actor
from torch.distributions.categorical import Categorical
from loguru import logger

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
    if hasattr(layer, 'weights') and layer.weights is not None:
        torch.nn.init.zeros_(layer.weights)
    if hasattr(layer, 'biaes') and layer.biaes is not None:
        torch.nn.init.zeros_(layer.biaes)
    return layer

class CkaRlAgent(nn.Module):
    def __init__(self, envs, base_dir, latest_dir, 
                 fix_alpha: bool = False,
                 alpha_factor: float = 1/100,
                 global_alpha: bool = True,
                 delta_theta_mode: str = "T",
                 fuse_encoder: bool = False,
                 fuse_actor: bool = True,
                 reset_actor: bool = False,
                 use_alpha_scale: bool = True,
                 alpha_init: str = "Randn",
                 alpha_major: float = 0.6,
                 pool_size: int = 3,
                 map_location=None):
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.fuse_encoder = fuse_encoder
        self.global_alpha = global_alpha
        self.fuse_actor = fuse_actor
        self.pool_size = pool_size
        self.hidden_dim = 512
        self.envs = envs
        self.i = 0
        assert(fuse_encoder or fuse_actor)

        logger.info(f"FuseAgent: fuse encoder = {fuse_encoder}, fuse actor = {fuse_actor}")
        self.setup_vectors(base_dir, latest_dir)
        
        # Alpha Setting
        self.setup_alpha(num_vectors=self.num_vectors, 
                         fix_alpha=fix_alpha,alpha_init=alpha_init,
                         alpha_major=alpha_major,alpha_factor=alpha_factor,
                         use_alpha_scale=use_alpha_scale)
        
        # Actor 's fuse or not 
        if self.fuse_actor:
            logger.debug("CKA-RL adapt actor")
            self.actor = FuseActor( hidden_dim=512,
                                    layer_init=layer_init,
                                    n_actions=envs.single_action_space.n,
                                    alpha=self.alpha,
                                    alpha_scale=self.alpha_scale,
                                    num_weights=self.num_vectors,
                                    global_alpha=self.global_alpha)
            if self.num_vectors > 0:
                self.actor.setup_fuse_layers(self.actor_base, self.actor_vectors)
        else:
            if latest_dir is not None and reset_actor is False:
                logger.info(f"Loading actor from {latest_dir}")
                self.actor = torch.load(f"{latest_dir}/actor.pt", map_location=map_location)
            else:
                logger.info("Train actor from scratch")
                self.actor = Actor( hidden_dim=512,
                                    n_actions=envs.single_action_space.n,
                                    layer_init=layer_init)
        
        # TODO encoder
        if latest_dir is not None:
            logger.info(f"Loading encoder from {latest_dir}")
            self.network = torch.load(f"{latest_dir}/encoder.pt", map_location=map_location)
        else:
            logger.info("Train encoder from scratch")
            self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)

        # Critic 
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None,log_writter=None, global_step=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        # Log alpha
        if log_writter is not None and global_step is not None and self.alpha is not None:
            normalized_alpha = torch.softmax(self.alpha, dim=0)
            for i, alpha_i in enumerate(normalized_alpha):
                log_writter.add_scalar(
                    f"alpha/{i}", alpha_i.item(), global_step
                )
                
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        # for actor, merge `theta + alpha * tau` to `theta` if delta_theta_mode  == 'TAT'
        if self.delta_theta_mode == "TAT":
            if self.fuse_actor:
                logger.info(f"save actor weight as theta + alpha * tau")
                self.merge_actor_weight()
            if self.fuse_encoder:
                logger.info(f"save encoder weight as theta + alpha * tau")
                self.merge_encoder_weight()
        else:
            logger.info("save weight as theta")
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/critic.pt")

    def load(dirname, envs, load_critic=True, reset_actor=False, map_location=None):
        model = CkaRlAgent(envs, None, None,)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)            
        return model

    def merge_actor_weight(self):
        if self.alpha is None:
            return 
        
        logger.info("merge actor's weight")
        self.actor.merge_weight()

    def merge_encoder_weight(self):
        if self.alpha is None or self.fuse_encoder is False:
            return 
        
        logger.info("merge encoder's weight")
        self.network.merge_weight()

    def log_alphas(self):
        if self.alpha is not None:
            if self.fuse_actor:
                logger.info("Actor's alphas")
                self.actor.log_alphas()
            if self.fuse_encoder:
                logger.info("Encoder's alphas")
                self.network.log_alphas()
                
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
            # logger.info(f"Alpha's shape: {self.alpha.shape}, Alpha: {self.alpha.data}, Alpha scale: {self.alpha_scale.data}")
        else:
            self.alpha = None
            self.alpha_scale = None
            
    def setup_vectors(self, base_dir, latest_dir):
        if base_dir == None:
            self.num_vectors = 0
        elif latest_dir == None:
            self.num_vectors = 1
        else:
            if self.fuse_actor:
                logger.debug("Setup actor's vectors")
                base_model = torch.load(f"{base_dir}/actor.pt", map_location=None)
                self.actor_base = base_model.get_base()
                latest_model = torch.load(f"{latest_dir}/actor.pt", map_location=None)
                self.actor_vectors, self.num_vectors = latest_model.get_vectors(self.actor_base)
                self.merge_vectors(self.actor_vectors)         
            #TODO encoder
            # if self.fuse_encoder:
            #     logger.info("Setup actor's vectors")
            #     latest_model = torch.load(f"{latest_dir}/encoder.pt", map_location=None)
            #     self.encoder_vectors = latest_model.get_vectors()   
        
    def merge_vectors(self, vectors):
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
            self.num_vectors = self.pool_size
            for i_vectors in vectors:
                merge(i_vectors)
        