import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from math import sqrt

__all__ = ['convert_kvcache_mistral_heavy_recent', 'MistralAttention_heavy_hitter']


import torch
from torch import nn
import math
import numpy as np
from einops import rearrange

import torch

def memory_efficient_softmax(x, dim):
    # Subtract max for numerical stability (log-sum-exp trick)
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    
    # Compute softmax
    softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return softmax_x



#### USA ####
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
        
    @staticmethod
    def backward(ctx, grad_output):
        #straight through estimator is unstable
        #return grad_output
        input, = ctx.saved_tensors
        # try tanh derivative
        return (1 - torch.square(torch.tanh(input))) * grad_output

def ste_sign(input):
    return SignSTE.apply(input)

DEFAULT_USA_CFG = {
    'lth_int_dim' : 128,
    'lth_final_dim': 32,
    'lth_thold' : 0
}

class USA(nn.Module):
    def __init__(self, num_heads, head_dim, usa_params = DEFAULT_USA_CFG):
        super(USA, self).__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.int_dim = usa_params['lth_int_dim']
        self.lth_final_dim = usa_params['lth_final_dim']
        self.lth_thold = usa_params['lth_thold']
        self.learning_to_hash_transformation_k = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, self.int_dim), 
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.lth_final_dim)
                                    ) for i in range(self.num_heads)])
        self.learning_to_hash_transformation_q = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.lth_final_dim)
                                      ) for i in range(self.num_heads)])
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, K, Q, hard=False):

        b,a,sk,d = K.shape
        _,_,sq,d = Q.shape

        Klifted = torch.zeros((b,a,sk,self.lth_final_dim), device=K.device, dtype=K.dtype)
        Qlifted = torch.zeros((b,a,sq,self.lth_final_dim), device=Q.device, dtype=Q.dtype)
        
        for i in range(self.num_heads):
            Klifted[:,i,:,:] = self.learning_to_hash_transformation_k[i](K[:,i,:,:])
            Qlifted[:,i,:,:] = self.learning_to_hash_transformation_q[i](Q[:,i,:,:])

        if hard:
            Q = ste_sign(Qlifted)
            K = ste_sign(Klifted)
        else:
            Q = torch.tanh(Qlifted)
            K = torch.tanh(Klifted)

        bsz, _, q_seq_len, _ = Q.size()
        _, _, k_seq_len, _ = K.size()
        q = rearrange(Q, 'b h t d -> (b h) t d')
        k = rearrange(K, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        span_scores = torch.empty(bsz * self.num_heads, q_seq_len, k_seq_len, dtype=Q.dtype,
                                   device=Q.device)

        span_scores = rearrange(torch.baddbmm(span_scores, q, k, beta=0, alpha=1.0),
                                 '(b h) t s -> b h t s', h=self.num_heads)



        query_length, key_length = Q.size(-2), K.size(-2)
        causal_mask = torch.tril(torch.ones(query_length,key_length,device=K.device), diagonal=key_length - query_length).bool()
        mask_value = torch.finfo(span_scores.dtype).min
        mask_value = torch.full([], mask_value, dtype=span_scores.dtype, device=span_scores.device)
        span_scores = torch.where(causal_mask, span_scores.to(span_scores.dtype), mask_value)
        # mask 
        if not hard:
            span_scores = nn.functional.sigmoid(span_scores - self.lth_thold)

        return span_scores, K
    

    
    def k_embedding(self, vectors, hard=False):
        b,a,sk,d = vectors.shape
        embeddings = torch.zeros((b,a,sk,self.lth_final_dim), device=vectors.device)
        for i in range(self.num_heads): # convert to bmm
            embeddings[:,i,:,:] = self.learning_to_hash_transformation_k[i](vectors[:,i,:,:])
            
        if hard:
            embeddings = ste_sign(embeddings)
        else:
            embeddings = torch.tanh(embeddings)
        return embeddings

    def q_embedding(self, vectors, hard=False):
        b,a,sk,d = vectors.shape
        embeddings = torch.zeros((b,a,sk,self.lth_final_dim), device=vectors.device)
        for i in range(self.num_heads): # convert to bmm
            embeddings[:,i,:,:] = self.learning_to_hash_transformation_q[i](vectors[:,i,:,:])
            
        if hard:
            embeddings = ste_sign(embeddings)
        else:
            embeddings = torch.tanh(embeddings)
        return embeddings


class MistralAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.init_budget = config.init_budget
        self.heavy_budget = config.heavy_budget
        self.recent_budget = config.recent_budget
        self.cache_budget_records = [] # determined by USA
        self.usa_module = None
        self.usa_retrieve_depth = config.usa_retrieve_depth
        self.usa_eval_mode = config.usa_eval_mode
        self.usa_module_dtype = None

        # stateful ... needs to be reset every new example
        self.past_key_signatures = None

        # memory usage reducer
        self.use_softmax_maxtrick = False

        # stats
        self.collect_stats = False
        self.overlaps = {}
        self.recalls = {}
        self.precision = {}
        self.print_offloading_flag = False
        self.offloading_length = 25000


        #train usa
        self.train_usa = False
        self.tr_loss_func = None
        self.tr_optimizer = None

    def __repr__(self):
        return f"{super().__repr__()}\nSparsification Setting(topk:{self.heavy_budget}  edge:{self.init_budget, self.recent_budget} \n eval_mode:{self.usa_eval_mode} stats:{self.collect_stats}\n Istraining:{self.train_usa} )"


    def _reset_state(self):
        self.past_key_signatures = None
        #self.overlaps = {}
        self.print_offloading_flag = False


    @torch.inference_mode(False)
    def train_step(self, K, Q, target, target_mask):
        K = K.clone() # need a separate tensor for backward
        Q = Q.clone()
        target = target.clone()
        span, _ = self.usa_module(K, Q) # 0,1
        span = span.masked_fill(torch.logical_not(target_mask), 0) # remove all the fringe elements with the same mask applied to target
        loss = self.tr_loss_func(span, target)
        self.tr_optimizer.zero_grad()
        loss.backward()
        self.tr_optimizer.step()
        if self.layer_idx == 5:
            print(self.layer_idx, K.shape[2], "Train Loss:", loss.item(), flush=True)
    

    def usa_local_compute(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None        ):
        ''' independent computation from forward pass to enable logging even when we do full attention
             Expects that the KV Cache is already on the GPU and that handling is outside the function
        '''

        if (past_key_value is None or 
            len(past_key_value.key_cache) <= self.layer_idx or
            past_key_value.key_cache[self.layer_idx].shape[-2]  % 1024 != 0):
            return 

        # prepare keys and queries.
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
           
            
        if position_embeddings is None:
            cos, sin = self.rotary_emb(query_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = past_key_value.key_cache[self.layer_idx] # keys already appended in cache
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        kv_seq_len = key_states.shape[2]


        # target
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 
        # causal_mask + recent budget + init budget
        causal_heavy_recent_mask = torch.tril(torch.ones(q_len,kv_seq_len,device=attn_weights.device), diagonal=kv_seq_len-q_len-self.recent_budget).bool()
        causal_heavy_recent_mask[:,:self.init_budget] = False
        
        attn_weights.masked_fill_(torch.logical_not(causal_heavy_recent_mask), torch.finfo(attn_weights.dtype).min)
        target = torch.zeros_like(attn_weights)
        _,idx = torch.topk(attn_weights, dim=-1, k=32) # [B,A,S,T]
        view_idx = idx.view(-1,idx.shape[-1])
        view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
        target.view(-1)[view_idx.view(-1)] = 1.0

        # predicted # dont want it to be conditioned on retrieval algorithm .. since I just want to measure the quality
        if self.train_usa:
            self.train_step(key_states, query_states, target, causal_heavy_recent_mask)
        else:
            span, _ = self.usa_module(key_states.to(self.usa_module_dtype), query_states.to(self.usa_module_dtype), hard=True)
            span.masked_fill_(torch.logical_not(causal_heavy_recent_mask),torch.finfo(span.dtype).min)
            pred = torch.zeros_like(span)
            _,idx = torch.topk(span, dim=-1, k=kv_seq_len // 16) # [B,A,S,T]
            view_idx = idx.view(-1,idx.shape[-1])
            view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * span.shape[-1]
            pred.view(-1)[view_idx.view(-1)] = 1.0

            # stats.
            overlap = pred * target
            overlap_ratio = torch.sum(overlap, dim=-1) / torch.sum(target, dim=-1)
            
            ## add to collection
            if kv_seq_len not in self.overlaps.keys():
                self.overlaps[kv_seq_len] = [0,0,0,0,0] # sum, sqsum, ct, mean, std
                
            self.overlaps[kv_seq_len][0] += overlap_ratio.sum().item()
            self.overlaps[kv_seq_len][1] += torch.square(overlap_ratio).sum().item()
            self.overlaps[kv_seq_len][2] += overlap_ratio.numel()
            self.overlaps[kv_seq_len][3] = self.overlaps[kv_seq_len][0] / self.overlaps[kv_seq_len][2]
            self.overlaps[kv_seq_len][4] = sqrt(self.overlaps[kv_seq_len][1] / self.overlaps[kv_seq_len][2] - self.overlaps[kv_seq_len][3]**2)

            if self.layer_idx == 5:
                print(self.overlaps) 


    def compute_mask_multi(self, key_states, query_states):
        bsz = query_states.shape[0]
        q = query_states.shape[-2]
        k = key_states.shape[-2]
        if self.past_key_signatures is None:
            span, K = self.usa_module(key_states.to(self.usa_module_dtype), query_states.to(self.usa_module_dtype), hard=True)
            #self.past_key_signatures = K # TODO(test)
            self.past_key_signatures = None
        else:
            # TODO(test)
            current_q_embedding = self.usa_module.q_embedding(query_states.to(self.usa_module_dtype), hard=True)
            current_k_embeddings = self.usa_module.k_embedding(key_states[:,:,self.past_key_signatures.shape[-2]:,:].to(self.usa_module_dtype), hard=True)
            total_k_embeddings = torch.cat([self.past_key_signatures, current_k_embeddings], dim=-2)
            self.past_key_signatures = total_k_embeddings
            current_q_embedding = rearrange(current_q_embedding, 'b h t d -> (b h) t d')
            total_k_embeddings = rearrange(total_k_embeddings, 'b h s d -> (b h) d s')
            span = torch.empty(bsz * self.num_heads, q, k, dtype=current_q_embedding.dtype,
                                   device=current_q_embedding.device)
            span = rearrange(torch.baddbmm(span, current_q_embedding, total_k_embeddings, beta=0, alpha=1.0),
                                 '(b h) t s -> b h t s', h=self.num_heads)


        causal_heavy_recent_mask = torch.tril(torch.ones(q,k,device=key_states.device), diagonal=k-q-self.recent_budget).bool()
        causal_heavy_recent_mask[:,:self.init_budget] = False
        mask = torch.zeros_like(span).bool() # True is keep and False is throw away
        span.masked_fill_( torch.logical_not(causal_heavy_recent_mask ),torch.finfo(span.dtype).min)
        mask[:,:,:,:] = torch.tril(torch.ones(q,k,device=key_states.device), diagonal=k-q).bool()
        mask = mask * torch.logical_not(causal_heavy_recent_mask)

        values, indices = span.sort(dim=-1, descending=True)
        if self.heavy_budget > 1.0:
            heavy_budget = int(self.heavy_budget)
        else:
            heavy_budget = int(k * self.heavy_budget)
        heavy_budget = heavy_budget
        if self.usa_eval_mode == 'simple':
            keep_indices = indices[:,:,:,:heavy_budget]
            mask.scatter_(3, keep_indices, True)
        else:
            depth_thold = values[:,:,:,:1] - 2*self.usa_retrieve_depth # every mismatch addds a diff of 2
            num_thold = values[:,:,:,heavy_budget-1:heavy_budget]
            thold = torch.maximum(depth_thold, num_thold)
            mask.masked_fill_(span >= thold, True)
        # self.cache_budget_records.append(torch.mean(torch.sum(mask.float(), dim=-1)).mean().item())
        # print(self.cache_budget_records)
        return mask

    def compute_mask(self, key_states, query_states):
        bsz = query_states.shape[0]
        q = query_states.shape[-2]
        k = key_states.shape[-2]
        assert q == 1

        if self.past_key_signatures is None:
            span, K = self.usa_module(key_states.to(self.usa_module_dtype), query_states.to(self.usa_module_dtype), hard=True)
            #self.past_key_signatures = K # TODO(test)
            self.past_key_signatures = None
        else:
            # TODO(test)
            current_q_embedding = self.usa_module.q_embedding(query_states.to(self.usa_module_dtype), hard=True)
            current_k_embeddings = self.usa_module.k_embedding(key_states[:,:,self.past_key_signatures.shape[-2]:,:].to(self.usa_module_dtype), hard=True)
            total_k_embeddings = torch.cat([self.past_key_signatures, current_k_embeddings], dim=-2)
            self.past_key_signatures = total_k_embeddings
            current_q_embedding = rearrange(current_q_embedding, 'b h t d -> (b h) t d')
            total_k_embeddings = rearrange(total_k_embeddings, 'b h s d -> (b h) d s')
            span = torch.empty(bsz * self.num_heads, q, k, dtype=current_q_embedding.dtype,
                                   device=current_q_embedding.device)
            span = rearrange(torch.baddbmm(span, current_q_embedding, total_k_embeddings, beta=0, alpha=1.0),
                                 '(b h) t s -> b h t s', h=self.num_heads)


        mask = torch.zeros_like(span).bool()
        mask[:,:,:,:self.init_budget] = True
        mask[:,:,:,-self.recent_budget:] = True

        span.masked_fill_(mask, torch.finfo(span.dtype).min)
        values, indices = span.sort(dim=-1, descending=True)
        if self.heavy_budget > 1.0:
            heavy_budget = int(self.heavy_budget)
        else:
            heavy_budget = int(k * self.heavy_budget)
        heavy_budget = heavy_budget
        if self.usa_eval_mode == 'simple':
            keep_indices = indices[:,:,:,:heavy_budget]
            mask.scatter_(3, keep_indices, True)
        else:
            depth_thold = values[:,:,:,:1] - 2*self.usa_retrieve_depth # every mismatch addds a diff of 2
            num_thold = values[:,:,:,heavy_budget-1:heavy_budget]
            thold = torch.maximum(depth_thold, num_thold)
            mask.masked_fill_(span >= thold, True)
        # self.cache_budget_records.append(torch.mean(torch.sum(mask.float(), dim=-1)).mean().item())
        # print(self.cache_budget_records)
        return mask
        
    def ensure_gpu(self, past_key_value, device):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx 
            and (not past_key_value.key_cache[self.layer_idx].is_cuda)):
            #print("onboarding layer", self.layer_idx)
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].to(device)
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].to(device)

    def offload_if_necessary_cpu(self, past_key_value):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx  
            and past_key_value.key_cache[self.layer_idx].shape[2] >=self.offloading_length):
            if self.print_offloading_flag == False and self.layer_idx == 0:
                print("OFFLOADING ENABLED >>")
                self.print_offloading_flag = True
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].cpu()
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].cpu()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        self.ensure_gpu(past_key_value, hidden_states.device)

        if q_len > 128 or self.train_usa:
            return_value =  self.flash_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                #position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.collect_stats or self.train_usa:
                self.usa_local_compute(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    position_embeddings
                    )
            self.offload_if_necessary_cpu(past_key_value)
            return return_value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
           
            
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        kv_seq_len = key_states.shape[-2]


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        elif q_len == kv_seq_len:
            boolean_mask = torch.tril(torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=attn_weights.device))
            attention_mask = torch.zeros(q_len, kv_seq_len, dtype=torch.float16, device=attn_weights.device)
            attention_mask = attention_mask.masked_fill(boolean_mask == False, torch.finfo(attn_weights.dtype).min).view(1, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        sparse_mask = self.compute_mask_multi(key_states, query_states) # True = keep and False = throw away
        attn_weights.masked_fill_(torch.logical_not(sparse_mask), torch.finfo(attn_weights.dtype).min)

        
        # upcast attention to fp32
        if self.use_softmax_maxtrick:
            attn_weights = memory_efficient_softmax(attn_weights, dim=-1) # avoids upcast
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.offload_if_necessary_cpu(past_key_value)
        return attn_output, attn_weights, past_key_value

def load_usa(config, path):
    CFG = {
    'lth_int_dim' : config.lth_init_dim,
    'lth_final_dim': config.lth_final_dim,
    'lth_thold' : config.lth_thold
    }
    modules = []
    for i in range(config.num_hidden_layers):
        modules.append(USA(config.num_attention_heads, config.head_dim, CFG))
    USA_MOD = nn.ModuleList(modules)
    if path is not None:
        USA_MOD.load_state_dict(torch.load(path))
    return USA_MOD.cuda()



def convert_usa(model, config, usa_modules, collect_stats, train_usa):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_usa(module, config, usa_modules, collect_stats, train_usa)

        if isinstance(module, MistralAttention):
            device = next(module.parameters()).device
            new_module = MistralAttention_heavy_hitter(config, module.layer_idx).bfloat16().to(device)
            new_module.load_state_dict(module.state_dict())
            new_module.usa_module = usa_modules[module.layer_idx]
            new_module.usa_module_dtype = usa_modules[module.layer_idx].learning_to_hash_transformation_k[0][0].weight.dtype
            new_module.rotary_emb = module.rotary_emb # Somehow rotary emb iniitalized does not match the rotary emb orignal. why? Needs to be looked at. But dont have time. this fixes it
            model._modules[name] = new_module
            model._modules[name].flash_forward = module.forward
            model._modules[name].collect_stats = collect_stats
            model._modules[name].train_usa = train_usa
            
    return model


def reset_usa(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = reset_usa(module)

        if isinstance(module, MistralAttention_heavy_hitter):
            module._reset_state()

    return model


def set_train_usa_mode(model, loss_func=None, optimizer=None):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = set_train_usa_mode(module, loss_func, optimizer)

        if isinstance(module, MistralAttention_heavy_hitter):
            module.train_usa = True
            if loss_func is not None:
                module.tr_loss_func = loss_func
            if optimizer is not None:
                module.tr_optimizer = optimizer

    return model


def set_eval_usa_mode(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = set_eval_usa_mode(module)

        if isinstance(module, MistralAttention_heavy_hitter):
            module.train_usa = False

    return model


def print_stats(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = print_stats(module)

        if isinstance(module, MistralAttention_heavy_hitter):
            if module.layer_idx == 5:
                print(module.layer_idx)
                print(module.overlaps)

    return model
