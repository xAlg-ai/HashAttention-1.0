import torch
import sys

import triton
import triton.language as tl
import math

from sparse import fwd_sparse, torch_fwd_sparse
from flash_fwd_sparse import flash_fwd_sparse
# from transformers.modeling_flash_attention_utils import _flash_attention_forward
from bit_count_long import gpu_bit_count_long
import torch.nn as nn
from torch.nn.functional import silu,relu
from torch import matmul, sign


def torch_attention_naive(query_states, key_states, value_states, out, attention_mask):
    '''
        query_states: b,a,q,d
        key_states: b,a,k,d
        value_stats: b,a,k,d
        attention_mask: (1,1,q,k)
    '''
    head_dim = query_states.shape[-1]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output

def usa_attention(query_states, key_states, value_states, out, attention_mask, usa_module, usa_biases, key_label, query_label_states, topk, power, match, count):
    # assert q == 1
    #q.shape = a,1,d
    query_label = relu(sign(matmul(silu(matmul(silu(matmul(query_label_states, usa_module[0]) + usa_biases[0]), usa_module[1]) + usa_biases[1]), usa_module[2]) + usa_biases[2])).long() # H, B, D
    query_label = torch.sum(query_label * power, dim=-1).long() # H,B,32 #TODO use 32 bits uint32 in final cuda kernel 
    #key_label : b,a,k,1
    match = torch.bitwise_not(torch.bitwise_xor(query_label.unsqueeze(-1), key_label), out=match) # # b,a,k,1

    # count = match
    # count = torch.empty_like(match)
    gpu_bit_count_long(match, count)

    # TODO run additional bitcount on match. Can be done in a fused kernel implementaiton later using _popc/_popcll  https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html

    _, indices = torch.topk(count, topk, dim=-1)
    indices = indices.transpose(0,1).contiguous()
    #print(query_states.shape, key_states.shape, value_states.shape, out.shape, indices.shape, attention_mask.shape)
    fwd_sparse(query_states, key_states, value_states, out, indices, attention_mask)
    return out

def usa_attention_flash(query_states, key_states, value_states, out, attention_mask, usa_module, usa_biases, key_label, query_label_states, topk, power, match, count, sm_scale=None, req_to_token=None, mid_out=None, mid_o_logexpsum=None, BLOCK_SEQ=None):
    # assert q == 1
    #q.shape = a,1,d
    query_label = relu(sign(matmul(silu(matmul(silu(matmul(query_label_states, usa_module[0]) + usa_biases[0]), usa_module[1]) + usa_biases[1]), usa_module[2]) + usa_biases[2])).long() # H, B, D
    query_label = torch.sum(query_label * power, dim=-1).long() # H,B,32 #TODO use 32 bits uint32 in final cuda kernel 
    #key_label : b,a,k,1
    match = torch.bitwise_not(torch.bitwise_xor(query_label.unsqueeze(-1), key_label), out=match) # # b,a,k,1

    # count = match
    # count = torch.empty_like(match)
    gpu_bit_count_long(match, count)

    # TODO run additional bitcount on match. Can be done in a fused kernel implementaiton later using _popc/_popcll  https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html

    _, indices = torch.topk(count, topk, dim=-1)
    indices = indices.transpose(0,1).contiguous()
    #print(query_states.shape, key_states.shape, value_states.shape, out.shape, indices.shape, attention_mask.shape)
    # fwd_sparse(query_states, key_states, value_states, out, indices, attention_mask)
    flash_fwd_sparse(query_states, key_states, value_states, out, indices, sm_scale=sm_scale, req_to_token=req_to_token, heavy_token_num=topk, mid_out=mid_out, mid_o_logexpsum=mid_o_logexpsum, BLOCK_SEQ=BLOCK_SEQ)
    return out
 

def usa_attention_profile(query_states, key_states, value_states, out, attention_mask, usa_module, usa_biases, key_label, query_label_states, topk, power):
    # assert q == 1
    #q.shape = a,1,d
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t3 = torch.cuda.Event(enable_timing=True)
    t4 = torch.cuda.Event(enable_timing=True)

    def part1():
        query_label = relu(sign(matmul(silu(matmul(silu(matmul(query_label_states, usa_module[0]) + usa_biases[0]), usa_module[1]) + usa_biases[1]), usa_module[2]) + usa_biases[2])).long() # H, B, D
        query_label = torch.sum(query_label * power, dim=-1).long() # H,B,32
        #key_label : b,a,k,1
        match = torch.bitwise_xor(query_label.unsqueeze(-1), key_label) # # b,a,k,1
        # additional bitcount needed TODO
        _, indices = torch.topk(match, topk, dim=-1)
        indices = indices.transpose(0,1).contiguous()
        return indices
    part1 = torch.compile(part1, fullgraph=True)
    torch.cuda.synchronize()
    #warmup
    for i in range(10):
        part1()
    torch.cuda.synchronize()
    t1.record()
    for i in range(1000):
        part1()
    torch.cuda.synchronize()
    t2.record()
    indices = part1()
    for i in range(10):
        fwd_sparse(query_states, key_states, value_states, out, indices, attention_mask)

    torch.cuda.synchronize()
    t3.record()
    for i in range(1000):
        fwd_sparse(query_states, key_states, value_states, out, indices, attention_mask)
    t4.record()
    torch.cuda.synchronize()
    print("Time in parts. query_label", t1.elapsed_time(t2) / 1000, "fwd_sparse:", t3.elapsed_time(t4) / 1000)
    return out

     
torch_attention_naive = torch.compile(torch_attention_naive, fullgraph=True)
# usa_attention = torch.compile(usa_attention, fullgraph=True)
def test_torch_att(B, N_CTX, H, D):
    import time
    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}")
    dtype = torch.float16

    q = torch.empty((B, H, 1, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B, H, N_CTX, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B, H, N_CTX, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)

    out = torch.empty((B, H, 1, D), dtype=dtype, device="cuda")
    attn_mask = torch.zeros((1,1,1,N_CTX), dtype=dtype, device="cuda")
    # Warm up
    for _ in range(10):  # not sure if reading hte same arrays will give us read times from global memory 
        torch_attention_naive(q, k, v, out, attn_mask)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        torch_attention_naive(q, k, v, out, attn_mask)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))
    return (t2 - t1) / run_iter


def test_usa_att(B, N_CTX, H, D, HEAVY_CONST):
    import time
    run_iter = 1000
    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CONST: {HEAVY_CONST}")
    dtype = torch.float16
    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    usa_module = [torch.empty(H,D,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,D,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,D,32, dtype=dtype).normal_(mean=0, std=0.1).cuda()]
    usa_biases = [torch.empty(H,1,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,1,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,1,32, dtype=dtype).normal_(mean=0, std=0.1).cuda()]
    power = 2**torch.arange(32).long().reshape(1,1,32).cuda()
    torch_match = torch.empty((H, B, N_CTX), dtype=torch.int64, device='cuda')
    torch_count = torch.empty((H, B, N_CTX), dtype=torch.int64, device='cuda')

    # Use flash sparse forward kernel from sglang, so there is page attention implementation
    BLOCK_SEQ = 256
    req_to_tokens = torch.arange(B * N_CTX, dtype=torch.int64, device='cuda').reshape(B, N_CTX)
    block_seq_num = (HEAVY_CONST + BLOCK_SEQ - 1) // BLOCK_SEQ

    mid_out = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num, q.shape[-1]],
            dtype=torch.float32,
            device=q.device,
    )
    mid_o_logexpsum = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num],
            dtype=torch.float32,
            device=q.device,
    )
    sm_scale = 1.0 / (D ** 0.5)

    #key_labels = torch.randint(2**32, (H,B, N_CTX), device='cuda', dtype=torch.int64) 
    # generated via MLP transformation given below 
    # keys and queries use separate usa_modules . But for timing using hte same
    key_labels = relu(sign(matmul(silu(matmul(silu(matmul(k.transpose(0,1), usa_module[0]) + usa_biases[0]), usa_module[1]) + usa_biases[1]), usa_module[2]) + usa_biases[2])).long() # H, B, D
    key_labels = torch.sum(key_labels * power, dim=-1).long().reshape(H, B, N_CTX).contiguous() # H,B,32 #TODO use 32 bits uint32 in final cuda kernel 
    query_label_states = q.reshape(B, H, D).transpose(0,1) # H, B, D

    attention_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")
    # Warm up
    for i in range(10):
        # usa_attention(q, k, v, out, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count)
        usa_attention_flash(q, k, v, out, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count, sm_scale=sm_scale, req_to_token=req_to_tokens, mid_out=mid_out, mid_o_logexpsum=mid_o_logexpsum, BLOCK_SEQ=BLOCK_SEQ)

    
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(0, run_iter):
        # usa_attention(q, k, v, out, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count)
        usa_attention_flash(q, k, v, out, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count, sm_scale=sm_scale, req_to_token=req_to_tokens, mid_out=mid_out, mid_o_logexpsum=mid_o_logexpsum, BLOCK_SEQ=BLOCK_SEQ)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))


    #for i in range(20):
    #    usa_attention_profile(q, k, v, out, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power)
    return (t2 - t1) / run_iter

def test_flash_correctness(B, N_CTX, H, D, HEAVY_CONST):
    import time
    run_iter = 1000
    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CONST: {HEAVY_CONST}")
    dtype = torch.float16
    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    out_normal = torch.empty((B, H, D), dtype=dtype, device="cuda")
    out_flash = torch.empty((B, H, D), dtype=dtype, device="cuda")

    usa_module = [torch.empty(H,D,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,D,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,D,32, dtype=dtype).normal_(mean=0, std=0.1).cuda()]
    usa_biases = [torch.empty(H,1,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,1,D, dtype=dtype).normal_(mean=0, std=0.1).cuda(),
                  torch.empty(H,1,32, dtype=dtype).normal_(mean=0, std=0.1).cuda()]
    power = 2**torch.arange(32).long().reshape(1,1,32).cuda()
    torch_match = torch.empty((H, B, N_CTX), dtype=torch.int64, device='cuda')
    torch_count = torch.empty((H, B, N_CTX), dtype=torch.int64, device='cuda')

    # Use flash sparse forward kernel from sglang, so there is page attention implementation
    BLOCK_SEQ = 256
    req_to_tokens = torch.arange(B * N_CTX, dtype=torch.int64, device='cuda').reshape(B, N_CTX)
    block_seq_num = (HEAVY_CONST + BLOCK_SEQ - 1) // BLOCK_SEQ

    mid_out = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num, q.shape[-1]],
            dtype=torch.float32,
            device=q.device,
    )
    mid_o_logexpsum = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num],
            dtype=torch.float32,
            device=q.device,
    )
    sm_scale = 1.0 / (D ** 0.5)

    #key_labels = torch.randint(2**32, (H,B, N_CTX), device='cuda', dtype=torch.int64) 
    # generated via MLP transformation given below 
    # keys and queries use separate usa_modules . But for timing using hte same
    key_labels = relu(sign(matmul(silu(matmul(silu(matmul(k.transpose(0,1), usa_module[0]) + usa_biases[0]), usa_module[1]) + usa_biases[1]), usa_module[2]) + usa_biases[2])).long() # H, B, D
    key_labels = torch.sum(key_labels * power, dim=-1).long().reshape(H, B, N_CTX).contiguous() # H,B,32 #TODO use 32 bits uint32 in final cuda kernel 
    query_label_states = q.reshape(B, H, D).transpose(0,1) # H, B, D

    attention_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")
    # Warm up
    usa_attention(q, k, v, out_normal, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count)

    usa_attention_flash(q, k, v, out_flash, attention_mask, usa_module, usa_biases, key_labels, query_label_states, HEAVY_CONST, power, torch_match, torch_count, sm_scale=sm_scale, req_to_token=req_to_tokens, mid_out=mid_out, mid_o_logexpsum=mid_o_logexpsum, BLOCK_SEQ=BLOCK_SEQ)

    print("normal:", out_normal[0,:5,:5])
    print("flash:", out_flash[0,:5,:5])

    assert torch.allclose(out_normal, out_flash, atol=1e-2, rtol=0)


if __name__ == '__main__':

    # bszs = [1, 4, 8, 16, 32]
    # ctxs = [2048, 4096, 8192, 16384]

    bsz = int(sys.argv[1])
    ctx = int(sys.argv[2])

    sparsity_level = 32
    h = 32
    d = 128

    # times = []
    
    #print(f"bsz: {bsz}, ctx: {ctx}, time: {test_torch_att(bsz, ctx, h, d)}")
    
    # print(f"bsz: {bsz}, ctx: {ctx}, time: {test_usa_att(bsz, ctx, h, d, ctx // sparsity_level)}")

    test_flash_correctness(bsz, ctx, h, d, ctx // sparsity_level)
