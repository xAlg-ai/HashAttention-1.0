import torch
import cupy as cp

bit_count_kernel_long = cp.RawKernel(r'''
extern "C" __global__
void bit_count_kernel_long(const unsigned long long int* input, unsigned long long int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __popcll(input[idx]);
    }
}
''', 'bit_count_kernel_long')


def gpu_bit_count_long(input_tensor, output_tensor):
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    if input_tensor.dtype != torch.int64:
        raise ValueError("Input tensor must be of type torch.int64 (long)")
    
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.cuda()
    
    input_cp = cp.from_dlpack(input_tensor)
    output_cp = cp.from_dlpack(output_tensor)
    
    total_elements = input_cp.size
    threads_per_block = 256
    blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
    
    bit_count_kernel_long((blocks_per_grid,), (threads_per_block,),
                          (input_cp, output_cp, total_elements))


def test_gpu_bit_count_long():
    torch.manual_seed(42)
    
    H, B, S = 10, 20, 30
    input_torch = torch.randint(0, 2**63-1, (H, B, S), dtype=torch.int64, device='cuda')
    result_gpu = torch.zeros((H, B, S), dtype=torch.int64, device='cuda')
    
    gpu_bit_count_long(input_torch, result_gpu)
    
    input_cpu = input_torch.cpu()
    result_cpu = torch.tensor([[bin(x.item()).count('1') for x in row] for row in input_cpu.reshape(-1, S)], 
                              dtype=torch.int64).reshape(H, B, S)
    
    result_gpu_cpu = result_gpu.cpu()
    
    assert torch.all(result_gpu_cpu == result_cpu), "GPU and CPU results do not match!"
    print("GPU and CPU results match!")
    
    print("\nSample results:")
    print("Input:", input_torch[0, 0, :5].cpu().numpy())
    print("Input binary:", [bin(x) for x in input_torch[0, 0, :5].cpu().numpy()])
    print("GPU output:", result_gpu[0, 0, :5].cpu().numpy())
    print("CPU output:", result_cpu[0, 0, :5].numpy())

if __name__ == "__main__":
    test_gpu_bit_count_long()
