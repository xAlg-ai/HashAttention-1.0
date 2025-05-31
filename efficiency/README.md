# Efficiency Evaluation

This directory provides tools and scripts for evaluating the efficiency of HashAttention and related kernels.

## Kernels

- Each file in the `kernels/` directory can be executed independently to benchmark the efficiency of specific operations.
- Some kernels are adapted from the Double Sparsity project.
- These kernels are designed for fine-grained performance analysis and comparison.

## End-to-End Evaluation

- The end-to-end evaluation pipeline is currently a work in progress (WIP).
- Once completed, it will enable comprehensive benchmarking of full model workflows.

## Usage

1. To evaluate a specific kernel, navigate to the `kernels/` directory and run the desired script independently.
2. For more details on each kernel, refer to the comments and documentation within the respective files.

