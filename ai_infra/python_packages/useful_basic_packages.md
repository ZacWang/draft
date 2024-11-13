
## 常见的 NVIDIA 库

统一的命名格式：
- nvidia-xxx-cu12
```
    nvidia-nvtx-cu12, 
    nvidia-nvjitlink-cu12, 
    nvidia-nccl-cu12, 
    nvidia-curand-cu12, 
    nvidia-cufft-cu12, 
    nvidia-cuda-runtime-cu12, 
    nvidia-cuda-nvrtc-cu12, 
    nvidia-cuda-cupti-cu12, 
    nvidia-cublas-cu12, 
    nvidia-cusparse-cu12, 
    nvidia-cudnn-cu12, 
    nvidia-cusolver-cu12
```

- 运行时:
    - cuda-runtime: CUDA Runtime Library, 用于CUDA编程, CUDA程序运行的核心组件
    - nccl: NVIDIA 通信库, 用于多GPU通信
    - cuda-nvrtc: CUDA NVRTC Library, 用于CUDA运行时编译器
    - nvjitlink: NVIDIA JIT链接器库, 用于运行时编译和链接CUDA代码
- 计算库：
    - curand: CUDA Random Number Generation Library, 用于生成随机数
    - cublas: CUDA Basic Linear Algebra Subroutines Library, 用于线性代数运算
    - cusparse: CUDA Sparse Library, 用于稀疏矩阵运算
    - cusolver: CUDA Solver Library, 提供各种数值求解器功能
    - cufft: CUDA Fast Fourier Transform Library, 用于快速傅里叶变换
    - cudnn: CUDA Deep Neural Network Library, 深度学习原语的高度优化实现
- 性能分析：
    - nvtx: NVIDIA Tools Extension (NVTX) 库,用于性能分析和调试
    - cuda-cupti: CUDA Profiling Tools Interface Library, 用于性能分析和调试
