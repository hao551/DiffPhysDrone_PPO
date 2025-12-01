from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quadsim_cuda',# 模块名称，可以import quadsim_cuda

    ext_modules=[# 模块名称
        CUDAExtension('quadsim_cuda', [# CUDA扩展
            'quadsim.cpp',# C++接口文件
            'quadsim_kernel.cu',# CUDA内核文件
            'dynamics_kernel.cu',# 动力学内核文件
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension# 使用PyTorch的构建扩展
    })
