# rnn2d

The purpose of this library is to have a open source implementation of the 
most common 2D Recurrent Neural Network (RNN) layers, for both CPUs and GPUs.

2D RNNs are widely used in many applications manipulating 2D objects, like
images. For instance, 2D-LSTMs have become the state-of-the-art in Handwritten
Text Recognition, and, yet, it is very hard to find an open source CPU
implementation which is well optimized and parallelized, and it is even more
difficult to find a GPU implementation.

I am also including bindings for Torch, since it is the Deep Learning framework
that I am currently using.

## Principles

1. Open source: MIT License.
2. CPU and GPU: BLAS and CUDA.
3. Efficiency: both memory and speed, controlling the tradeoff if possible.
4. Portability: you should be able to easily use the library in your favorite
   Deep Learning frameworks (i.e. Tensorflow, Theano, Torch, etc).

## Available layers:
- [LSTM-2D](https://github.com/jpuigcerver/rnn2d/wiki/LSTM-2D)

## Requirements:

- GNU C++11 compiler (once the library is compiled, you can use it from C, C++03, etc)
- CMake 3.0
- Google Logging (Glog)
- BLAS implementation (ATLAS, OpenBLAS, Intel MKL, etc)
- If you want the GPU implementation:
  - CUDA 
  - cuBLAS

It's also recommended (but not required) to have the following packages:

- OpenMP, for faster CPU implementations.
- Google Perftools, for faster memory allocation in the CPU.
- Google Test and Google Mock, for testing.
- Google Benchmark, for benchmarking.
