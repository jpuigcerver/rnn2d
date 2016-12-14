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

## Available layers
- [LSTM 2D](https://github.com/jpuigcerver/rnn2d/wiki/LSTM-2D)
- [Tile 2D](https://github.com/jpuigcerver/rnn2d/wiki/Tile-2D)

## Requirements

- GNU C++11 compiler (once the library is compiled, you can use it from C, C++03, etc)
- CMake 3.0
- Google Logging (Glog)
- BLAS implementation (ATLAS, OpenBLAS, Intel MKL, etc)
- If you want the GPU implementation:
  - CUDA toolkit
  - cuBLAS 2 (included with CUDA toolkit >= 6.0)
  - Thurst (included with CUDA toolkit >= 6.0)

It's also recommended (but not required) to have the following packages:

- OpenMP, for faster CPU implementations.
- Google Perftools, for faster memory allocation in the CPU.
- Google Test and Google Mock, for testing.
- Google Benchmark, for benchmarking.

## Install

If you are going to use this library from Torch, I recommend to install it using the provided rock:

```bash
$ luarocks install https://raw.githubusercontent.com/jpuigcerver/rnn2d/master/torch/rnn2d-scm-1.rockspec
```
If you want to do a more costumized install, clone the repository and `cd` into it. Then, you'll just
need to use cmake to compile and install the library as with any CMake install.


```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DBLAS_VENDORS=ATLAS;GENERIC -DWITH_CUDA=ON -DWITH_TORCH=ON
$ make -j8
$ make install
```

`BLAS_VENDORS` is a semicolon-separated list containing different BLAS implementations to search
for. In this example, it will first try to use the ATLAS implementation (recommended) if available
and, otherwise, it will use the generic BLAS implementation.

`WITH_CUDA` indicates that the CUDA implementation of the layers should be compiled and installed.
By default this is ON. Of course, if CMake does not find the CUDA toolkit, it will ignore this flag.
You can use the variable `CUDA_TOOLKIT_ROOT_DIR` to help CMake find your CUDA installation.

`WITH_TORCH` indicates that the Torch bindings for the layers should also be compiled and installed.
By default this is ON. Again, if CMake does not find a Torch installation in your PATH, it will
ignore this flag. You can use the variable `TORCH_ROOT` to help CMake find the Torch installation.

There are other variables that CMake supports to help it find other required or recommended
packages. If CMake can't find a dependency, take a look at the `cmake/Find*.cmake` files.
