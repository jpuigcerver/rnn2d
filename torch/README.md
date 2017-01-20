# rnn2d for Torch

Thanks to the Lua FFI library you can use rnn2d from Torch.

The following binding nn modules are implemented and can be used as any other
layer in your model:

- LSTM
- Tile

## Requirements

- Torch
- nn package
- ffi package
- cutorch package (only if GPU support is required)

## Instructions

The easiest way to install rnn2d with the Torch bindings is by using the
rock in this folder.

If you cloned the repository, just go the the root directory of the project and
type in your terminal:

```sh
luarocks make torch/rnn2d-scm-1.rockspec
```

You can also directly clone the repository and install it, using
`luarocks install`:

```sh
luarocks install https://raw.githubusercontent.com/jpuigcerver/rnn2d/master/torch/rnn2d-scm-1.rockspec
```

This will place all the required files to the Torch install directory.

If you are going to use rnn2d as a separate C++/C library, you might prefer to
install it using CMake.

```sh
mkdir build && cd build
cmake -DWITH_TORCH=ON -DCMAKE_BUILT_TYPE=RELEASE
make
make install
```

If CMake is not able to find your Torch install directory, you can use the
variable TORCH_ROOT to let CMake know the location of Torch in your system.

Once you have installed the library, do not forget to update the
LD_LIBRARY_PATH environment variable appropriately, so that Lua FFI can load
the required dynamic libraries.

## Usage

```lua
require 'rnn2d'

local m = nn.Sequential()
m:add(rnn2d.Tile(2, 2))
m:add(rnn2d.LSTM(3, 10))
```