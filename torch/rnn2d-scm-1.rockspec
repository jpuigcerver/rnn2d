package = "rnn2d"
version = "scm-1"

source = {
  url = "git://github.com/jpuigcerver/rnn2d.git"
}

description = {
  summary = "CPU and GPU implementations of some 2D RNN layers",
  detailed = [[
  ]],
  homepage = "https://github.com/jpuigcerver/rnn2d",
  license = "MIT",
  maintainer = "Joan Puigcerver <joapuipe@prhlt.upv.es>"
}

dependencies = {
  "torch >= 7.0",
  "luaffi",
  "nn"
}

build = {
  type = "cmake",
  variables = {
    TORCH_ROOT = "$(LUA_BINDIR)/..",
    CMAKE_BUILD_TYPE = "RELEASE",
    CMAKE_INSTALL_PREFIX = "$(PREFIX)",
    INST_LUADIR = "$(LUADIR)",
    WITH_TESTS = "OFF",
    WITH_BENCHMARKS = "OFF",
  }
}
