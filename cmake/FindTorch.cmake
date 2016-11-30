## Find Torch
##
## This will define the following variables:
## - TORCH_FOUND         : TRUE if a Torch installation was found.
## - TORCH_BIN_PATH      : Directory containing the Torch binary (th).
## - TORCH_LUALIB_PATH   : Directory containing dyn libraries for Lua modules.
## - TORCH_LUA_PATH      : Directory containing the Lua modules.
## - TORCH_INCLUDE_DIRS  : Include directory needed to compile with Torch.
## - TORCH_LIBRARIES     : Libraries needed to compile with Torch.
##
## You can specify the variable TORCH_ROOT to help CMake to find all the
## required files. E.g:
## cmake -DTORCH_ROOT=/opt/torch/install
##

INCLUDE(FindPackageHandleStandardArgs)

SET(TORCH_ROOT ""
  CACHE PATH "Folder containing the root Torch install directory")

IF (NOT TORCH_ROOT STREQUAL "")
  MESSAGE(STATUS "TORCH_ROOT = ${TORCH_ROOT}")
ENDIF()

FIND_PATH(TORCH_BIN_PATH
  NAMES th
  HINTS ${TORCH_ROOT}/bin
  DOC "Path where the Torch binaries are located."
)
MARK_AS_ADVANCED(TORCH_BIN_PATH)

IF (TORCH_ROOT STREQUAL "")
  GET_FILENAME_COMPONENT(TORCH_ROOT ${TORCH_BIN_PATH}/.. ABSOLUTE)
ENDIF()

FIND_PATH(TORCH_LUALIB_PATH
  NAMES libtorch.so
  HINTS ${TORCH_ROOT}/lib/lua ${TORCH_BIN_PATH}/../lib/lua
  PATH_SUFFIXES 5.1 5.2 5.3
  DOC "Path where the dynamic libraries of Lua modules are located."
)
MARK_AS_ADVANCED(TORCH_LUALIB_PATH)

FIND_PATH(TORCH_LUA_PATH
  NAMES torch/init.lua
  HINTS ${TORCH_ROOT}/share/lua ${TORCH_BIN_PATH}/../share/lua
  PATH_SUFFIXES 5.1 5.2 5.3
  DOC "Path where the Lua modules are located."
)
MARK_AS_ADVANCED(TORCH_LUA_PATH)

FIND_PATH(TORCH_INCLUDE_DIR
  NAMES TH/TH.h
  HINTS ${TORCH_ROOT}/include ${TORCH_BIN_PATH}/../include
  DOC "Path where the Torch includes are located."
)
MARK_AS_ADVANCED(TORCH_INCLUDE_DIR)

FIND_LIBRARY(TORCH_TH_LIBRARY
  NAMES TH
  HINTS ${TORCH_ROOT}/lib ${TORCH_BIN_PATH}/../lib
)
MARK_AS_ADVANCED(TORCH_TH_LIBRARY)

FIND_LIBRARY(TORCH_LUAT_LIBRARY
  NAMES luaT
  HINTS ${TORCH_ROOT}/lib ${TORCH_BIN_PATH}/../lib
)
MARK_AS_ADVANCED(TORCH_LUAT_LIBRARY)

FIND_LIBRARY(TORCH_LUAJIT_LIBRARY
  NAMES luajit
  HINTS ${TORCH_ROOT}/lib ${TORCH_BIN_PATH}/../lib
)
MARK_AS_ADVANCED(TORCH_LUAJIT_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TORCH
  DEFAULT_MSG TORCH_ROOT TORCH_BIN_PATH TORCH_LUALIB_PATH TORCH_LUA_PATH TORCH_INCLUDE_DIR
  TORCH_LUAT_LIBRARY TORCH_TH_LIBRARY)

IF (TORCH_FOUND)
  MESSAGE(STATUS "Found Torch7 installed in ${TORCH_ROOT}")
  SET(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIR})
  IF (TORCH_LUAJIT_LIBRARY)
    SET(TORCH_LIBRARIES ${TORCH_LUAJIT_LIBRARY})
  ENDIF()
  SET(TORCH_LIBRARIES ${TORCH_LIBRARIES} ${TORCH_LUAT_LIBRARY} ${TORCH_TH_LIBRARY})
ENDIF()
