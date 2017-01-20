ADD_TEST(NAME torch_lstm_test
  COMMAND "${TORCH_BIN_PATH}/th" "${CMAKE_CURRENT_SOURCE_DIR}/LSTM.lua")

# Set LUA_PATH and LD_LIBRARY_PATH to be able to run the tests.
# We need to escape semicolons (;) in LUA_PATH with \;, beause CMake uses ; to
# split elements in lists.
SET(NEW_LUA_PATH "${CMAKE_SOURCE_DIR}/torch/?/init.lua;${CMAKE_SOURCE_DIR}/torch/?.lua;$ENV{LUA_PATH}")
STRING(REPLACE ";" "\\;" NEW_LUA_PATH "${NEW_LUA_PATH}")
SET_PROPERTY(TEST torch_lstm_test APPEND PROPERTY
  ENVIRONMENT "LUA_PATH=${NEW_LUA_PATH}")
SET_PROPERTY(TEST torch_lstm_test APPEND PROPERTY
  ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}")
