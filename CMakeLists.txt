cmake_minimum_required(VERSION 3.20)

project(minimind_orangepi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MINIMIND_ENABLE_ASCEND "Build with Ascend ACL support when the toolkit is available" ON)

set(MINIMIND_ASCEND_TOOLKIT_ROOT "" CACHE PATH "Ascend CANN toolkit root")
if(NOT MINIMIND_ASCEND_TOOLKIT_ROOT AND DEFINED ENV{ASCEND_TOOLKIT_ROOT})
  set(MINIMIND_ASCEND_TOOLKIT_ROOT "$ENV{ASCEND_TOOLKIT_ROOT}")
endif()
if(NOT MINIMIND_ASCEND_TOOLKIT_ROOT AND DEFINED ENV{ASCEND_HOME_PATH})
  set(MINIMIND_ASCEND_TOOLKIT_ROOT "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT MINIMIND_ASCEND_TOOLKIT_ROOT)
  if(EXISTS "/usr/local/Ascend/cann/latest")
    set(MINIMIND_ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/cann/latest")
  elseif(EXISTS "/usr/local/Ascend/cann")
    set(MINIMIND_ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/cann")
  elseif(EXISTS "/usr/local/Ascend/cann-8.5.0")
    set(MINIMIND_ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/cann-8.5.0")
  endif()
endif()

set(MINIMIND_ASCEND_FOUND OFF)
if(MINIMIND_ENABLE_ASCEND AND MINIMIND_ASCEND_TOOLKIT_ROOT)
  find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
    HINTS
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/include"
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/aarch64-linux/include"
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/x86_64-linux/include"
    NO_DEFAULT_PATH)

  find_library(ASCEND_ACL_LIBRARY ascendcl
    HINTS
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/lib64"
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/aarch64-linux/lib64"
      "${MINIMIND_ASCEND_TOOLKIT_ROOT}/x86_64-linux/lib64"
    NO_DEFAULT_PATH)

  if(ASCEND_ACL_INCLUDE_DIR AND ASCEND_ACL_LIBRARY)
    set(MINIMIND_ASCEND_FOUND ON)
  endif()
endif()

add_library(minimind_orangepi_runtime STATIC
  src/csrc/lib/acl_context.cpp
  src/csrc/lib/ops.cpp
  src/csrc/lib/tensor.cpp
  src/csrc/lib/tokenizer.cpp
  src/csrc/lib/weights.cpp
  src/csrc/models/minimind/config.cpp
  src/csrc/models/minimind/decoder_layer.cpp
  src/csrc/models/minimind/language_model.cpp
  src/csrc/models/minimind/rope.cpp
)

target_include_directories(minimind_orangepi_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
)

target_compile_options(minimind_orangepi_runtime PRIVATE -Wall -Wextra -Wpedantic)

target_compile_definitions(minimind_orangepi_runtime
  PUBLIC
    $<$<BOOL:${MINIMIND_ASCEND_FOUND}>:MINIMIND_USE_ASCEND=1>
)

if(MINIMIND_ASCEND_FOUND)
  target_include_directories(minimind_orangepi_runtime PRIVATE ${ASCEND_ACL_INCLUDE_DIR})
  target_link_libraries(minimind_orangepi_runtime PUBLIC ${ASCEND_ACL_LIBRARY})
  message(STATUS "Ascend ACL found: ${ASCEND_ACL_LIBRARY}")
else()
  message(STATUS "Ascend ACL not found; building host-only runtime tests")
endif()

add_executable(minimind_generate src/csrc/tools/minimind_generate.cpp)
target_link_libraries(minimind_generate PRIVATE minimind_orangepi_runtime)
target_compile_options(minimind_generate PRIVATE -Wall -Wextra -Wpedantic)

function(minimind_add_unit_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE minimind_orangepi_runtime)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

enable_testing()
minimind_add_unit_test(test_smoke tests/unit/test_smoke.cpp)
minimind_add_unit_test(test_tensor tests/unit/test_tensor.cpp)
minimind_add_unit_test(test_minimind_config tests/unit/test_minimind_config.cpp)
minimind_add_unit_test(test_rope tests/unit/test_rope.cpp)
minimind_add_unit_test(test_weights tests/unit/test_weights.cpp)
minimind_add_unit_test(test_tokenizer tests/unit/test_tokenizer.cpp)
minimind_add_unit_test(test_decoder_layer tests/unit/test_decoder_layer.cpp)
minimind_add_unit_test(test_language_model tests/unit/test_language_model.cpp)
