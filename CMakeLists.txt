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
set(MINIMIND_CUSTOM_OPP_ROOT "" CACHE PATH "MiniMind custom OPP install root")
set(MINIMIND_ASCEND_TOOLKIT_CANDIDATES)
foreach(candidate
    "${MINIMIND_ASCEND_TOOLKIT_ROOT}"
    "$ENV{ASCEND_TOOLKIT_ROOT}"
    "$ENV{ASCEND_HOME_PATH}"
    "/usr/local/Ascend/cann/latest"
    "/usr/local/Ascend/cann"
    "/usr/local/Ascend/cann-8.5.0")
  if(candidate AND EXISTS "${candidate}")
    list(APPEND MINIMIND_ASCEND_TOOLKIT_CANDIDATES "${candidate}")
  endif()
endforeach()

set(MINIMIND_CUSTOM_OPP_CANDIDATES)
foreach(candidate
    "${MINIMIND_CUSTOM_OPP_ROOT}"
    "$ENV{ASCEND_CUSTOM_OPP_PATH}"
    "${CMAKE_CURRENT_SOURCE_DIR}/custom_opp_install"
    "${CMAKE_CURRENT_SOURCE_DIR}/custom_opp_install/vendors/minimind_orangepi")
  if(candidate AND EXISTS "${candidate}/op_api/include/aclnn_mini_mind_rms_norm.h")
    list(APPEND MINIMIND_CUSTOM_OPP_CANDIDATES "${candidate}")
  elseif(candidate AND EXISTS "${candidate}/vendors/minimind_orangepi/op_api/include/aclnn_mini_mind_rms_norm.h")
    list(APPEND MINIMIND_CUSTOM_OPP_CANDIDATES "${candidate}/vendors/minimind_orangepi")
  endif()
endforeach()

set(MINIMIND_ASCEND_FOUND OFF)
foreach(candidate IN LISTS MINIMIND_ASCEND_TOOLKIT_CANDIDATES)
  if(MINIMIND_ASCEND_FOUND)
    break()
  endif()
  find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
    HINTS
      "${candidate}/include"
      "${candidate}/aarch64-linux/include"
      "${candidate}/x86_64-linux/include"
    NO_DEFAULT_PATH)
  find_library(ASCEND_ACL_LIBRARY ascendcl
    HINTS
      "${candidate}/lib64"
      "${candidate}/aarch64-linux/lib64"
      "${candidate}/x86_64-linux/lib64"
    NO_DEFAULT_PATH)
  find_library(ASCEND_OPAPI_LIBRARY opapi
    HINTS
      "${candidate}/lib64"
      "${candidate}/aarch64-linux/lib64"
      "${candidate}/x86_64-linux/lib64"
    NO_DEFAULT_PATH)
  find_library(ASCEND_NNOPBASE_LIBRARY nnopbase
    HINTS
      "${candidate}/lib64"
      "${candidate}/aarch64-linux/lib64"
      "${candidate}/x86_64-linux/lib64"
    NO_DEFAULT_PATH)

  if(MINIMIND_ENABLE_ASCEND AND ASCEND_ACL_INCLUDE_DIR AND ASCEND_ACL_LIBRARY AND ASCEND_OPAPI_LIBRARY AND ASCEND_NNOPBASE_LIBRARY)
    set(MINIMIND_ASCEND_TOOLKIT_ROOT "${candidate}" CACHE PATH "Ascend CANN toolkit root" FORCE)
    set(MINIMIND_ASCEND_FOUND ON)
  endif()
endforeach()

add_library(minimind_orangepi_runtime STATIC
  src/csrc/lib/acl_context.cpp
  src/csrc/lib/ops.cpp
  src/csrc/lib/tensor.cpp
  src/csrc/lib/tokenizer.cpp
  src/csrc/lib/weights.cpp
  src/csrc/models/minimind/ascend_custom_ops.cpp
  src/csrc/models/minimind/ascend_matmul.cpp
  src/csrc/models/minimind/config.cpp
  src/csrc/models/minimind/decoder_layer.cpp
  src/csrc/models/minimind/language_model.cpp
  src/csrc/models/minimind/rope.cpp
  src/csrc/models/minimind/runtime_loader.cpp
  src/csrc/models/minimind_v/config.cpp
  src/csrc/models/minimind_v/projector.cpp
  src/csrc/models/minimind_v/vision_encoder.cpp
  src/csrc/models/minimind_v/vlm_model.cpp
  src/csrc/models/minimind_o/config.cpp
  src/csrc/models/minimind_o/thinker.cpp
  src/csrc/models/minimind_o/talker.cpp
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
  get_filename_component(ASCEND_ACL_LIBRARY_DIR "${ASCEND_ACL_LIBRARY}" DIRECTORY)
  get_filename_component(ASCEND_OPAPI_LIBRARY_DIR "${ASCEND_OPAPI_LIBRARY}" DIRECTORY)
  target_link_libraries(minimind_orangepi_runtime PUBLIC ${ASCEND_ACL_LIBRARY} ${ASCEND_OPAPI_LIBRARY} ${ASCEND_NNOPBASE_LIBRARY})
  set(MINIMIND_RUNTIME_RPATH "${ASCEND_ACL_LIBRARY_DIR};${ASCEND_OPAPI_LIBRARY_DIR}")
  if(MINIMIND_CUSTOM_OPP_CANDIDATES)
    list(GET MINIMIND_CUSTOM_OPP_CANDIDATES 0 MINIMIND_CUSTOM_OPP_ROOT)
    set(MINIMIND_CUSTOM_OPP_INCLUDE_DIR "${MINIMIND_CUSTOM_OPP_ROOT}/op_api/include")
    set(MINIMIND_CUSTOM_OPAPI_LIBRARY "${MINIMIND_CUSTOM_OPP_ROOT}/op_api/lib/libcust_opapi.so")
    target_include_directories(minimind_orangepi_runtime PRIVATE ${MINIMIND_CUSTOM_OPP_INCLUDE_DIR})
    target_link_libraries(minimind_orangepi_runtime PUBLIC ${MINIMIND_CUSTOM_OPAPI_LIBRARY})
    target_compile_definitions(minimind_orangepi_runtime PUBLIC
      MINIMIND_USE_CUSTOM_ASCEND_OPS=1
      MINIMIND_CUSTOM_OPP_ROOT="${MINIMIND_CUSTOM_OPP_ROOT}")
    list(APPEND MINIMIND_RUNTIME_RPATH "${MINIMIND_CUSTOM_OPP_ROOT}/op_api/lib")
    message(STATUS "MiniMind custom OPP found: ${MINIMIND_CUSTOM_OPP_ROOT}")
  endif()
  list(JOIN MINIMIND_RUNTIME_RPATH ";" MINIMIND_RUNTIME_RPATH_JOINED)
  set(CMAKE_BUILD_RPATH "${MINIMIND_RUNTIME_RPATH_JOINED}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
  message(STATUS "Ascend ACL found: ${ASCEND_ACL_LIBRARY}")
  message(STATUS "Ascend opapi found: ${ASCEND_OPAPI_LIBRARY}")
else()
  message(STATUS "Ascend ACL not found; building host-only runtime tests")
endif()

add_executable(minimind_generate src/csrc/tools/minimind_generate.cpp)
target_link_libraries(minimind_generate PRIVATE minimind_orangepi_runtime)
target_compile_options(minimind_generate PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bench_decode bench/bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE minimind_orangepi_runtime)
target_compile_options(bench_decode PRIVATE -Wall -Wextra -Wpedantic)

add_executable(check_prefill_consistency bench/check_prefill_consistency.cpp)
target_link_libraries(check_prefill_consistency PRIVATE minimind_orangepi_runtime)
target_compile_options(check_prefill_consistency PRIVATE -Wall -Wextra -Wpedantic)

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
minimind_add_unit_test(test_ops tests/unit/test_ops.cpp)
minimind_add_unit_test(test_ascend_custom_ops tests/unit/test_ascend_custom_ops.cpp)
minimind_add_unit_test(test_tokenizer tests/unit/test_tokenizer.cpp)
minimind_add_unit_test(test_decoder_layer tests/unit/test_decoder_layer.cpp)
minimind_add_unit_test(test_language_model tests/unit/test_language_model.cpp)
minimind_add_unit_test(test_runtime_loader tests/unit/test_runtime_loader.cpp)
minimind_add_unit_test(test_moe_top1 tests/unit/test_moe_top1.cpp)
minimind_add_unit_test(test_minimind_v_config tests/unit/test_minimind_v_config.cpp)
minimind_add_unit_test(test_projector tests/unit/test_projector.cpp)
minimind_add_unit_test(test_vlm_model tests/unit/test_vlm_model.cpp)
minimind_add_unit_test(test_minimind_o_config tests/unit/test_minimind_o_config.cpp)
