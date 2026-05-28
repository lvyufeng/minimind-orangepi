foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
  add_ops_info_target(TARGET ops_info_gen_${compute_unit}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
    OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
    INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
  )
endforeach()

add_npu_support_target(TARGET npu_supported_ops
  OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
  OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
  INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
        FILES_MATCHING PATTERN "*.cpp")
