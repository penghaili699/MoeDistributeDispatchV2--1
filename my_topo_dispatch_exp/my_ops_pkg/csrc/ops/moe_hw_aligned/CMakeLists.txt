set(HW_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_hw_aligned_kernel.cpp)
set(HW_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/moe_hw_aligned_plugin.cpp)
register_direct_launch_op("${HW_KERNEL_SRCS}" op_kernel "${HW_PLUGIN_SRCS}" op_kernel)
