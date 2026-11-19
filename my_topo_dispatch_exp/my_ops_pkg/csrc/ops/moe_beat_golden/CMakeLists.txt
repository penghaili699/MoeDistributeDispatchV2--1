set(BG_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_beat_golden_kernel.cpp)
set(BG_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/moe_beat_golden_plugin.cpp)
register_direct_launch_op("${BG_KERNEL_SRCS}" op_kernel "${BG_PLUGIN_SRCS}" op_kernel)
