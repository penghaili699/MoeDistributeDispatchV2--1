set(E2E_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_dispatch_e2e_kernel.cpp)
set(E2E_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/moe_dispatch_e2e_plugin.cpp)
register_direct_launch_op("${E2E_KERNEL_SRCS}" op_kernel "${E2E_PLUGIN_SRCS}" op_kernel)
