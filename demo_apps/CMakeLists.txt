cmake_minimum_required(VERSION 3.15...3.27)
project(wosx-demo-apps)

################################################################################
# add dependencies
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope)
list(APPEND WOSX_DEMO_APPS_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/stb)
list(APPEND WOSX_DEMO_APPS_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nlohmann)
list(APPEND WOSX_DEMO_APPS_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/utils)

################################################################################
# build basic 2D demo app (cpu backend)
add_executable(basic_2d_demo basic_2d/demo.cpp)
target_link_libraries(basic_2d_demo wosx)
target_include_directories(basic_2d_demo PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})
# this demo does not link polyscope, so it must emit the stb implementation itself
target_compile_definitions(basic_2d_demo PRIVATE WOSX_DEFINE_STB_IMPLEMENTATION)

if(MSVC)
    target_compile_options(basic_2d_demo PRIVATE /bigobj)
endif()

# build demo apps (gpu backend)
if(WOSX_ENABLE_GPU_SUPPORT)
    add_executable(basic_2d_gpu_demo basic_2d/gpu_demo.cpp)
    target_link_libraries(basic_2d_gpu_demo wosx)
    target_include_directories(basic_2d_gpu_demo PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})
    # this demo does not link polyscope, so it must emit the stb implementation itself
    target_compile_definitions(basic_2d_gpu_demo PRIVATE WOSX_DEFINE_STB_IMPLEMENTATION)

    add_executable(heterogeneous_rectilinear_walk_gpu heterogeneous_layers/rectilinear_walk_gpu.cpp)
    target_link_libraries(heterogeneous_rectilinear_walk_gpu wosx)

    add_executable(heterogeneous_triangle_mesh_walk_gpu heterogeneous_layers/triangle_mesh_walk_gpu.cpp)
    target_link_libraries(heterogeneous_triangle_mesh_walk_gpu wosx)
    target_include_directories(heterogeneous_triangle_mesh_walk_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/happly)

    add_executable(electrostatics electrostatics/app.cpp)
    target_link_libraries(electrostatics wosx polyscope)
    target_include_directories(electrostatics PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})

    add_executable(thermal_conduction thermal_conduction/app.cpp)
    target_link_libraries(thermal_conduction wosx polyscope)
    target_include_directories(thermal_conduction PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})

    add_executable(potential_flow potential_flow/app.cpp)
    target_link_libraries(potential_flow wosx polyscope)
    target_include_directories(potential_flow PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})

    add_executable(geometric_deformation geometric_deformation/app.cpp)
    target_link_libraries(geometric_deformation wosx polyscope)
    target_include_directories(geometric_deformation PRIVATE ${WOSX_DEMO_APPS_DEPS_INCLUDES})

    if(MSVC)
        target_compile_options(basic_2d_gpu_demo PRIVATE /bigobj)
        target_compile_options(electrostatics PRIVATE /bigobj)
        target_compile_options(thermal_conduction PRIVATE /bigobj)
        target_compile_options(potential_flow PRIVATE /bigobj)
        target_compile_options(geometric_deformation PRIVATE /bigobj)
    endif()
endif()
