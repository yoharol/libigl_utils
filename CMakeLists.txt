cmake_minimum_required(VERSION 3.21)

project(libigl_utils)

set(CMAKE_CXX_STANDARD 17)

get_filename_component(ABS_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)

message(${ABS_SOURCE_PATH})

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(pxr PATHS "/usr/local/USD" REQUIRED)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

option(LIBIGL_USE_STATIC_LIBRARY ON)

get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("Imported targets: ${importTargets}")

add_definitions(-DASSETS_PATH="${ABS_SOURCE_PATH}/models")
add_definitions(-DTUTORIAL_SHARED_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models")

message(${PXR_INCLUDE_DIRS})

function(AddLibiglExec func_name)
  add_executable(${func_name})
  target_sources(
    ${func_name}
    PRIVATE
    ${func_name}.cpp
  )
  target_include_directories(${func_name}
    PRIVATE
    ${PXR_INCLUDE_DIRS}
  )
  target_link_libraries(${func_name}
    PRIVATE
    igl::core
    igl::glfw
    usd
    usdGeom
    vt
  )


endfunction()

AddLibiglExec(ex1_base)

AddLibiglExec(ex2_dual_skinning)

AddLibiglExec(ex3_biharmonic_weights)

AddLibiglExec(ex4_fast_skinning_transformation)

AddLibiglExec(ex5_starfish)

AddLibiglExec(ex6_arap)

AddLibiglExec(ex7_starfish_arap)
AddLibiglExec(ex8_elip_fast)
AddLibiglExec(ex8_elip_arap)

AddLibiglExec(ex9_cube_fast)
AddLibiglExec(ex9_cube_arap)


AddLibiglExec(ex10_mushroom_fast)
AddLibiglExec(ex10_mushroom_arap)


AddLibiglExec(ex11_bunny_fast)
AddLibiglExec(ex11_bunny_arap)


AddLibiglExec(ex12_ghost_fast)
AddLibiglExec(ex12_ghost_arap)
AddLibiglExec(ex13_bbw)
AddLibiglExec(ex14_cage_bbw)
AddLibiglExec(ex15_cage_mesh_bbw)
AddLibiglExec(ex16_cheb)
AddLibiglExec(ex17_spot)
