cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME ptp.object-search.point-process-entropy-planners)
include(cmake/pods.cmake)

add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -fPIC )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)


add_library( ptp.object-search.point-process-entropy-planners SHARED
  src/one_action_entropy_reduction_planner.cpp
  )

pods_install_headers( 
  src/one_action_entropy_reduction_planner.hpp
  DESTINATION
  point-process-entropy-planners
  )

pods_use_pkg_config_packages(ptp.object-search.point-process-entropy-planners
  p2l-planner-core
  p2l-igmm-point-process
  p2l-ruler-point-process
  )

pods_install_libraries( ptp.object-search.point-process-entropy-planners )
pods_install_pkg_config_file(ptp.object-search.point-process-entropy-planners
    CFLAGS
    LIBS -lptp.object-search.point-process-entropy-planners
    REQUIRES boost-1.54.0 p2l-planner-core p2l-igmm-point-process p2l-ruler-point-process
    VERSION 0.0.1)
  
  
