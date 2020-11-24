
cmake_minimum_required(VERSION 2.8)
 
project(DeformableMIMultiStage)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(DeformableMIMultiStage MACOSX_BUNDLE DeformableMIMultiStage.cxx)
target_link_libraries(DeformableMIMultiStage ${ITK_LIBRARIES})
