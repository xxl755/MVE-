cmake_minimum_required(VERSION 3.5)
project(ImageBasedModellingEdu)#roject name
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})#put .cmake output file into new project

# Eigen (required)
find_package(Eigen REQUIRED)#find package(needed is required),into findeigen.cmake.there are many search path(sudo install in usr,our install in usr/local)
include_directories(${EIGEN_INCLUDE_DIRS})#include eigen header file,it haven't the library,don' need link
add_definitions(-DEIGEN_USE_NEW_STDVECTOR -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)


set(CMAKE_CXX_STANDARD 11)#project standard
set(CMAKE_CXX_FLAGS "-fPIC")#prevent mistake

add_subdirectory(core)#add others library
add_subdirectory(util)
add_subdirectory(features)
add_subdirectory(sfm)
add_subdirectory(mvs)
add_subdirectory(examples)
add_subdirectory(surface)
add_subdirectory(texturing)
add_subdirectory(3rdParty/mrf)
add_subdirectory(3rdParty/coldet)
add_subdirectory(3rdParty/gco)
add_subdirectory(Job)
#add_subdirectory(3rdParty/rayint)
#add_subdirectory(3rdParty/eigen)
#add_subdirectory(3rdParty/mapmap)