project(work1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")
find_package(OpenCV REQUIRED)
set(BA_FILE bundle_adjustment.cc calibration.h calibration.cpp )
add_executable(Job_BA_test ${BA_FILE})
target_link_libraries(Job_BA_test sfm util core features ${OpenCV_LIBS})

