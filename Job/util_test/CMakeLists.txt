project(incremental_test)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")
find_package(OpenCV REQUIRED)
set(INCREMENTAL_TEST
        incremental_test.cpp functions.cc functions.h defines.h calibration.cpp calibration.h)
add_executable(incremental_test ${INCREMENTAL_TEST})
target_link_libraries(incremental_test sfm util core features ${OpenCV_LIBS})