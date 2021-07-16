project(work2)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")

set(INCREMENTAL_SFM_SOURCES
        class4_test_incremental_sfm.cc defines.h functions.cc functions.h)
add_executable(Job_incremental_sfm ${INCREMENTAL_SFM_SOURCES})
target_link_libraries(Job_incremental_sfm sfm util core features)

