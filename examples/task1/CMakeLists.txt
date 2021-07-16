project(class1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories("../..")#include all the brary，back 2 step

set(VISUAL_HEADER visualizer.h)  #${VISUAL_HEADER}
set(VISUAL_SOURCE visualizer.cc)#set function is to get a variable for file

# test features
set(FEATURES_FILES
        class1_test_features.cc
        )
add_executable(task1_test_features ${FEATURES_FILES} ${VISUAL_HEADER} ${VISUAL_SOURCE})#get execute file
target_link_libraries(task1_test_features util core features)#project link to library

# test matching
set(MATCHING_FILES
        class1_test_matching.cc
        )
add_executable(task1_test_matching ${MATCHING_FILES} ${VISUAL_HEADER} ${VISUAL_SOURCE})
target_link_libraries(task1_test_matching sfm util core features)

