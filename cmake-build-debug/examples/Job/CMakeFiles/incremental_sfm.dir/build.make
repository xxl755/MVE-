# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/xxl/桌面/myfile/clion-2020.2.5/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/xxl/桌面/myfile/clion-2020.2.5/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xxl/桌面/mybin/modeling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xxl/桌面/mybin/modeling/cmake-build-debug

# Include any dependencies generated for this target.
include examples/Job/CMakeFiles/incremental_sfm.dir/depend.make

# Include the progress variables for this target.
include examples/Job/CMakeFiles/incremental_sfm.dir/progress.make

# Include the compile flags for this target's objects.
include examples/Job/CMakeFiles/incremental_sfm.dir/flags.make

examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o: examples/Job/CMakeFiles/incremental_sfm.dir/flags.make
examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o: ../examples/Job/class4_test_incremental_sfm.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o -c /home/xxl/桌面/mybin/modeling/examples/Job/class4_test_incremental_sfm.cc

examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.i"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/examples/Job/class4_test_incremental_sfm.cc > CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.i

examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.s"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/examples/Job/class4_test_incremental_sfm.cc -o CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.s

examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.o: examples/Job/CMakeFiles/incremental_sfm.dir/flags.make
examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.o: ../examples/Job/functions.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.o"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_sfm.dir/functions.cc.o -c /home/xxl/桌面/mybin/modeling/examples/Job/functions.cc

examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_sfm.dir/functions.cc.i"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/examples/Job/functions.cc > CMakeFiles/incremental_sfm.dir/functions.cc.i

examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_sfm.dir/functions.cc.s"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/examples/Job/functions.cc -o CMakeFiles/incremental_sfm.dir/functions.cc.s

# Object files for target incremental_sfm
incremental_sfm_OBJECTS = \
"CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o" \
"CMakeFiles/incremental_sfm.dir/functions.cc.o"

# External object files for target incremental_sfm
incremental_sfm_EXTERNAL_OBJECTS =

examples/Job/incremental_sfm: examples/Job/CMakeFiles/incremental_sfm.dir/class4_test_incremental_sfm.cc.o
examples/Job/incremental_sfm: examples/Job/CMakeFiles/incremental_sfm.dir/functions.cc.o
examples/Job/incremental_sfm: examples/Job/CMakeFiles/incremental_sfm.dir/build.make
examples/Job/incremental_sfm: sfm/libsfm.a
examples/Job/incremental_sfm: util/libutil.a
examples/Job/incremental_sfm: core/libcore.a
examples/Job/incremental_sfm: features/libfeatures.a
examples/Job/incremental_sfm: core/libcore.a
examples/Job/incremental_sfm: util/libutil.a
examples/Job/incremental_sfm: /usr/lib/x86_64-linux-gnu/libpng.so
examples/Job/incremental_sfm: /usr/lib/x86_64-linux-gnu/libz.so
examples/Job/incremental_sfm: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/Job/incremental_sfm: /usr/lib/x86_64-linux-gnu/libtiff.so
examples/Job/incremental_sfm: examples/Job/CMakeFiles/incremental_sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xxl/桌面/mybin/modeling/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable incremental_sfm"
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/incremental_sfm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/Job/CMakeFiles/incremental_sfm.dir/build: examples/Job/incremental_sfm

.PHONY : examples/Job/CMakeFiles/incremental_sfm.dir/build

examples/Job/CMakeFiles/incremental_sfm.dir/clean:
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job && $(CMAKE_COMMAND) -P CMakeFiles/incremental_sfm.dir/cmake_clean.cmake
.PHONY : examples/Job/CMakeFiles/incremental_sfm.dir/clean

examples/Job/CMakeFiles/incremental_sfm.dir/depend:
	cd /home/xxl/桌面/mybin/modeling/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xxl/桌面/mybin/modeling /home/xxl/桌面/mybin/modeling/examples/Job /home/xxl/桌面/mybin/modeling/cmake-build-debug /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job /home/xxl/桌面/mybin/modeling/cmake-build-debug/examples/Job/CMakeFiles/incremental_sfm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/Job/CMakeFiles/incremental_sfm.dir/depend

