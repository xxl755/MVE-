# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xxl/桌面/mybin/modeling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xxl/桌面/mybin/modeling/build

# Include any dependencies generated for this target.
include sfm/CMakeFiles/sfm.dir/depend.make

# Include the progress variables for this target.
include sfm/CMakeFiles/sfm.dir/progress.make

# Include the compile flags for this target's objects.
include sfm/CMakeFiles/sfm.dir/flags.make

sfm/CMakeFiles/sfm.dir/camera_database.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/camera_database.cc.o: ../sfm/camera_database.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sfm/CMakeFiles/sfm.dir/camera_database.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/camera_database.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/camera_database.cc

sfm/CMakeFiles/sfm.dir/camera_database.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/camera_database.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/camera_database.cc > CMakeFiles/sfm.dir/camera_database.cc.i

sfm/CMakeFiles/sfm.dir/camera_database.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/camera_database.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/camera_database.cc -o CMakeFiles/sfm.dir/camera_database.cc.s

sfm/CMakeFiles/sfm.dir/bundler_common.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_common.cc.o: ../sfm/bundler_common.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_common.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_common.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_common.cc

sfm/CMakeFiles/sfm.dir/bundler_common.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_common.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_common.cc > CMakeFiles/sfm.dir/bundler_common.cc.i

sfm/CMakeFiles/sfm.dir/bundler_common.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_common.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_common.cc -o CMakeFiles/sfm.dir/bundler_common.cc.s

sfm/CMakeFiles/sfm.dir/feature_set.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/feature_set.cc.o: ../sfm/feature_set.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object sfm/CMakeFiles/sfm.dir/feature_set.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/feature_set.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/feature_set.cc

sfm/CMakeFiles/sfm.dir/feature_set.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/feature_set.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/feature_set.cc > CMakeFiles/sfm.dir/feature_set.cc.i

sfm/CMakeFiles/sfm.dir/feature_set.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/feature_set.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/feature_set.cc -o CMakeFiles/sfm.dir/feature_set.cc.s

sfm/CMakeFiles/sfm.dir/ransac.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/ransac.cc.o: ../sfm/ransac.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object sfm/CMakeFiles/sfm.dir/ransac.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/ransac.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/ransac.cc

sfm/CMakeFiles/sfm.dir/ransac.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/ransac.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/ransac.cc > CMakeFiles/sfm.dir/ransac.cc.i

sfm/CMakeFiles/sfm.dir/ransac.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/ransac.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/ransac.cc -o CMakeFiles/sfm.dir/ransac.cc.s

sfm/CMakeFiles/sfm.dir/fundamental.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/fundamental.cc.o: ../sfm/fundamental.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object sfm/CMakeFiles/sfm.dir/fundamental.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/fundamental.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/fundamental.cc

sfm/CMakeFiles/sfm.dir/fundamental.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/fundamental.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/fundamental.cc > CMakeFiles/sfm.dir/fundamental.cc.i

sfm/CMakeFiles/sfm.dir/fundamental.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/fundamental.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/fundamental.cc -o CMakeFiles/sfm.dir/fundamental.cc.s

sfm/CMakeFiles/sfm.dir/ransac_homography.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/ransac_homography.cc.o: ../sfm/ransac_homography.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object sfm/CMakeFiles/sfm.dir/ransac_homography.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/ransac_homography.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/ransac_homography.cc

sfm/CMakeFiles/sfm.dir/ransac_homography.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/ransac_homography.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/ransac_homography.cc > CMakeFiles/sfm.dir/ransac_homography.cc.i

sfm/CMakeFiles/sfm.dir/ransac_homography.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/ransac_homography.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/ransac_homography.cc -o CMakeFiles/sfm.dir/ransac_homography.cc.s

sfm/CMakeFiles/sfm.dir/homography.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/homography.cc.o: ../sfm/homography.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object sfm/CMakeFiles/sfm.dir/homography.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/homography.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/homography.cc

sfm/CMakeFiles/sfm.dir/homography.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/homography.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/homography.cc > CMakeFiles/sfm.dir/homography.cc.i

sfm/CMakeFiles/sfm.dir/homography.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/homography.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/homography.cc -o CMakeFiles/sfm.dir/homography.cc.s

sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.o: ../sfm/ransac_fundamental.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/ransac_fundamental.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/ransac_fundamental.cc

sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/ransac_fundamental.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/ransac_fundamental.cc > CMakeFiles/sfm.dir/ransac_fundamental.cc.i

sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/ransac_fundamental.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/ransac_fundamental.cc -o CMakeFiles/sfm.dir/ransac_fundamental.cc.s

sfm/CMakeFiles/sfm.dir/pose_p3p.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/pose_p3p.cc.o: ../sfm/pose_p3p.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object sfm/CMakeFiles/sfm.dir/pose_p3p.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/pose_p3p.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/pose_p3p.cc

sfm/CMakeFiles/sfm.dir/pose_p3p.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/pose_p3p.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/pose_p3p.cc > CMakeFiles/sfm.dir/pose_p3p.cc.i

sfm/CMakeFiles/sfm.dir/pose_p3p.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/pose_p3p.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/pose_p3p.cc -o CMakeFiles/sfm.dir/pose_p3p.cc.s

sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o: ../sfm/ransac_pose_p3p.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/ransac_pose_p3p.cc

sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/ransac_pose_p3p.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/ransac_pose_p3p.cc > CMakeFiles/sfm.dir/ransac_pose_p3p.cc.i

sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/ransac_pose_p3p.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/ransac_pose_p3p.cc -o CMakeFiles/sfm.dir/ransac_pose_p3p.cc.s

sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.o: ../sfm/bundle_adjustment.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundle_adjustment.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundle_adjustment.cc

sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundle_adjustment.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundle_adjustment.cc > CMakeFiles/sfm.dir/bundle_adjustment.cc.i

sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundle_adjustment.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundle_adjustment.cc -o CMakeFiles/sfm.dir/bundle_adjustment.cc.s

sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.o: ../sfm/ba_linear_solver.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/ba_linear_solver.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/ba_linear_solver.cc

sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/ba_linear_solver.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/ba_linear_solver.cc > CMakeFiles/sfm.dir/ba_linear_solver.cc.i

sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/ba_linear_solver.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/ba_linear_solver.cc -o CMakeFiles/sfm.dir/ba_linear_solver.cc.s

sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.o: ../sfm/extract_focal_length.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/extract_focal_length.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/extract_focal_length.cc

sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/extract_focal_length.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/extract_focal_length.cc > CMakeFiles/sfm.dir/extract_focal_length.cc.i

sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/extract_focal_length.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/extract_focal_length.cc -o CMakeFiles/sfm.dir/extract_focal_length.cc.s

sfm/CMakeFiles/sfm.dir/triangulate.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/triangulate.cc.o: ../sfm/triangulate.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object sfm/CMakeFiles/sfm.dir/triangulate.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/triangulate.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/triangulate.cc

sfm/CMakeFiles/sfm.dir/triangulate.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/triangulate.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/triangulate.cc > CMakeFiles/sfm.dir/triangulate.cc.i

sfm/CMakeFiles/sfm.dir/triangulate.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/triangulate.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/triangulate.cc -o CMakeFiles/sfm.dir/triangulate.cc.s

sfm/CMakeFiles/sfm.dir/bundler_features.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_features.cc.o: ../sfm/bundler_features.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_features.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_features.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_features.cc

sfm/CMakeFiles/sfm.dir/bundler_features.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_features.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_features.cc > CMakeFiles/sfm.dir/bundler_features.cc.i

sfm/CMakeFiles/sfm.dir/bundler_features.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_features.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_features.cc -o CMakeFiles/sfm.dir/bundler_features.cc.s

sfm/CMakeFiles/sfm.dir/bundler_matching.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_matching.cc.o: ../sfm/bundler_matching.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_matching.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_matching.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_matching.cc

sfm/CMakeFiles/sfm.dir/bundler_matching.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_matching.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_matching.cc > CMakeFiles/sfm.dir/bundler_matching.cc.i

sfm/CMakeFiles/sfm.dir/bundler_matching.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_matching.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_matching.cc -o CMakeFiles/sfm.dir/bundler_matching.cc.s

sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.o: ../sfm/bundler_intrinsics.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_intrinsics.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_intrinsics.cc

sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_intrinsics.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_intrinsics.cc > CMakeFiles/sfm.dir/bundler_intrinsics.cc.i

sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_intrinsics.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_intrinsics.cc -o CMakeFiles/sfm.dir/bundler_intrinsics.cc.s

sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.o: ../sfm/bundler_tracks.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_tracks.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_tracks.cc

sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_tracks.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_tracks.cc > CMakeFiles/sfm.dir/bundler_tracks.cc.i

sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_tracks.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_tracks.cc -o CMakeFiles/sfm.dir/bundler_tracks.cc.s

sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.o: ../sfm/bundler_incremental.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_incremental.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_incremental.cc

sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_incremental.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_incremental.cc > CMakeFiles/sfm.dir/bundler_incremental.cc.i

sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_incremental.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_incremental.cc -o CMakeFiles/sfm.dir/bundler_incremental.cc.s

sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.o: sfm/CMakeFiles/sfm.dir/flags.make
sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.o: ../sfm/bundler_init_pair.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Building CXX object sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.o"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/bundler_init_pair.cc.o -c /home/xxl/桌面/mybin/modeling/sfm/bundler_init_pair.cc

sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/bundler_init_pair.cc.i"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxl/桌面/mybin/modeling/sfm/bundler_init_pair.cc > CMakeFiles/sfm.dir/bundler_init_pair.cc.i

sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/bundler_init_pair.cc.s"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxl/桌面/mybin/modeling/sfm/bundler_init_pair.cc -o CMakeFiles/sfm.dir/bundler_init_pair.cc.s

# Object files for target sfm
sfm_OBJECTS = \
"CMakeFiles/sfm.dir/camera_database.cc.o" \
"CMakeFiles/sfm.dir/bundler_common.cc.o" \
"CMakeFiles/sfm.dir/feature_set.cc.o" \
"CMakeFiles/sfm.dir/ransac.cc.o" \
"CMakeFiles/sfm.dir/fundamental.cc.o" \
"CMakeFiles/sfm.dir/ransac_homography.cc.o" \
"CMakeFiles/sfm.dir/homography.cc.o" \
"CMakeFiles/sfm.dir/ransac_fundamental.cc.o" \
"CMakeFiles/sfm.dir/pose_p3p.cc.o" \
"CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o" \
"CMakeFiles/sfm.dir/bundle_adjustment.cc.o" \
"CMakeFiles/sfm.dir/ba_linear_solver.cc.o" \
"CMakeFiles/sfm.dir/extract_focal_length.cc.o" \
"CMakeFiles/sfm.dir/triangulate.cc.o" \
"CMakeFiles/sfm.dir/bundler_features.cc.o" \
"CMakeFiles/sfm.dir/bundler_matching.cc.o" \
"CMakeFiles/sfm.dir/bundler_intrinsics.cc.o" \
"CMakeFiles/sfm.dir/bundler_tracks.cc.o" \
"CMakeFiles/sfm.dir/bundler_incremental.cc.o" \
"CMakeFiles/sfm.dir/bundler_init_pair.cc.o"

# External object files for target sfm
sfm_EXTERNAL_OBJECTS =

sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/camera_database.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_common.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/feature_set.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/ransac.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/fundamental.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/ransac_homography.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/homography.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/ransac_fundamental.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/pose_p3p.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/ransac_pose_p3p.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundle_adjustment.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/ba_linear_solver.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/extract_focal_length.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/triangulate.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_features.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_matching.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_intrinsics.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_tracks.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_incremental.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/bundler_init_pair.cc.o
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/build.make
sfm/libsfm.a: sfm/CMakeFiles/sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xxl/桌面/mybin/modeling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_21) "Linking CXX static library libsfm.a"
	cd /home/xxl/桌面/mybin/modeling/build/sfm && $(CMAKE_COMMAND) -P CMakeFiles/sfm.dir/cmake_clean_target.cmake
	cd /home/xxl/桌面/mybin/modeling/build/sfm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sfm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sfm/CMakeFiles/sfm.dir/build: sfm/libsfm.a

.PHONY : sfm/CMakeFiles/sfm.dir/build

sfm/CMakeFiles/sfm.dir/clean:
	cd /home/xxl/桌面/mybin/modeling/build/sfm && $(CMAKE_COMMAND) -P CMakeFiles/sfm.dir/cmake_clean.cmake
.PHONY : sfm/CMakeFiles/sfm.dir/clean

sfm/CMakeFiles/sfm.dir/depend:
	cd /home/xxl/桌面/mybin/modeling/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xxl/桌面/mybin/modeling /home/xxl/桌面/mybin/modeling/sfm /home/xxl/桌面/mybin/modeling/build /home/xxl/桌面/mybin/modeling/build/sfm /home/xxl/桌面/mybin/modeling/build/sfm/CMakeFiles/sfm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sfm/CMakeFiles/sfm.dir/depend
