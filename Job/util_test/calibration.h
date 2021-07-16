//
// Created by xxl on 2021/5/7.
//

#ifndef IMAGEBASEDMODELLINGEDU_CALIBRATION_H
#define IMAGEBASEDMODELLINGEDU_CALIBRATION_H
#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sys/io.h>
#include <dirent.h>
#include <vector>
#include <string>

const int BOARD_SCALE=34;
//#define BOARD_HEIGHT 6
const int BOARD_HEIGHT=4;
//#define BOARD_WIDTH 8
const int BOARD_WIDTH=6;

using namespace std;
using namespace cv;
void getFiles(string path, vector<string>& files);

int getCornerpoints(vector<string>&files,vector<Point2f>& image_points_buf, vector<vector<Point2f>>& image_points_seq);

void calibration(string filepath);

void getCameraMatrix(string filename,Mat& innexMatrix);

string readLine(string filename,int line);
#endif //IMAGEBASEDMODELLINGEDU_CALIBRATION_H
