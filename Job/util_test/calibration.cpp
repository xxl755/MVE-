//
// Created by xxl on 2021/5/7.
//
#include "calibration.h"
void getFiles(string path,vector<string>& files){
    struct dirent *ptr;
    DIR *dir;

    dir=opendir(path.c_str());//返回文件夹类

    cout << "标定图片列表: "<< endl;
    while((ptr=readdir(dir))!=NULL)//怪怪的读入文件名,返回文件类
    {

        //跳过'.'和'..'两个目
        if(ptr->d_name[0] == '.')//跳过文件夹
            continue;
//cout << ptr->d_name << endl;
        files.push_back(ptr->d_name);
    }//读尽文件名
    for (int i = 0; i < files.size(); ++i)
    {
        cout << files[i] << endl;
    }

    closedir(dir);
}

int getCornerpoints(vector<string>&files,vector<Point2f>& image_points_buf, vector<vector<Point2f>>& image_points_seq,Size& image_size){

    Size board_size = Size(BOARD_HEIGHT,BOARD_WIDTH);    /* 标定板上每行、列的角点数 */
    int image_count=0;
    for(int i=0;i<files.size();i++) {
        string filepath = "./Job/chess/";
        filepath.append(files[i]);
        Mat imageInput = imread(filepath);
        if (imageInput.empty())
            return 1;

        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf)) {
            cout << "can not find chessboard corners!\n";  // 找不到角点
            continue;
        } else {
            image_count++;
            if (image_count == 1)  //读入第一张图片时获取图像宽高信息
            {
                image_size.width = imageInput.cols;
                image_size.height = imageInput.rows;
                cout << "image_size.width = " << image_size.width << endl;
                cout << "image_size.height = " << image_size.height << endl;
            }

            Mat view_gray;

            cvtColor(imageInput, view_gray, COLOR_RGB2GRAY);

            /* 亚像素精确化 */
            //find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5)); //对粗提取的角点进行精确化
            cornerSubPix(view_gray, image_points_buf,
                         Size(5, 5),
                         Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
                                      30,        // max number of iterations
                                      0.1));     // min accuracy

            image_points_seq.push_back(image_points_buf);  //保存亚像素角点

            /* 在图像上显示角点位置 */
            drawChessboardCorners(view_gray, board_size, image_points_buf, true); //用于在图片中标记角点

            //写入文件
            string savePath="./Job/corner_chess/";
            savePath.append(files[i]);
            imwrite(savePath,view_gray);
        }
    }
    return image_count;
}

void calibration(string filepath){
    vector<string> files;
    files.clear();

    getFiles(filepath, files );

    //读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
    cout<<"开始提取角点………………"<<endl;
//    int image_count=0;  /* 图像数量 */

    Size board_size = Size(BOARD_HEIGHT,BOARD_WIDTH);    /* 标定板上每行、列的角点数 */
    vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
    vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
    //  string filetest="chess0.jpeg";
    Size image_size;
    int image_count=getCornerpoints(files,image_points_buf,image_points_seq,image_size);

    int total = image_points_seq.size();
    cout<< "共使用了"<<total << "幅图片"<<endl;
    cout<<"角点提取完成！\n";


    cout<<"开始标定………………\n";
    /*棋盘三维信息*/
    Size square_size = Size(BOARD_SCALE,BOARD_SCALE);  /* 实际测量得到的标定板上每个棋盘格的大小 */
    vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
    /*内外参数*/
    Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */
    vector<int> point_counts;  // 每幅图像中角点的数量
    Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
    vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
    /* 初始化标定板上角点的三维坐标 */
    int i,j,t;
    for (t=0;t<image_count;t++)
    {
        vector<Point3f> tempPointSet;
        for (i=0;i<board_size.height;i++)
        {
            for (j=0;j<board_size.width;j++)
            {
                Point3f realPoint;
                /* 假设标定板放在世界坐标系中z=0的平面上 *///标定版原点设在板子左上角
                realPoint.x = i*square_size.width;
                realPoint.y = j*square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);//每幅图的真实三维点坐标都定为相同
    }

    /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
    for (i=0;i<image_count;i++)
    {
        point_counts.push_back(board_size.width*board_size.height);
    }


    /* 开始标定 *///三维点世界坐标系向量，二维点像素坐标系向量，图片像素坐标系尺寸，5个扭曲3个径向，2个切向，旋转向量，平移向量
    calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,CALIB_RATIONAL_MODEL);
    cout<<"标定完成！\n";

    //对标定结果进行评价
    ofstream fout("./Job/caliberation_result.txt");  /* 保存标定结果的文件 */

    double total_err = 0.0; /* 所有图像的平均误差的总和 */
    double err = 0.0; /* 每幅图像的平均误差 */
    vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
    cout<<"\t每幅图像的标定误差：\n";
    fout<<"每幅图像的标定误差：\n";
    for (i=0;i<image_count;i++)
    {
        vector<Point3f> tempPointSet=object_points[i];//每幅图的三维点世界坐标系，每幅图48点
        /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点image_points2 */
        projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);
        /* 计算新的投影点和旧的投影点之间的误差*/
        vector<Point2f> tempImagePoint = image_points_seq[i];
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);//图像Mat类别
        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
        for (int j = 0 ; j < tempImagePoint.size(); j++)
        {
            image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
            tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);//平方和
        total_err += err/=  point_counts[i];
        cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
        fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
    }

    cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;
    fout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;

    //保存定标结果
    cout<<"开始保存定标结果………………"<<endl;
    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

//只需要相机内参,不能直接输出保存
    fout<<"相机内参数矩阵："<<endl;
//    fout<<cameraMatrix<<endl<<endl;
    for (int r = 0; r < cameraMatrix.rows; r++)
    {
        for (int c = 0; c < cameraMatrix.cols; c++)
        {
            double data = cameraMatrix.at<double>(r,c); //读取数据，at<type> - type 是矩阵元素的具体数据格式
            fout << data << "\t" ; //每列数据用 tab 隔开
        }
        fout << endl; //换行
    }
//    for (int h=0;h<cameraMatrix.rows;h++){
//        for (int i = 0; i < cameraMatrix.cols; ++i) {
//            double f=cameraMatrix.at<double>(h,i);
//            cout<<f<<endl;
//        }
//    }
    fout<<"畸变系数：\n";
    fout<<distCoeffs<<endl<<endl<<endl;
    for (int i=0; i<image_count; i++)
    {
        fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;
        fout<<tvecsMat[i]<<endl;
        /* 将旋转向量转换为相对应的旋转矩阵 */
        Rodrigues(tvecsMat[i],rotation_matrix);
        fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;
        fout<<rotation_matrix<<endl;
        fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;
        fout<<rvecsMat[i]<<endl<<endl;
    }
    cout<<"完成保存"<<endl;
    fout<<endl;

}


ifstream & seek_to_line(ifstream & in, int line)
//将打开的文件in，定位到line行。
{
    int i;
    char buf[1024];
    in.seekg(0, ios::beg);  //定位到文件开始。
    for (i = 0; i < line; i++) {
        in.getline(buf, sizeof(buf));//读取行。读入数据流不做任何处理，相当于跳过
    }
    return in;
}

void getCameraMatrix(string filename,Mat& k_matrix){
    ifstream readLine;
    int n=0;
    string lines="相机内参数矩阵：";
    readLine.open(filename.c_str());

    string temp;

    if(readLine.fail()){
        return ;
    }
    else {
        while (getline(readLine, temp, '\n')) {
            n++;
            if (temp == lines) {
                readLine.close();
                break;
            }
        }
      ifstream readFile(filename.c_str());
        seek_to_line(readFile,n);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                readFile >> k_matrix.at<float>(i, j);
                cout << k_matrix.at<float>(i, j) << endl;
            }
        }
        readFile.close();
    }

}