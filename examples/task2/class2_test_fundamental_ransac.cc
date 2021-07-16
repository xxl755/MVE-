/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <util/system.h>
#include <assert.h>
#include <sfm/ransac_fundamental.h>
#include "math/functions.h"
#include "sfm/fundamental.h"
#include "sfm/correspondence.h"
#include "math/matrix_svd.h"

typedef math::Matrix<double, 3, 3> FundamentalMatrix;


/**
 * \description 用于RANSAC采样成功所需要的采样次数
 * @param p -- 内点的概率
 * @param K --拟合模型需要的样本个数，对应基础矩阵num_samples=8
 * @param z  -- 预期的采样成功的概率
 *                          log(1-z)
 *       需要的采样次数 M = -----------
 *                          log(1-p^K)
 * Example: For p = 50%, z = 99%, n = 8: M = log(0.001) / log(0.99609) = 1176.
 * 需要采样1176次从而保证RANSAC的成功率不低于0.99.
 * @return
 */
int  calc_ransac_iterations (double p,
                           int K,
                           double z = 0.99){

    double prob_all_good = math::fastpow(p, K);
    double num_iterations = std::log(1.0 - z)
                            / std::log(1.0 - prob_all_good);
    return static_cast<int>(math::round(num_iterations));

}

/**
 * \description 给定基础矩阵和一对匹配点，计算匹配点的sampson 距离，用于判断匹配点是否是内点,
 * 计算公式如下：
 *              SD = (x'Fx)^2 / ( (Fx)_1^2 + (Fx)_2^2 + (x'F)_1^2 + (x'F)_2^2 )
 * @param F-- 基础矩阵
 * @param m-- 匹配对
 * @return
 */
double  calc_sampson_distance (FundamentalMatrix const& F, sfm::Correspondence2D2D const& m) {
//m是匹配两点的结构体
    double p2_F_p1 = 0.0;//x2*F*x1
    p2_F_p1 += m.p2[0] * (m.p1[0] * F[0] + m.p1[1] * F[1] + F[2]);//p1,p2都是(x,y,1）列向量，但这里计算顺序是 p2’(Fp1)
    p2_F_p1 += m.p2[1] * (m.p1[0] * F[3] + m.p1[1] * F[4] + F[5]);
    p2_F_p1 +=     1.0 * (m.p1[0] * F[6] + m.p1[1] * F[7] + F[8]);
    p2_F_p1 *= p2_F_p1;

    double sum = 0.0;
    sum += math::fastpow(m.p1[0] * F[0] + m.p1[1] * F[1] + F[2], 2);//获得(FX1)^2的第一行，就是(FX1)_x^2
    sum += math::fastpow(m.p1[0] * F[3] + m.p1[1] * F[4] + F[5], 2);
    sum += math::fastpow(m.p2[0] * F[0] + m.p2[1] * F[3] + F[6], 2);
    sum += math::fastpow(m.p2[0] * F[1] + m.p2[1] * F[4] + F[7], 2);

    return p2_F_p1 / sum;
}
/**
 * \description 8点发估计相机基础矩阵
 * @param pset1 -- 第一个视角的特征点
 * @param pset2 -- 第二个视角的特征点
 * @return 估计的基础矩阵
 */
void calc_fundamental_8_point (math::Matrix<double, 3, 8> const& pset1
        , math::Matrix<double, 3, 8> const& pset2
        ,FundamentalMatrix &F
){
    /* direct linear transform */
    math::Matrix<double, 8, 9> A;//基础矩阵F求解 x1Fx2，F有8个自由度，将矩阵乘开，一组点能求出一个F自由度元素，求解8行9列
    for(int i=0; i<8; i++)
    {//一组对应点求到一个本质矩阵元素
        math::Vec3d p1  = pset1.col(i);
        math::Vec3d p2 = pset2.col(i);

        A(i, 0) = p1[0]*p2[0];
        A(i, 1) = p1[1]*p2[0];
        A(i, 2) = p2[0];
        A(i, 3) = p1[0]*p2[1];
        A(i, 4) = p1[1]*p2[1];
        A(i, 5) = p2[1];
        A(i, 6) = p1[0];
        A(i, 7) = p1[1];
        A(i, 8) = 1.0;
    }

    math::Matrix<double, 9, 9> vv;
    math::matrix_svd<double, 8, 9>(A, nullptr, nullptr, &vv);
    math::Vector<double, 9> f = vv.col(8);//最小特征值对应的特征向量就是最优解

    F(0,0) = f[0]; F(0,1) = f[1]; F(0,2) = f[2];
    F(1,0) = f[3]; F(1,1) = f[4]; F(1,2) = f[5];
    F(2,0) = f[6]; F(2,1) = f[7]; F(2,2) = f[8];

    /* singularity constraint根据奇异值有0，重构基础矩阵 */
    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(F, &U, &S, &V);//svd分解最小特征值归0可以保证一定原图特征条件减小计算，滤波
    S(2,2)=0;
    F = U*S*V.transpose();
}

/**
 * \description 利用最小二乘法计算基础矩阵
 * @param matches--输入的匹配对 大于8对
 * @param F --基础矩阵
 */
void calc_fundamental_least_squares(sfm::Correspondences2D2D const & matches, FundamentalMatrix&F){

    if (matches.size() < 8)
        throw std::invalid_argument("At least 8 points required");
    /* Create Nx9 matrix A. Each correspondence creates on row in A. */
    std::vector<double> A(matches.size() * 9);//向量存储矩阵信息，每一组匹配点构成Af的一行A
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        sfm::Correspondence2D2D const& p = matches[i];
        A[i * 9 + 0] = p.p2[0] * p.p1[0];
        A[i * 9 + 1] = p.p2[0] * p.p1[1];
        A[i * 9 + 2] = p.p2[0] * 1.0;
        A[i * 9 + 3] = p.p2[1] * p.p1[0];
        A[i * 9 + 4] = p.p2[1] * p.p1[1];
        A[i * 9 + 5] = p.p2[1] * 1.0;
        A[i * 9 + 6] = 1.0     * p.p1[0];
        A[i * 9 + 7] = 1.0     * p.p1[1];
        A[i * 9 + 8] = 1.0     * 1.0;
    }

    /* Compute fundamental matrix using SVD. */
    std::vector<double> vv(9 * 9);
    //matrix_svd (T const* mat_a, int rows, int cols,T* mat_u, T* vec_s, T* mat_v, T const& epsilon)，传入指针
    math::matrix_svd<double>(&A[0], matches.size(), 9, nullptr, nullptr, &vv[0]);
    double v0=vv[0];
    /* Use last column of V as solution. *///获得最小二乘法SVD分解的最后一列作为最小特征值对应的最优解的基础矩阵
    for (int i = 0; i < 9; ++i)
        F[i] = vv[i * 9 + 8];

    /* singularity constraint */
    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(F, &U, &S, &V);
    S(2,2)=0;
    F = U*S*V.transpose();
}
/**
 * \description 给定匹配对和基础矩阵，计算内点的个数
 * @param matches
 * @param F
 * @return
 */
std::vector<int> find_inliers(sfm::Correspondences2D2D const & matches
    ,FundamentalMatrix const & F, const double & thresh){
    const double squared_thresh = thresh* thresh;

    std::vector<int> inliers;
    for(int i=0; i< matches.size(); i++){//遍历选择一组匹配点，
        double error = calc_sampson_distance(F, matches[i]);//内点距离计算
        if(error< squared_thresh){//内点判断标准
            inliers.push_back(i);//将满足门限的匹配点保存索引
        }
    }
    return inliers;
}



int main(int argc, char *argv[]){

    /** 加载归一化后的匹配对 */
    sfm::Correspondences2D2D corr_all;//Correspondence2D2D结构体向量，结构体内存储2个2为坐标
    std::ifstream in("./examples/task2/correspondences.txt");//文件中提供匹配点<fstream>
    assert(in.is_open());//<assert.h>

    std::string line, word;
    int n_line = 0;//加载匹配点
    while(getline(in, line)){//getline将in中内容一行行读入line

        std::stringstream stream(line);//stringstream用于多种类型与string的转换<sstream>
        if(n_line==0){//读入第一行
            int n_corrs = 0;//读入数据文件第一行，274总的点个数
            stream>> n_corrs;
            corr_all.resize(n_corrs);

            n_line ++;
            continue;
        }
        if(n_line>0){

            stream>>corr_all[n_line-1].p1[0]>>corr_all[n_line-1].p1[1]
                  >>corr_all[n_line-1].p2[0]>>corr_all[n_line-1].p2[1];//根据向量索引从0起，找到结构体并数据流入结构体内属性
        }
        n_line++;
    }

    /* 估计计算采用次数 */
    const float inlier_ratio =0.5;//内点概率p
    const int n_samples=8;//最少采样点个数K
    int n_iterations = calc_ransac_iterations(inlier_ratio, n_samples);//M=LOG(1-Z)/LOG(1-P^K),设k=0.99,迭代次数为1177

    // 内点阈值用于判读匹配对是否为内点
    const double inlier_thresh = 0.0015;

    // ransac 最终估计的内点
    std::vector<int> best_inliers;

    std::cout << "RANSAC-F: Running for " << n_iterations
              << " iterations, threshold " << inlier_thresh
              << "..." << std::endl;
    for(int i=0; i<n_iterations; i++){

        /* 1.0 随机找到8对不重复的匹配点 */
        std::set<int> indices;//set内存放不重复内容
        while(indices.size()<8){
            indices.insert(util::system::rand_int() % corr_all.size());//0~274之间任意索引
        }//set内插入无顺序插入

        math::Matrix<double, 3, 8> pset1, pset2;
        std::set<int>::const_iterator iter = indices.cbegin();//访问set元素但不能用迭代器改变元素值
        for(int j=0; j<8; j++, iter++){
          //  int data=*iter;
            sfm::Correspondence2D2D const & match = corr_all[*iter];//match存放匹配两点的结构体
//给定确定行0，1，2，变换列录入矩阵
            pset1(0, j) = match.p1[0];//x,y,1
            pset1(1, j) = match.p1[1];
            pset1(2, j) = 1.0;
          //  Matrix是一个double，24维队列，每8个存放对应位置数字
            pset2(0, j) = match.p2[0];
            pset2(1, j) = match.p2[1];
            pset2(2, j) = 1.0;
        }

        /*2.0 8点法估计相机基础矩阵*/
        FundamentalMatrix F;//3x3矩阵
        calc_fundamental_8_point(pset1, pset2,F);

        /*3.0 统计所有的内点个数*///传入所有匹配点组向量，刚随机的基础矩阵，内点门限
        std::vector<int> inlier_indices = find_inliers(corr_all, F, inlier_thresh);

        if(inlier_indices.size()> best_inliers.size()){

//            std::cout << "RANSAC-F: Iteration " << i
//                      << ", inliers " << inlier_indices.size() << " ("
//                      << (100.0 * inlier_indices.size() / corr_all.size())
//                      << "%)" << std::endl;
            best_inliers.swap(inlier_indices);
        }
    }

    sfm::Correspondences2D2D corr_f;
    for(int i=0; i< best_inliers.size(); i++){
        corr_f.push_back(corr_all[best_inliers[i]]);//访问并存储所有内点索引
    }

    /*利用所有的内点进行最小二乘估计 目标函数：min(|Af|'*|Af|)，类似于Ax=0[A是多项式待求解，多个x已知]*/
    FundamentalMatrix F;
    calc_fundamental_least_squares(corr_f, F);//最小二乘拟合，类似八点法，区别在于A矩阵采用点数更多

    std::cout<<"inlier number: "<< best_inliers.size()<<std::endl;
    std::cout<<"F\n: "<< F<<std::endl;

    std::cout<<"result should be: \n"
             <<"inliner number: 272\n"
             <<"F: \n"
             <<"-0.00961384 -0.0309071 0.703297\n"
             <<"0.0448265 -0.00158655 -0.0555796\n"
             <<"-0.703477 0.0648517 -0.0117791\n";

    return 0;
}