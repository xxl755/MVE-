/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <set>
#include <atomic>
#include <iostream>

#include "util/system.h"
#include "math/matrix_tools.h"
#include "sfm/ransac_pose_p3p.h"
#include "sfm/pose_p3p.h"

SFM_NAMESPACE_BEGIN

RansacPoseP3P::RansacPoseP3P (Options const& options)
    : opts(options)
{
}

void
RansacPoseP3P::estimate (Correspondences2D3D const& corresp,
    math::Matrix<double, 3, 3> const& k_matrix, Result* result) const
{//ransac方法利用匹配二维点与三维点计算相机姿态（四组），正确姿态下三维点重投影误差小，同时找到最多内点的姿态，获得内点与姿态
    if (this->opts.verbose_output)
    {
        std::cout << "RANSAC-3: Running for " << this->opts.max_iterations
            << " iterations, threshold " << this->opts.threshold
            << "..." << std::endl;
    }

    /* Pre-compute inverse K matrix to compute directions from corresp. */
    math::Matrix<double, 3, 3> inv_k_matrix = math::matrix_inverse(k_matrix);
    std::atomic<int> num_iterations;

#pragma omp parallel
    {
        std::vector<int> inliers;
        inliers.reserve(corresp.size());
#pragma omp for
        for (int i = 0; i < this->opts.max_iterations; ++i)//ransac重复迭代取随机三组点
        {
            int iteration = i;
            if (this->opts.verbose_output)
                iteration = num_iterations++;

            /* Compute up to four poses [R|t] using P3P algorithm. */
            PutativePoses poses;
            this->compute_p3p(corresp, inv_k_matrix, &poses);//corresp内存放三维点对应二维点信息，初始相机内参矩阵逆

            /* Check all putative solutions and count inliers. */
            for (std::size_t j = 0; j < poses.size(); ++j)
            {//每次p3p遍历四组姿态矩阵，用匹配的三维点与二维点计算重投影误差，正确姿态能找到最多的内点；迭代会找到最好正确姿态
                this->find_inliers(corresp, k_matrix, poses[j], &inliers);
#pragma omp critical
                if (inliers.size() > result->inliers.size())
                {
                    result->pose = poses[j];
                    std::swap(result->inliers, inliers);
                    inliers.reserve(corresp.size());

                    if (this->opts.verbose_output)
                    {
                        std::cout << "RANSAC-3: Iteration " << iteration
                            << ", inliers " << result->inliers.size() << " ("
                            << (100.0 * result->inliers.size() / corresp.size())
                            << "%)" << std::endl;
                    }
                }
            }
        }
    }
}

void
RansacPoseP3P::compute_p3p (Correspondences2D3D const& corresp,
    math::Matrix<double, 3, 3> const& inv_k_matrix,
    PutativePoses* poses) const
{//已知三维点和二维点，利用投影矩阵采用ransac线性变换，u=TX,需要至少3组点(文中方法6组点)解T进行解得T=【KR|Kt】，对KR来QR分解上三角K，正交矩阵R
    if (corresp.size() < 3)
        throw std::invalid_argument("At least 3 correspondences required");

    /* Draw 3 unique random numbers. */
    std::set<int> result;
    while (result.size() < 3)
        result.insert(util::system::rand_int() % corresp.size());

    std::set<int>::const_iterator iter = result.begin();
    Correspondence2D3D const& c1(corresp[*iter++]);//索引是赋值后迭代器++后解引用
    Correspondence2D3D const& c2(corresp[*iter++]);
    Correspondence2D3D const& c3(corresp[*iter]);
    pose_p3p_kneip(
        math::Vec3d(c1.p3d), math::Vec3d(c2.p3d), math::Vec3d(c3.p3d),
        inv_k_matrix.mult(math::Vec3d(c1.p2d[0], c1.p2d[1], 1.0)),
        inv_k_matrix.mult(math::Vec3d(c2.p2d[0], c2.p2d[1], 1.0)),
        inv_k_matrix.mult(math::Vec3d(c3.p2d[0], c3.p2d[1], 1.0)),
        poses);//将K^-1 乘以uv这样解得结果QR分解得到R,t
}

void
RansacPoseP3P::find_inliers (Correspondences2D3D const& corresp,
    math::Matrix<double, 3, 3> const& k_matrix,
    Pose const& pose, std::vector<int>* inliers) const
{
    inliers->resize(0);
    double const square_threshold = MATH_POW2(this->opts.threshold);//0.005^2
    for (std::size_t i = 0; i < corresp.size(); ++i)
    {
        Correspondence2D3D const& c = corresp[i];
        math::Vec4d p3d(c.p3d[0], c.p3d[1], c.p3d[2], 1.0);
        math::Vec3d p2d = k_matrix * (pose * p3d);//重投影点
        double square_error = MATH_POW2(p2d[0] / p2d[2] - c.p2d[0])
            + MATH_POW2(p2d[1] / p2d[2] - c.p2d[1]);//重投影误差
        if (square_error < square_threshold)
            inliers->push_back(i);
    }
}

SFM_NAMESPACE_END
