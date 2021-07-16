/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

#include "util/system.h"
#include "math/algo.h"
#include "sfm/ransac_fundamental.h"

SFM_NAMESPACE_BEGIN

RansacFundamental::RansacFundamental (Options const& options)
    : opts(options)
{
}

void
RansacFundamental::estimate (Correspondences2D2D const& matches, Result* result)
{
    if (this->opts.verbose_output)
    {
        std::cout << "RANSAC-F: Running for " << this->opts.max_iterations
            << " iterations, threshold " << this->opts.threshold
            << "..." << std::endl;
    }

    std::vector<int> inliers;
    inliers.reserve(matches.size());//内点向量size按全部都是匹配点分配
    for (int iteration = 0; iteration < this->opts.max_iterations; ++iteration)
    {
        FundamentalMatrix fundamental;
        this->estimate_8_point(matches, &fundamental);//重构后的基础矩阵

        this->find_inliers(matches, fundamental, &inliers);
        if (inliers.size() > result->inliers.size())//找到的内点数>现在的内点数
        {
            if (this->opts.verbose_output)
            {
                std::cout << "RANSAC-F: Iteration " << iteration
                    << ", inliers " << inliers.size() << " ("
                    << (100.0 * inliers.size() / matches.size())
                    << "%)" << std::endl;
            }

            result->fundamental = fundamental;
            //替换基础矩阵，直到内点的size是最大时，保留此时的基础矩阵

            std::swap(result->inliers, inliers);//替换内点
            inliers.reserve(matches.size());//存放内点向量清空
        }
    }
}

void
RansacFundamental::estimate_8_point (Correspondences2D2D const& matches,
    FundamentalMatrix* fundamental)
{
    if (matches.size() < 8)
        throw std::invalid_argument("At least 8 matches required");

    /*
     * Draw 8 random numbers in the interval [0, matches.size() - 1]
     * without duplicates. This is done by keeping a set with drawn numbers.
     */
    std::set<int> result;
    while (result.size() < 8)
        result.insert(util::system::rand_int() % matches.size());//result插入随机匹配点组索引

    math::Matrix<double, 3, 8> pset1, pset2;
    std::set<int>::const_iterator iter = result.begin();
    for (int i = 0; i < 8; ++i, ++iter)//八组匹配点的成像坐标位置（u，v，1）
    {
        Correspondence2D2D const& match = matches[*iter];
        pset1(0, i) = match.p1[0];
        pset1(1, i) = match.p1[1];
        pset1(2, i) = 1.0;
        pset2(0, i) = match.p2[0];
        pset2(1, i) = match.p2[1];
        pset2(2, i) = 1.0;
    }
    /* Compute fundamental matrix using normalized 8-point. */
    sfm::fundamental_8_point(pset1, pset2, fundamental);

    sfm::enforce_fundamental_constraints(fundamental);//重构满足存在最小特征值为0的基础矩阵
}

void
RansacFundamental::find_inliers (Correspondences2D2D const& matches,
    FundamentalMatrix const& fundamental, std::vector<int>* result)
{
    result->resize(0);
    double const squared_thres = this->opts.threshold * this->opts.threshold;
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        double error = sampson_distance(fundamental, matches[i]);//两匹配点的桑普森距离满足阈值才看做内点
        if (error < squared_thres)
            result->push_back(i);
    }
}

SFM_NAMESPACE_END
