/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "util/exception.h"
#include "util/timer.h"
#include "features/sift.h"
#include "sfm/ransac.h"
#include "sfm/bundler_matching.h"
#include "features/cascade_hashing.h"
#include "features/exhaustive_matching.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <features/cascade_hashing.h>


SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

Matching::Matching (Options const& options, Progress* progress)
    : opts(options)
    , progress(progress)
{
    switch (this->opts.matcher_type)
    {
        case MATCHER_EXHAUSTIVE:
            this->matcher.reset(new features::ExhaustiveMatching());
            break;
        case MATCHER_CASCADE_HASHING:
            this->matcher.reset(new features::CascadeHashing());
            break;
        default:
            throw std::runtime_error("Unhandled matcher type");
    }
}

void
Matching::init (ViewportList* viewports)
{
    if (viewports == nullptr)
        throw std::invalid_argument("Viewports must not be null");

    this->viewports = viewports;
    this->matcher->init(viewports);//matcher类下初始化存放所有匹配点描述子

    /* Free descriptors. */
    for (std::size_t i = 0; i < viewports->size(); i++)
        viewports->at(i).features.clear_descriptors();
}

void
Matching::compute (PairwiseMatching* pairwise_matching)
{//匹配点计算：遍历所有两两视角可能，描述子匹配得到一组视角下两视角的特征点匹配索引，ransac+桑普森距离筛选内点，保留内点匹配点
    if (this->viewports == nullptr)
        throw std::runtime_error("Viewports must not be null");

    // 视角的个数
    std::size_t num_viewports = this->viewports->size();
    std::size_t num_pairs = num_viewports * (num_viewports - 1) / 2;//可两两匹配视角数
    std::size_t num_done = 0;

    if (this->progress != nullptr)
    {
        this->progress->num_total = num_pairs;
        this->progress->num_done = 0;
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < num_pairs; ++i)
    {
#pragma omp critical
        {
            num_done += 1;
            if (this->progress != nullptr)
                this->progress->num_done += 1;

            float percent = (num_done * 1000 / num_pairs) / 10.0f;
            std::cout << "\rMatching pair " << num_done << " of "
                << num_pairs << " (" << percent << "%)..." << std::flush;
        }

        int const view_1_id = (int)(0.5 + std::sqrt(0.25 + 2.0 * i));//1,2,2;上下两组视角对应关系
        int const view_2_id = (int)i - view_1_id * (view_1_id - 1) / 2;//0,0,1
        if (this->opts.match_num_previous_frames != 0
            && view_2_id + this->opts.match_num_previous_frames < view_1_id)
            continue;

        // 遍历两个视角
        FeatureSet const& view_1 = this->viewports->at(view_1_id).features;//获得匹配的视角的特征点位置、颜色信息
        FeatureSet const& view_2 = this->viewports->at(view_2_id).features;
        if (view_1.positions.empty() || view_2.positions.empty())
            continue;

       // 两个视角之间进行匹配
        util::WallTimer timer;
        std::stringstream message;
        CorrespondenceIndices matches;//结构体向量（存放两个视角下匹配的特征点匹配点索引pair）
        this->two_view_matching(view_1_id, view_2_id, &matches, message);//传入视角索引，根据匹配点找内点两特征点归一化位置matches向量存放匹配点索引匹配对，
        std::size_t matching_time = timer.get_elapsed();

        if (matches.empty())
        {
#pragma omp critical
            std::cout << "\rPair (" << view_1_id << ","
                << view_2_id << ") rejected, "
                << message.str() << std::endl;
            continue;
        }

        /* Successful two view matching. Add the pair.TwoViewMatching 结构体包括两视角索引，两视角下特征点索引*/
        TwoViewMatching matching;//结构体：两个视角idview_1_id，视角下对应匹配点索引matchers向量
        matching.view_1_id = view_1_id;
        matching.view_2_id = view_2_id;
        std::swap(matching.matches, matches);//将matches向量内存储的两个视角下特征点索引pair交给matches

#pragma omp critical
        {
            pairwise_matching->push_back(matching);
            std::cout << "\rPair (" << view_1_id << ","
                << view_2_id << ") matched, " << matching.matches.size()
                << " inliers, took " << matching_time << " ms." << std::endl;
        }
    }

    std::cout << "\rFound a total of " << pairwise_matching->size()
        << " matching image pairs." << std::endl;
}

void
Matching::two_view_matching (int view_1_id, int view_2_id,
    CorrespondenceIndices* matches, std::stringstream& message)
{//两视角根据该类viewports下特征信息，两视角分别向对方匹配描述子向量，去除匹配不一致点，8点ransac基础矩阵用桑普森矩阵获得最多的内点放在matches内
    FeatureSet const& view_1 = this->viewports->at(view_1_id).features;
    FeatureSet const& view_2 = this->viewports->at(view_2_id).features;
    /* Low-res matching if number of features is large. */
    if (this->opts.use_lowres_matching
        && view_1.positions.size() * view_2.positions.size() > 1000000)
    {
        int const num_matches = this->matcher->pairwise_match_lowres(view_1_id,
            view_2_id, this->opts.num_lowres_features);
        if (num_matches < this->opts.min_lowres_matches)
        {
            message << "only " << num_matches
                << " of " << this->opts.min_lowres_matches
                << " low-res matches.";
            return;
        }
    }

    /* Perform two-view descriptor matching. */
    sfm::Matching::Result matching_result;//存放，matches_1_2与matches_2_1两向量，向量索引对应的value是另一个向量的索引
    this->matcher->pairwise_match(view_1_id, view_2_id, &matching_result);//matcher类指针访问类下pairwise_match
    int num_matches = sfm::Matching::count_consistent_matches(matching_result);//matching_result有正有负，负代表特征点未匹配到，统计匹配成功

    /* Require at least 8 matches. Check threshold. */
    int const min_matches_thres = std::max(8, this->opts.min_feature_matches);//24
    if (num_matches < min_matches_thres)
    {
        message << "matches below threshold of "
            << min_matches_thres << ".";
        return;
    }

    /* Build correspondences from feature matching result. */
    sfm::Correspondences2D2D unfiltered_matches;//匹配点坐标2
    sfm::CorrespondenceIndices unfiltered_indices;//匹配点索引对pair
    {
        std::vector<int> const& m12 = matching_result.matches_1_2;
        for (std::size_t i = 0; i < m12.size(); ++i)
        {
            if (m12[i] < 0)//没有匹配到的特征点跳过
                continue;

            sfm::Correspondence2D2D match;//结构体，内部两个位置数组2
            match.p1[0] = view_1.positions[i][0];//match传入有匹配点的特征点位置
            match.p1[1] = view_1.positions[i][1];
            match.p2[0] = view_2.positions[m12[i]][0];
            match.p2[1] = view_2.positions[m12[i]][1];
            unfiltered_matches.push_back(match);//未过滤匹配点向量填充匹配点索引组
            unfiltered_indices.push_back(std::make_pair(i, m12[i]));
        }
    }

    /* Compute fundamental matrix using RANSAC. */
    sfm::RansacFundamental::Result ransac_result;//结构体内含基础矩阵3x3，认为是内点的匹配点pair向量索引
    int num_inliers = 0;
    {
        sfm::RansacFundamental ransac(this->opts.ransac_opts);
        ransac.estimate(unfiltered_matches, &ransac_result);
        num_inliers = ransac_result.inliers.size();
    }

    /* Require at least 8 inlier matches. */
    int const min_inlier_thres = std::max(8, this->opts.min_matching_inliers);
    if (num_inliers < min_inlier_thres)
    {
        message << "inliers below threshold of "
            << min_inlier_thres << ".";
        return;
    }

    /* Create Two-View matching result. */
    matches->clear();
    matches->reserve(num_inliers);
    for (int i = 0; i < num_inliers; ++i)
    {
        int const inlier_id = ransac_result.inliers[i];//是内点的匹配点的索引
        matches->push_back(unfiltered_indices[inlier_id]);//matches向量存入两幅图特征点索引pair
    }
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
