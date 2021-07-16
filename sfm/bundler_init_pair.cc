/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <random>

#include "sfm/ransac_homography.h"
#include "sfm/triangulate.h"
#include "sfm/bundler_init_pair.h"

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void
InitialPair::compute_pair (Result* result)
{//将现有track的特征点视角及索引与viewports的features下位置信息转为candidates类的一组视角对应所有特征点组的结构；
    // candidates中匹配点组最多视角在前，四个标准评价：1.匹配点组数量足够多；
    // 2.ransac4点得到的单应矩阵判断两视角不是旋转关系，满足单应矩阵内点不宜过多；
    // 3.匹配点组确定基础矩阵，本征矩阵以及本征矩阵UV结果构建相机姿态，利用方法二确定正确相对姿态，取各组匹配点与三维点夹角中值作为夹角满足一定大小
    //4.为以上条件综合打分；利用匹配点组与相机姿态重建三维点，三维点与两相机构成向量夹角要足够大，重投影相机坐标系z轴应为正，重投影误差不应过大
    if (this->viewports == nullptr || this->tracks == nullptr)
        throw std::invalid_argument("Null viewports or tracks");

    std::cout << "Searching for initial pair..." << std::endl;
    result->view_1_id = -1;
    result->view_2_id = -1;

    // 根据匹配的点的个数，找到候选的匹配对
    /* Convert tracks to pairwise information. */
    std::vector<CandidatePair> candidates;//CandidatePair结构体：两视角索引，匹配点向量（匹配点结构体（u0v0，u1v1））
    this->compute_candidate_pairs(&candidates);//候选对计算，统计每个相机对间匹配点位置
    /* Sort the candidate pairs by number of matches. */
    std::sort(candidates.rbegin(), candidates.rend());//根据店个数向量排序，匹配点最多的视角在前
//candidates等同于之前PairwiseMatching信息，有视角与视角下所有匹配对索引，但相较多了特征点位置信息
    /*
     * Search for a good initial pair and return the first pair that
     * satisfies all thresholds (min matches, max homography inliers,
     * min triangulation angle). If no pair satisfies all thresholds, the
     * pair with the best score is returned.
     */
    bool found_pair = false;
    std::size_t found_pair_id = std::numeric_limits<std::size_t>::max();
    std::vector<float> pair_scores(candidates.size(), 0.0f);//根据评分选择匹配对
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < candidates.size(); ++i)//从匹配点最多的开始遍历，满足所有条件的图像自然是匹配对最多的一组图，作为初始值
    {
        if (found_pair)//找到一组满足标准视角足以
             continue;

        //标准1： 匹配点的个数大50对
        /* Reject pairs with 8 or fewer matches. */
        CandidatePair const& candidate = candidates[i];//获得视角与track匹配点组坐标信息
        std::size_t num_matches = candidate.matches.size();
        if (num_matches < static_cast<std::size_t>(this->opts.min_num_matches))//匹配点组数量太少
        {
            this->debug_output(candidate);
            continue;
        }

        //标准2： 单应矩阵矩阵的内点比例数过高有可能纯旋转要舍弃，ransac方法计算单应矩阵
        /* 随机4组匹配点，求单应矩阵；通过匹配点向另一个匹配点视角利用单应矩阵投影，计算均方差是否满足阈值判断是否内点*/
        std::size_t num_inliers = this->compute_homography_inliers(candidate);//一组视角内点最多单应矩阵下的内点数量
        float percentage = static_cast<float>(num_inliers) / num_matches;

        if (percentage > this->opts.max_homography_inliers)
        {
            this->debug_output(candidate, num_inliers);
            continue;
        }

        // 标准3：相机基线要足够长(用三角量测的夹角衡量），
        /* Compute initial pair pose. */
        CameraPose pose1, pose2;//计算本质矩阵分解的四个姿态，给定匹配对，使用所有视角下匹配点重构基础矩阵，给焦距得本征矩阵与四组外参，三角化与方法2找到外参
        bool const found_pose = this->compute_pose(candidate, &pose1, &pose2);
        //pose1是只有焦距内参，外参是世界坐标系重合；pose2相机内参只有焦距，
        // 外参：利用全部内点组最小二乘并重构特征值得到基础矩阵F，利用两个内参K得到本征矩阵，svd分解得到UV用于构建相机相对姿态
        //并用方法二判断相机正确姿态
        if (!found_pose) {
            this->debug_output(candidate, num_inliers);
            continue;
        }
        /* Rejects pairs with bad triangulation angle.
         * 所有匹配对三角量测， 所有匹配对三角化，计算所有的匹配对构成射线夹角，计算夹角中值，中值小于5°舍去，基线不够长*/
        //匹配点坐标px对应相机投影矩阵的逆得到PP1与PP2，利用归一化向量乘积得到cosΘ，通过限制两段都比中间小与大，再限制中间值范围；
        double const angle = this->angle_for_pose(candidate, pose1, pose2);//以中值求反余弦为两姿态角度
        //利用匹配点数，单应矩阵内点数量与三角量测得到的两姿态角度综合评分
        pair_scores[i] = this->score_for_pair(candidate, num_inliers, angle);
        this->debug_output(candidate, num_inliers, angle);
        if (angle < this->opts.min_triangulation_angle)
            continue;
        // 标准4：成功的三角量测的个数>50%，所有匹配点三角量测，统计匹配点组三角量测成功数（三角量测得到的三维点重投影到原平面，计算投影误差）
        /* Run triangulation to ensure correct pair */
        Triangulate::Options triangulate_opts;
        Triangulate triangulator(triangulate_opts);
        std::vector<CameraPose const*> poses;
        poses.push_back(&pose1);
        poses.push_back(&pose2);
        std::size_t successful_triangulations = 0;
        std::vector<math::Vec2f> positions(2);//两元素向量，存放两个位置
        Triangulate::Statistics stats;
        for (std::size_t j = 0; j < candidate.matches.size(); ++j) {
            positions[0] = math::Vec2f(candidate.matches[j].p1);
            positions[1] = math::Vec2f(candidate.matches[j].p2);
            math::Vec3d pos3d;
            if (triangulator.triangulate(poses, positions, &pos3d, &stats))
                //按个传入匹配点组，直接线性变换建立A矩阵并分解确定三维点坐标；
                // 归一化三维点至相机的向量，两向量夹角应足够大；
                // 相机坐标系下三维点坐标z轴应大于0，且重投影误差有最大值限制；
                successful_triangulations += 1;
        }
        if (successful_triangulations * 2 < candidate.matches.size())
            continue;


#pragma omp critical
        if (i < found_pair_id)//满足四个限制的相机对作为结果记录，记录相机视角索引与相机内外参；选择最先出现满足的结果
        {
            result->view_1_id = candidate.view_1_id;
            result->view_2_id = candidate.view_2_id;
            result->view_1_pose = pose1;
            result->view_2_pose = pose2;
            found_pair_id = i;
            found_pair = true;
        }
    }

    /* Return if a pair satisfying all thresholds has been found. */
    if (found_pair)
        return;
//没有一组图片同时满足以上四个条件，得到最好的一组相机组
    /* Return pair with best score (larger than 0.0). */
    std::cout << "Searching for pair with best score..." << std::endl;
    float best_score = 0.0f;
    std::size_t best_pair_id = 0;
    for (std::size_t i = 0; i < pair_scores.size(); ++i)
    {
        if (pair_scores[i] <= best_score)
            continue;

        best_score = pair_scores[i];
        best_pair_id = i;
    }

    /* Recompute pose for pair with best score. */
    if (best_score > 0.0f)//初始两视角至少评分不为0，否则result视角索引保持为-1
    {
        result->view_1_id = candidates[best_pair_id].view_1_id;
        result->view_2_id = candidates[best_pair_id].view_2_id;
        this->compute_pose(candidates[best_pair_id],
            &result->view_1_pose, &result->view_2_pose);
    }
}

void
InitialPair::compute_pair (int view_1_id, int view_2_id, Result* result)
{
    if (view_1_id > view_2_id)
        std::swap(view_1_id, view_2_id);

    /* Convert tracks to pairwise information. */
    std::vector<CandidatePair> candidates;
    this->compute_candidate_pairs(&candidates);

    /* Find candidate pair. */
    CandidatePair* candidate = nullptr;
    for (std::size_t i = 0; candidate == nullptr && i < candidates.size(); ++i)
    {
        if (view_1_id == candidates[i].view_1_id
            && view_2_id == candidates[i].view_2_id)
            candidate = &candidates[i];
    }
    if (candidate == nullptr)
        throw std::runtime_error("No matches for initial pair");

    /* Compute initial pair pose. */
    result->view_1_id = view_1_id;
    result->view_2_id = view_2_id;
    bool const found_pose = this->compute_pose(*candidate,
        &result->view_1_pose, &result->view_2_pose);
    if (!found_pose)
        throw std::runtime_error("Cannot compute pose for initial pair");
}

void
InitialPair::compute_candidate_pairs (CandidatePairs* candidates)
{//利用tracks下feature视角与特征点索引信息；建立两两视角匹配对向量candidate_lookup用于统计两视角组，默认为-1（图片无联系）
    // 根据tracks两视角信息，并用作两视角candidate_lookup索引，对应匹配对向量value为默认-1时创建candidates传入视角索引
    //改变candidate_lookup的value使其不为-1等于candidates.size，说明两视角存在联系；向candidates对应视角（pair_id）下传入特征点
    /*
     * Convert the tracks to pairwise information. This is similar to using
     * the pairwise matching result directly, however, the tracks have been
     * further filtered during track generation.
     */
    int const num_viewports = static_cast<int>(this->viewports->size());
    std::vector<int> candidate_lookup(MATH_POW2(num_viewports), -1);//图片数^2个-1向量，用于统计每两两视角下track总数
    candidates->reserve(1000);//candidates统计一组视角下所有的track的特征点组坐标
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        for (std::size_t j = 1; j < track.features.size(); ++j)//遍历track下所有特征
            for (std::size_t k = 0; k < j; ++k) {//保证两视角不相同

                int v1id = track.features[j].view_id;//获得一个Track所有信息，并保证小视角索引在前
                int v2id = track.features[k].view_id;
                int f1id = track.features[j].feature_id;
                int f2id = track.features[k].feature_id;
                if (v1id > v2id) {
                    std::swap(v1id, v2id);
                    std::swap(f1id, f2id);
                }

                /* Lookup pair. *///pair_id用于确定两视角间联系，candidate_lookup统计两两匹配图片组
                int const lookup_id = v1id * num_viewports + v2id;//每一组视角，0索引视角至下一个1视角相差一组num_viewports；+ v2id是偏移，表示两视角对应
                int pair_id = candidate_lookup[lookup_id];
                if (pair_id == -1)
                {//创建两两匹配图片组统计信息给candidate_lookup，并创建candidates传入两匹配图片视角索引
                    pair_id = static_cast<int>(candidates->size());//更新pair统计信息
                    candidate_lookup[lookup_id] = pair_id;
                    candidates->push_back(CandidatePair());//CandidatePair结构体：两视角索引与匹配两点位置（u，v）
                    candidates->back().view_1_id = v1id;
                    candidates->back().view_2_id = v2id;
                }

                /* Fill candidate with additional info. */
                Viewport const& view1 = this->viewports->at(v1id);//利用视角下特征点位置信息
                Viewport const& view2 = this->viewports->at(v2id);
                math::Vec2f const v1pos = view1.features.positions[f1id];
                math::Vec2f const v2pos = view2.features.positions[f2id];
                Correspondence2D2D match;
                std::copy(v1pos.begin(), v1pos.end(), match.p1);
                std::copy(v2pos.begin(), v2pos.end(), match.p2);
                candidates->at(pair_id).matches.push_back(match);
            }
    }
}

std::size_t
InitialPair::compute_homography_inliers (CandidatePair const& candidate)
{//随机4组匹配点，求单应矩阵；通过匹配点向另一个匹配点视角利用单应矩阵的逆投影，计算两点间距均方差是否满足阈值判断是否内点
    /* Execute homography RANSAC. */
    RansacHomography::Result ransac_result;
    RansacHomography homography_ransac(this->opts.homography_opts);
    homography_ransac.estimate(candidate.matches, &ransac_result);
    return ransac_result.inliers.size();//返回一组视角下存在的最多内点数
}

bool
InitialPair::compute_pose (CandidatePair const& candidate,
    CameraPose* pose1, CameraPose* pose2)
{//利用匹配点计算最小二乘下的基础矩阵F；利用相机视角得到焦距初始化相机1内外参数与相机2内参，综合得到本证矩阵，svd分解E得到矩阵元素得到相机四个姿态；
    // 方法二判定相机姿态
    /* Compute fundamental matrix from pair correspondences. */
    FundamentalMatrix fundamental;
    {
        Correspondences2D2D matches = candidate.matches;
        if (matches.size() > 1000ul) {
            std::mt19937 g;
            std::shuffle(matches.begin(), matches.end(), g);
            matches.resize(1000ul);
        }
        fundamental_least_squares(matches, &fundamental);//svd解得基础矩阵F最初版本，未保证第三个特征值为0
        enforce_fundamental_constraints(&fundamental);//基础矩阵重构
    }

    /* Populate K-matrices. */
    Viewport const& view_1 = this->viewports->at(candidate.view_1_id);
    Viewport const& view_2 = this->viewports->at(candidate.view_2_id);
    pose1->set_k_matrix(view_1.focal_length, 0.0, 0.0);//利用viewports信息给出相机初始内参矩阵K
    pose1->init_canonical_form();//相机1初始化外参R单位矩阵，t为0矩阵
    pose2->set_k_matrix(view_2.focal_length, 0.0, 0.0);

    /* Compute essential matrix from fundamental matrix (HZ (9.12)). */
    EssentialMatrix E = pose2->K.transposed() * fundamental * pose1->K;//E=K2^T *F*K1

    /* Compute pose from essential. */
    std::vector<CameraPose> poses;//CameraPose类，属性K R T
    pose_from_essential(E, &poses);//利用本质矩阵E的svd分解得到UV矩阵，得到四组相机姿态

    /* Find the correct pose using point test (HZ Fig. 9.12). */
    bool found_pose = false;
    for (std::size_t i = 0; i < poses.size(); ++i)//遍历四个相对姿态
    {
        poses[i].K = pose2->K;//调整相机姿态的内参矩阵
        if (is_consistent_pose(candidate.matches[0], *pose1, poses[i]))//利用三角化得到三维点计算方法2选定姿态
        {
            *pose2 = poses[i];
            found_pose = true;
            break;
        }
    }
    return found_pose;
}

double
InitialPair::angle_for_pose (CandidatePair const& candidate,
    CameraPose const& pose1, CameraPose const& pose2)
{
    /* Compute transformation from image coordinates to viewing direction. */
    math::Matrix3d T1 = pose1.R.transposed() * math::matrix_inverse(pose1.K);//旋转矩阵的逆=旋转矩阵转置
    math::Matrix3d T2 = pose2.R.transposed() * math::matrix_inverse(pose2.K);//x=K(R|T)Xw->Xw=K^-1 *(R|T)^-1 *x

    /* Compute triangulation angle for each correspondence. */
    std::vector<double> cos_angles;
    cos_angles.reserve(candidate.matches.size());
    for (std::size_t i = 0; i < candidate.matches.size(); ++i)
    {
        Correspondence2D2D const& match = candidate.matches[i];//遍历一组视角所有匹配点
        math::Vec3d p1(match.p1[0], match.p1[1], 1.0);
        p1 = T1.mult(p1).normalized();//向量PP1，去除平移t影响，直接对xc乘以KR的逆
        math::Vec3d p2(match.p2[0], match.p2[1], 1.0);
        p2 = T2.mult(p2).normalized();
        cos_angles.push_back(p1.dot(p2));//归一化后两向量相乘等于cosΘ
    }

    /* Return 50% median. */
    std::size_t median_index = cos_angles.size() / 2;
    std::nth_element(cos_angles.begin(),
        cos_angles.begin() + median_index, cos_angles.end());//以中间为界，前面比中间小，后面比中间大；两端内没有顺序
    double const cos_angle = math::clamp(cos_angles[median_index], -1.0, 1.0);//限制中间值，并取出
    return std::acos(cos_angle);//返回一个角度值
}

float
InitialPair::score_for_pair (CandidatePair const& candidate,
    std::size_t num_inliers, double angle)//考虑匹配点个数，单应矩阵内点个数，角度
{
    float const matches = static_cast<float>(candidate.matches.size());
    float const inliers = static_cast<float>(num_inliers) / matches;
    float const angle_d = MATH_RAD2DEG(angle);

    /* Score for matches (min: 20, good: 200). */
    float f1 = 2.0 / (1.0 + std::exp((20.0 - matches) * 6.0 / 200.0)) - 1.0;
    /* Score for angle (min 1 degree, good 8 degree). */
    float f2 = 2.0 / (1.0 + std::exp((1.0 - angle_d) * 6.0 / 8.0)) - 1.0;
    /* Score for H-Inliers (max 70%, good 40%). */
    float f3 = 2.0 / (1.0 + std::exp((inliers - 0.7) * 6.0 / 0.4)) - 1.0;

    f1 = math::clamp(f1, 0.0f, 1.0f);
    f2 = math::clamp(f2, 0.0f, 1.0f);
    f3 = math::clamp(f3, 0.0f, 1.0f);
    return f1 * f2 * f3;
}//得到匹配图片组的分数，按照分数排序，没有同时满足四个初始条件的匹配组就选分数最高

void
InitialPair::debug_output (CandidatePair const& candidate,
    std::size_t num_inliers, double angle)
{
    if (!this->opts.verbose_output)
        return;

    std::stringstream message;
    std::size_t num_matches = candidate.matches.size();
    message << "  Pair " << std::setw(3) << candidate.view_1_id
        << "," << std::setw(3) << candidate.view_2_id
        << ": " << std::setw(4) << num_matches << " matches";

    if (num_inliers > 0)
    {
        float percentage = static_cast<float>(num_inliers) / num_matches;
        message << ", " << std::setw(4) << num_inliers
            << " H-inliers (" << (int)(100.0f * percentage) << "%)";
    }

    if (angle > 0.0)
    {
        message << ", " << std::setw(5)
            << util::string::get_fixed(MATH_RAD2DEG(angle), 2)
            << " pair angle";
    }

#pragma omp critical
    std::cout << message.str() << std::endl;
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END

