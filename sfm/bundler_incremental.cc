/*
 * Copyright (C) 2015, Simon Fuhrmann, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <limits>
#include <iostream>
#include <utility>

#include "util/timer.h"
#include "math/transform.h"
#include "math/matrix_svd.h"
#include "math/matrix_tools.h"
#include "sfm/triangulate.h"
#include "sfm/bundle_adjustment.h"
#include "sfm/bundler_incremental.h"

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void
Incremental::initialize (ViewportList* viewports, TrackList* tracks,
    SurveyPointList* survey_points)
{
    this->viewports = viewports;
    this->tracks = tracks;
    this->survey_points = survey_points;

    if (this->viewports->empty())
        throw std::invalid_argument("No viewports given");

    /* Check if at least two cameras are initialized. */
    std::size_t num_valid_cameras = 0;
    for (std::size_t i = 0; i < this->viewports->size(); ++i)
        if (this->viewports->at(i).pose.is_valid())
            num_valid_cameras += 1;
    if (num_valid_cameras < 2)
        throw std::invalid_argument("Two or more valid cameras required");

    /* Set track positions to invalid state. */
    for (std::size_t i = 0; i < tracks->size(); ++i)
    {
        Track& track = tracks->at(i);
        track.invalidate();
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::find_next_views (std::vector<int>* next_views)
{/* Create mapping from valid tracks to view ID. */
    std::vector<std::pair<int, int> > valid_tracks(this->viewports->size());
    for (std::size_t i = 0; i < valid_tracks.size(); ++i)
        valid_tracks[i] = std::make_pair(0, static_cast<int>(i));
// make_pair匹配对second视角索引，first统计视角匹配到已建立三维点的track次数(该相机姿态未矫正)，最初只有初始对track三维点重建
    for (std::size_t i = 0; i < this->tracks->size(); ++i)//遍历场景中的track
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())//非重建过的track跳过
            continue;

        for (std::size_t j = 0; j < track.features.size(); ++j)//遍历重建过track的特征
        {
            int const view_id = track.features[j].view_id;//视角
            if (this->viewports->at(view_id).pose.is_valid())//如果该track的相机姿态调整过，不考虑改变，跳过
                continue;
            valid_tracks[view_id].first += 1;//未调整过的相机姿态可见三维点+1
        }
    }
//比较未重建姿态的二维特征点数量选下一个相机姿态
    /* Sort descending by number of valid tracks. */
    std::sort(valid_tracks.rbegin(), valid_tracks.rend());//按照可见三维点数量排序，由大到小

    /* Return unreconstructed views with most valid tracks. */
    next_views->clear();
    for (std::size_t i = 0; i < valid_tracks.size(); ++i)
    {
        if (valid_tracks[i].first > 6)
            //选择未计算相机姿态的视角但在已建好的track上存在对应二维点6个以上的视角（已建好姿态的匹配对first为0）
            next_views->push_back(valid_tracks[i].second);
    }
}

/* ---------------------------------------------------------------- */
// 通过p3p重建新视角的相机姿态，并通过RANSAC去除错误的Tracks
bool
Incremental::reconstruct_next_view (int view_id)
{//传入新视角索引，构建已建成track的三维点与该视角下对应二维点的关系，利用匹配对与相机内参ransac的p3p判断内点与相机姿态；
    // 新视角下认为track内视角索引的二维点是外点的索引去除track三维点与视角下二维点的联系
    Viewport const& viewport = this->viewports->at(view_id);//KRT都为空，有焦距，track_ids,features
    FeatureSet const& features = viewport.features;//colors,positions

    /* Collect all 2D-3D correspondences. */
    Correspondences2D3D corr;
    std::vector<int> track_ids;
    std::vector<int> feature_ids;
    for (std::size_t i = 0; i < viewport.track_ids.size(); ++i)//通过track找到三维与二维对应点
    {
        int const track_id = viewport.track_ids[i];
        //新视角下track_ids索引可能突然变大，因为当时利用二维点建立的track没有同时前两个视角下建立成功，是相对于前两个视角新的track，此时还没有三维点
        if (track_id < 0 || !this->tracks->at(track_id).is_valid())//小于0代表该索引处特征点点没有构建track或存在track，但track没有建好三维点
            continue;
        math::Vec2f const& pos2d = features.positions[i];//获得建好三维点根据索引找到该视角下的二维点与三维点
        math::Vec3f const& pos3d = this->tracks->at(track_id).pos;

        corr.push_back(Correspondence2D3D());//Correspondence2D3D类存放三维点坐标与二维点坐标
        Correspondence2D3D& c = corr.back();
        std::copy(pos3d.begin(), pos3d.end(), c.p3d);
        std::copy(pos2d.begin(), pos2d.end(), c.p2d);
        track_ids.push_back(track_id);//track_ids传入建好三维点的track索引作为value；
        feature_ids.push_back(i);//传入有三维点的track在该视角下对应二维点索引作为value
    }

    if (this->opts.verbose_output)
    {
        std::cout << "Collected " << corr.size()
            << " 2D-3D correspondences." << std::endl;
    }

    /* Initialize a temporary camera. *///初始化新相机内参
    CameraPose temp_camera;
    temp_camera.set_k_matrix(viewport.focal_length, 0.0, 0.0);

    /* Compute pose from 2D-3D correspondences using P3P. */
    util::WallTimer timer;
    RansacPoseP3P::Result ransac_result;
    {//已知三维点和二维点，利用投影矩阵采用ransac线性变换，u=TX,需要至少3组点解T进行解得T=【KR|Kt】，对KR来QR分解上三角K，正交矩阵R
        RansacPoseP3P ransac(this->opts.pose_p3p_opts);//ransac p3p，每3个点计算姿态，统计内点个数
        ransac.estimate(corr, temp_camera.K, &ransac_result);//传入三维点与对应二维点；相机初始内参，随机三组点的ransac方法；
    }//ransac_result类包括内点最多的相机姿态，内点在corr中三维二维匹配对的索引；
    // 计算过程是ransac一定次数利用p3p方法得到四组相机姿态能重投影误差得到一定内点的是正确姿态，选择内点最多的姿态

    /* Cancel if inliers are below a 33% threshold. *///最多的内点数小于总匹配点组的1/3
    if (3 * ransac_result.inliers.size() < corr.size())
    {
        if (this->opts.verbose_output)
            std::cout << "Only " << ransac_result.inliers.size()
                << " 2D-3D correspondences inliers ("
                << (100 * ransac_result.inliers.size() / corr.size())
                << "%). Skipping view." << std::endl;
        return false;
    }
    else if (this->opts.verbose_output)
    {
        std::cout << "Selected " << ransac_result.inliers.size()
            << " 2D-3D correspondences inliers ("
            << (100 * ransac_result.inliers.size() / corr.size())
            << "%), took " << timer.get_elapsed() << "ms." << std::endl;
    }
    /*
     * Remove outliers from tracks and tracks from viewport.
     * TODO: Once single cam BA has been performed and parameters for this
     * camera are optimized, evaluate outlier tracks and try to restore them.
     */
    for (std::size_t i = 0; i < ransac_result.inliers.size(); ++i)//tracks 内点track_ids都是-1
        track_ids[ransac_result.inliers[i]] = -1;
    for (std::size_t i = 0; i < track_ids.size(); ++i)
    {
        if (track_ids[i] < 0)
            continue;       //将新视角下认为是外点的三维点，将该路径清理
        this->tracks->at(track_ids[i]).remove_view(view_id);//前面视角确定了track的三维点，track将新视角看做外点移除视角索引与对应特征点索引
        this->viewports->at(view_id).track_ids[feature_ids[i]] = -1;//track移除视角后，视角下track_ids【特征点对应track】对应二维特征点索引剔除设为-1
    }

    track_ids.clear();
    feature_ids.clear();

    /* Commit camera using known K and computed R and t. *///对该视角利用ransac方法得到的pose更新视角相机外参
    {
        CameraPose& pose = this->viewports->at(view_id).pose;
        pose = temp_camera;
        pose.R = ransac_result.pose.delete_col(3);
        pose.t = ransac_result.pose.col(3);

        if (this->opts.verbose_output)
        {
            std::cout << "Reconstructed camera "
                << view_id << " with focal length "
                << pose.get_focal_length() << std::endl;
        }
    }

    if (this->survey_points != nullptr && !registered)
        this->try_registration();

    return true;
}

/* ---------------------------------------------------------------- */

void
Incremental::try_registration () {
    std::vector<math::Vec3d> p0;
    std::vector<math::Vec3d> p1;

    for (std::size_t i = 0; i < this->survey_points->size(); ++i)
    {
        SurveyPoint const& survey_point = this->survey_points->at(i);

        std::vector<math::Vec2f> pos;
        std::vector<CameraPose const*> poses;
        for (std::size_t j = 0; j < survey_point.observations.size(); ++j)
        {
            SurveyObservation const& obs = survey_point.observations[j];
            int const view_id = obs.view_id;
            if (!this->viewports->at(view_id).pose.is_valid())
                continue;

            pos.push_back(obs.pos);
            poses.push_back(&this->viewports->at(view_id).pose);
        }

        if (pos.size() < 2)
            continue;

        p0.push_back(triangulate_track(pos, poses));
        p1.push_back(survey_point.pos);
    }

    if (p0.size() < 3)
        return;

    /* Determine transformation. */
    math::Matrix3d R;
    double s;
    math::Vec3d t;
    if (!math::determine_transform(p0, p1, &R, &s, &t))
        return;

    /* Transform every camera. */
    for (std::size_t i = 0; i < this->viewports->size(); ++i)
    {
        Viewport& view = this->viewports->at(i);
        CameraPose& pose = view.pose;
        if (!pose.is_valid())
            continue;

        pose.t = -pose.R * R.transposed() * t + pose.t * s;
        pose.R = pose.R * R.transposed();
    }

    /* Transform every point. */
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        track.pos = R * s * track.pos + t;
    }

    this->registered = true;
}

/* ---------------------------------------------------------------- */

void
Incremental::triangulate_new_tracks (int min_num_views)
{//创建新track，通过判断track向量的三维点信息是否有效；三角量测后满足重建三维点条件的点添加给Incremental的track的pos坐标
    Triangulate::Options triangulate_opts;
    triangulate_opts.error_threshold = this->opts.new_track_error_threshold;
    triangulate_opts.angle_threshold = this->opts.min_triangulation_angle;
    triangulate_opts.min_num_views = min_num_views;

    Triangulate::Statistics stats;
    Triangulate triangulator(triangulate_opts);

    std::size_t initial_tracks_size = this->tracks->size();
    for (std::size_t i = 0; i < this->tracks->size(); ++i)//所有track进行遍历
    {
        /* Skip tracks that have already been triangulated. */
        Track const& track = this->tracks->at(i);
        if (track.is_valid())//如果track重建过，跳过该track
            continue;

        /*
         * Triangulate the track using all cameras. There can be more than two
         * cameras if the track was rejected in previous triangulation attempts.
         */
        std::vector<math::Vec2f> pos;
        std::vector<CameraPose const*> poses;
        std::vector<std::size_t> view_ids;
        std::vector<std::size_t> feature_ids;
        //收集一条track的相机姿态，特征点坐标，分别提出一条track下不同特征值的信息交给相机姿态poses，特征点位置pos，视角索引view_ids，特征点索引
        for (std::size_t j = 0; j < track.features.size(); ++j)//遍历track所有的特征点features（视角，特征点索引，相机姿态）
        {
            int const view_id = track.features[j].view_id;
            if (!this->viewports->at(view_id).pose.is_valid())//要求相机姿态确定，最开始只有满足条件的初始对相机姿态有效
                continue;//不确定姿态的视角对匹配点跳过该track下features
            int const feature_id = track.features[j].feature_id;
            pos.push_back(this->viewports->at(view_id)
                .features.positions[feature_id]);//获取特征点坐标
            poses.push_back(&this->viewports->at(view_id).pose);
            view_ids.push_back(view_id);
            feature_ids.push_back(feature_id);
        }

        /* Skip tracks with too few valid cameras. */
        if ((int)poses.size() < min_num_views)//track的相机姿态要>2,一个三维点的有效观察视角（已校正）要多于2才能三角化
            continue;

        // ransac 三角量测过程，多视角时，一组特征点可能存在不符合相机姿态与大多数重建的三维点的特征点
        /* Accept track if triangulation was successful. */
        std::vector<std::size_t> outlier;
        math::Vec3d track_pos;
        if (!triangulator.triangulate(poses, pos, &track_pos, &stats, &outlier))//ransac求解三维点过程，
            // 遍历姿态所有匹配对，三角量测计算重投影误差，两视角投影向量夹角不宜过小；相机坐标系z轴为正值；重投影误差不应过大
            continue;
        this->tracks->at(i).pos = track_pos;//对应位置补充三维点

        /* Check if track contains outliers *///重投影误差太大的作为外点
        if (outlier.size() == 0)
            continue;

        /* Split outliers from track and generate new track */
        Track & inlier_track = this->tracks->at(i);
        Track outlier_track;
        outlier_track.invalidate();
        outlier_track.color = inlier_track.color;
        for (std::size_t i = 0; i < outlier.size(); ++i)//外点删除，过滤track
        {
            int const view_id = view_ids[outlier[i]];
            int const feature_id = feature_ids[outlier[i]];
            /* Remove outlier from inlier track */
            inlier_track.remove_view(view_id);
            /* Add features to new track */
            outlier_track.features.emplace_back(view_id, feature_id);
            /* Change TrackID in viewports */
            this->viewports->at(view_id).track_ids[feature_id] =
                this->tracks->size();
        }
        this->tracks->push_back(outlier_track);
    }

    if (this->opts.verbose_output)
    {
        triangulator.print_statistics(stats, std::cout);
        std::cout << "  Splitted " << this->tracks->size()
            - initial_tracks_size << " new tracks." << std::endl;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_full (void)
{
    this->bundle_adjustment_intern(-1);
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_single_cam (int view_id)
{
    if (view_id < 0 || std::size_t(view_id) >= this->viewports->size()
        || !this->viewports->at(view_id).pose.is_valid())
        throw std::invalid_argument("Invalid view ID");
    this->bundle_adjustment_intern(view_id);
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_points_only (void)
{
    this->bundle_adjustment_intern(-2);
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_intern (int single_camera_ba)//最初选择初始相机对传入参数-1，传入相机索引必然大于0此时先优化相机内参
{//三种优化方式，不同点在于雅克比矩阵以及增量正规方程
    ba::BundleAdjustment::Options ba_opts;
    ba_opts.fixed_intrinsics = this->opts.ba_fixed_intrinsics;
    ba_opts.verbose_output = this->opts.verbose_ba;
    if (single_camera_ba >= 0)
        ba_opts.bundle_mode = ba::BundleAdjustment::BA_CAMERAS;//  仅仅优化相机
    else if (single_camera_ba == -2)
        ba_opts.bundle_mode = ba::BundleAdjustment::BA_POINTS;  // 仅仅优化三维点
    else if (single_camera_ba == -1)
        ba_opts.bundle_mode = ba::BundleAdjustment::BA_CAMERAS_AND_POINTS;  // 同时优化相机和三维点
    else
        throw std::invalid_argument("Invalid BA mode selection");

    /* Convert camera to BA data structures. */
    std::vector<ba::Camera> ba_cameras;
    std::vector<int> ba_cameras_mapping(this->viewports->size(), -1);
    for (std::size_t i = 0; i < this->viewports->size(); ++i)//读取有效相机参数给 ba_cameras
    {
        if (single_camera_ba >= 0 && int(i) != single_camera_ba)//当优化单个增量相机时，i=相机索引才继续否则跳过
            continue;

        Viewport const& view = this->viewports->at(i);//读取相机索引
        CameraPose const& pose = view.pose;
        if (!pose.is_valid())//判定有无视角初始内参焦距；用于判断初始对相机，该对相机没法输入索引，利用初始化信息找到两个姿态
            continue;

        ba::Camera cam;
        cam.focal_length = pose.get_focal_length();
        std::copy(pose.t.begin(), pose.t.end(), cam.translation);
        std::copy(pose.R.begin(), pose.R.end(), cam.rotation);
        std::copy(view.radial_distortion,
            view.radial_distortion + 2, cam.distortion);
        ba_cameras_mapping[i] = ba_cameras.size();//ba_cameras_mapping索引是相机姿态，value>0代表该姿态需要处理
        ba_cameras.push_back(cam);//之后BA类需要利用ba_cameras索引是ba_cameras_mapping的value，ba_cameras_mapping索引是真实相机视角
    }

    /* Convert tracks and observations to BA data structures.建立视角与三维点的索引联系 */
    std::vector<ba::Observation> ba_points_2d;//Observation结构体：特征点位置，相机索引，视角下特征点索引
    std::vector<ba::Point3D> ba_points_3d;//Point3D：三维点坐标
    std::vector<int> ba_tracks_mapping(this->tracks->size(), -1);//ba_tracks_mapping索引是track索引
    for (std::size_t i = 0; i < this->tracks->size(); ++i)//遍历所有track
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        /* Add corresponding 3D point to BA. */
        ba::Point3D point;//结构体三维点坐标数组
        std::copy(track.pos.begin(), track.pos.end(), point.pos);
        ba_tracks_mapping[i] = ba_points_3d.size();// ba_tracks_mapping存放ba_points_3d的索引；已建立三维点track的三维点索引
        ba_points_3d.push_back(point);

        /* Add all observations to BA. */
        for (std::size_t j = 0; j < track.features.size(); ++j)//遍历track下所有特征点
        {
            int const view_id = track.features[j].view_id;
            if (!this->viewports->at(view_id).pose.is_valid())//该视角内参没有被初始化或改动说明此时不需要优化track内该视角
                continue;
            if (single_camera_ba >= 0 && view_id != single_camera_ba)//单独优化一个相机且此时track下相机视角不是要优化姿态
                continue;

            int const feature_id = track.features[j].feature_id;
            Viewport const& view = this->viewports->at(view_id);
            math::Vec2f const& f2d = view.features.positions[feature_id];

            ba::Observation point;//结构体：二维点位置，相机索引，三维点索引
            std::copy(f2d.begin(), f2d.end(), point.pos);
            point.camera_id = ba_cameras_mapping[view_id];//camera_id是ba_cameras_mapping的value，对应ba_cameras_mapping索引信息是视角
            point.point_id = ba_tracks_mapping[i];//利用ba_tracks_mapping建立二维点属于三维点索引,是特征点对应已建立三维点track的索引
            ba_points_2d.push_back(point);
        }
    }

    for (std::size_t i = 0; registered && i < this->survey_points->size(); ++i)
    {
        SurveyPoint const& survey_point = this->survey_points->at(i);

        /* Add corresponding 3D point to BA. */
        ba::Point3D point;
        std::copy(survey_point.pos.begin(), survey_point.pos.end(), point.pos);
        point.is_constant = true;
        ba_points_3d.push_back(point);

        /* Add all observations to BA. */
        for (std::size_t j = 0; j < survey_point.observations.size(); ++j)
        {
            SurveyObservation const& obs = survey_point.observations[j];
            int const view_id = obs.view_id;
            if (!this->viewports->at(view_id).pose.is_valid())
                continue;
            if (single_camera_ba >= 0 && view_id != single_camera_ba)
                continue;

            ba::Observation point;
            std::copy(obs.pos.begin(), obs.pos.end(), point.pos);
            point.camera_id = ba_cameras_mapping[view_id];
            point.point_id = ba_points_3d.size() - 1;
            ba_points_2d.push_back(point);
        }
    }

    /* Run bundle adjustment. *///ba类：cameras相机内外参，points三维点坐标，oberservation[camera_id相机ba_cameras_mapping的value,]
    ba::BundleAdjustment ba(ba_opts);
    ba.set_cameras(&ba_cameras);
    ba.set_points(&ba_points_3d);
    ba.set_observations(&ba_points_2d);
    ba.optimize();//ba的cameras传入重建三维点的相机内外参，points传三维点信息，observations
    ba.print_status();

    /* Transfer cameras back to SfM data structures. *///更新BA优化后的增量类视角下pose内外参，
    std::size_t ba_cam_counter = 0;//计数优化的相机
    for (std::size_t i = 0; i < this->viewports->size(); ++i)
    {
        if (ba_cameras_mapping[i] == -1)//value=-1表示本次优化ba_cameras_mapping的索引代表视角没有优化
            continue;

        Viewport& view = this->viewports->at(i);//非优化视角上步已跳过，找到viewports下刚刚ba的视角
        CameraPose& pose = view.pose;
        ba::Camera const& cam = ba_cameras[ba_cam_counter];//获得更新的ba视角相机内外参

        if (this->opts.verbose_output && !this->opts.ba_fixed_intrinsics)
        {
            std::cout << "Camera " << std::setw(3) << i
                << ", focal length: "
                << util::string::get_fixed(pose.get_focal_length(), 5)
                << " -> "
                << util::string::get_fixed(cam.focal_length, 5)
                << ", distortion: "
                << util::string::get_fixed(cam.distortion[0], 5) << " "
                << util::string::get_fixed(cam.distortion[1], 5)
                << std::endl;
        }
//视角更新BA过后相机姿态
        std::copy(cam.translation, cam.translation + 3, pose.t.begin());
        std::copy(cam.rotation, cam.rotation + 9, pose.R.begin());
        std::copy(cam.distortion, cam.distortion + 2, view.radial_distortion);
        pose.set_k_matrix(cam.focal_length, 0.0, 0.0);
        ba_cam_counter += 1;
    }

    /* Exit if single camera BA is used. */
    if (single_camera_ba >= 0)//单个相机更新姿态，没有三维点更新
        return;

    /* Transfer tracks back to SfM data structures. *///更新ba后增量类track下三维点pos信息
    std::size_t ba_track_counter = 0;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        ba::Point3D const& point = ba_points_3d[ba_track_counter];
        std::copy(point.pos, point.pos + 3, track.pos.begin());
        ba_track_counter += 1;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::invalidate_large_error_tracks (void)
{//计算重投影误差，根据重投影误差中值给误差门限，误差大的track删去三维点信息
    /* Iterate over all tracks and sum reprojection error. */
    std::vector<std::pair<double, std::size_t> > all_errors;
    std::size_t num_valid_tracks = 0;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)//遍历track
    {
        if (!this->tracks->at(i).is_valid())//判断三维点是否值存在，存在就有效
            continue;

        num_valid_tracks += 1;
        math::Vec3f const& pos3d = this->tracks->at(i).pos;//得到track下重建的三维点坐标
        FeatureReferenceList const& ref = this->tracks->at(i).features;

        double total_error = 0.0f;
        int num_valid = 0;
        for (std::size_t j = 0; j < ref.size(); ++j)
        {
            /* Get pose and 2D position of feature. *///track下特征点的视角，特征点位置与索引
            int view_id = ref[j].view_id;
            int feature_id = ref[j].feature_id;

            Viewport const& viewport = this->viewports->at(view_id);
            CameraPose const& pose = viewport.pose;
            if (!pose.is_valid())
                continue;

            math::Vec2f const& pos2d = viewport.features.positions[feature_id];

            /* Project 3D feature and compute reprojection error. */
            math::Vec3d x = pose.R * pos3d + pose.t;//三维点向相机坐标系投影
            math::Vec2d x2d(x[0] / x[2], x[1] / x[2]);
            double r2 = x2d.square_norm();
            x2d *= (1.0 + r2 * (viewport.radial_distortion[0]
                + viewport.radial_distortion[1] * r2))
                * pose.get_focal_length();//相机坐标系向成像坐标系投影，二维点坐标
            total_error += (pos2d - x2d).square_norm();//三维点投影到视角，计算重投影误差
            num_valid += 1;
        }
        total_error /= static_cast<double>(num_valid);//计算平均误差，每次平均误差都考虑总误差
        all_errors.push_back(std::pair<double, int>(total_error, i));//all_errors,first平均误差，second索引
    }

    if (num_valid_tracks < 2)
        return;

    /* Find the 1/2 percentile. *///三维点投影到各个二维平面计算重投影误差，根据误差结果排序
    std::size_t const nth_position = all_errors.size() / 2;
    std::nth_element(all_errors.begin(), all_errors.begin() + nth_position, all_errors.end());//中值前值比中值小，中值后比中间大
    double const square_threshold = all_errors[nth_position].first * this->opts.track_error_threshold_factor;

    /* Delete all tracks with errors above the threshold. */
    int num_deleted_tracks = 0;
    for (std::size_t i = nth_position; i < all_errors.size(); ++i) {
        if (all_errors[i].first > square_threshold) {
            this->tracks->at(all_errors[i].second).invalidate();
            num_deleted_tracks += 1;
        }
    }

    if (this->opts.verbose_output)
    {
        float percent = 100.0f * static_cast<float>(num_deleted_tracks)
            / static_cast<float>(num_valid_tracks);
        std::cout << "Deleted " << num_deleted_tracks
            << " of " << num_valid_tracks << " tracks ("
            << util::string::get_fixed(percent, 2)
            << "%) above a threshold of "
            << std::sqrt(square_threshold) << "." << std::endl;
    }
}

/* ---------------------------------------------------------------- */
// 将三维点和相机坐标系归一化到固定范围内
void
Incremental::normalize_scene (void)
{
    this->registered = false;

    /* Compute AABB for all camera centers. */
    math::Vec3d aabb_min(std::numeric_limits<double>::max());
    math::Vec3d aabb_max(-std::numeric_limits<double>::max());
    math::Vec3d camera_mean(0.0);
    int num_valid_cameras = 0;
    for (std::size_t i = 0; i < this->viewports->size(); ++i)
    {
        CameraPose const& pose = this->viewports->at(i).pose;
        if (!pose.is_valid())
            continue;

        math::Vec3d center = -(pose.R.transposed() * pose.t);
        for (int j = 0; j < 3; ++j)
        {
            aabb_min[j] = std::min(center[j], aabb_min[j]);
            aabb_max[j] = std::max(center[j], aabb_max[j]);
        }
        camera_mean += center;
        num_valid_cameras += 1;
    }

    /* Compute scale and translation. */
    double scale = 10.0 / (aabb_max - aabb_min).maximum();
    math::Vec3d trans = -(camera_mean / static_cast<double>(num_valid_cameras));

    /* Transform every point. */
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        if (!this->tracks->at(i).is_valid())
            continue;

        this->tracks->at(i).pos = (this->tracks->at(i).pos + trans) * scale;
    }

    /* Transform every camera. */
    for (std::size_t i = 0; i < this->viewports->size(); ++i)
    {
        CameraPose& pose = this->viewports->at(i).pose;
        if (!pose.is_valid())
            continue;
        pose.t = pose.t * scale - pose.R * trans * scale;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::print_registration_error (void) const
{
    double sum = 0;
    int num_points = 0;
    for (std::size_t i = 0; i < this->survey_points->size(); ++i)
    {
        SurveyPoint const& survey_point = this->survey_points->at(i);

        std::vector<math::Vec2f> pos;
        std::vector<CameraPose const*> poses;
        for (std::size_t j = 0; j < survey_point.observations.size(); ++j)
        {
            SurveyObservation const& obs = survey_point.observations[j];
            int const view_id = obs.view_id;
            if (!this->viewports->at(view_id).pose.is_valid())
                continue;

            pos.push_back(obs.pos);
            poses.push_back(&this->viewports->at(view_id).pose);
        }

        if (pos.size() < 2)
            continue;

        math::Vec3d recon = triangulate_track(pos, poses);
        sum += (survey_point.pos - recon).square_norm();
        num_points += 1;
    }

    if (num_points > 0)
    {
        double mse = sum / num_points;
        std::cout << "Reconstructed " << num_points
            << " survey points with a MSE of " << mse << std::endl;
    }
    else
    {
        std::cout << "Failed to reconstruct all survey points." << std::endl;
    }
}

/* ---------------------------------------------------------------- */

core::Bundle::Ptr
Incremental::create_bundle (void) const
{
    if (this->opts.verbose_output && this->registered)
        this->print_registration_error();

    /* Create bundle data structure. */
    core::Bundle::Ptr bundle = core::Bundle::create();
    {
        /* Populate the cameras in the bundle. */
        core::Bundle::Cameras& bundle_cams = bundle->get_cameras();
        bundle_cams.resize(this->viewports->size());
        for (std::size_t i = 0; i < this->viewports->size(); ++i)
        {
            core::CameraInfo& cam = bundle_cams[i];
            Viewport const& viewport = this->viewports->at(i);
            CameraPose const& pose = viewport.pose;
            if (!pose.is_valid())
            {
                cam.flen = 0.0f;
                continue;
            }

            cam.flen = static_cast<float>(pose.get_focal_length());
            cam.ppoint[0] = pose.K[2] + 0.5f;
            cam.ppoint[1] = pose.K[5] + 0.5f;
            std::copy(pose.R.begin(), pose.R.end(), cam.rot);
            std::copy(pose.t.begin(), pose.t.end(), cam.trans);
            cam.dist[0] = viewport.radial_distortion[0]
                * MATH_POW2(pose.get_focal_length());
            cam.dist[1] = viewport.radial_distortion[1]
                * MATH_POW2(pose.get_focal_length());
        }

        /* Populate the features in the Bundle. */
        core::Bundle::Features& bundle_feats = bundle->get_features();
        bundle_feats.reserve(this->tracks->size());
        for (std::size_t i = 0; i < this->tracks->size(); ++i)
        {
            Track const& track = this->tracks->at(i);
            if (!track.is_valid())
                continue;

            /* Copy position and color of the track. */
            bundle_feats.push_back(core::Bundle::Feature3D());
            core::Bundle::Feature3D& f3d = bundle_feats.back();
            std::copy(track.pos.begin(), track.pos.end(), f3d.pos);
            f3d.color[0] = track.color[0] / 255.0f;
            f3d.color[1] = track.color[1] / 255.0f;
            f3d.color[2] = track.color[2] / 255.0f;
            f3d.refs.reserve(track.features.size());
            for (std::size_t j = 0; j < track.features.size(); ++j)
            {
                /* For each reference copy view ID, feature ID and 2D pos. */
                f3d.refs.push_back(core::Bundle::Feature2D());
                core::Bundle::Feature2D& f2d = f3d.refs.back();
                f2d.view_id = track.features[j].view_id;
                f2d.feature_id = track.features[j].feature_id;

                FeatureSet const& features
                    = this->viewports->at(f2d.view_id).features;
                math::Vec2f const& f2d_pos
                    = features.positions[f2d.feature_id];
                std::copy(f2d_pos.begin(), f2d_pos.end(), f2d.pos);
            }
        }
    }

    return bundle;
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
