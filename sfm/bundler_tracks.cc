/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <set>

#include "core/image_tools.h"
#include "core/image_drawing.h"
#include "sfm/bundler_tracks.h"

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

/*
 * Merges tracks and updates viewports accordingly.
 */
void
unify_tracks(int view1_tid, int view2_tid,
    TrackList* tracks, ViewportList* viewports)
{//view1_tid,view2_tid对应不同track索引
    /* Unify in larger track. 统一让view1_tid下的特征点信息最多*/
    if (tracks->at(view1_tid).features.size()
        < tracks->at(view2_tid).features.size())
        std::swap(view1_tid, view2_tid);

    Track& track1 = tracks->at(view1_tid);
    Track& track2 = tracks->at(view2_tid);

    for (std::size_t k = 0; k < track2.features.size(); ++k)
    {
        int const view_id = track2.features[k].view_id;
        int const feat_id = track2.features[k].feature_id;
        viewports->at(view_id).track_ids[feat_id] = view1_tid;//track2下元素信息，访问viewports对应视角下的特征点对应的track_ids,改为track1索引
    }//viewports下track_ids修正后，将原track2元素添加到track1
    track1.features.insert(track1.features.end(),
        track2.features.begin(), track2.features.end());
    /* Free old track's memory. clear() does not work. */
    track2.features = FeatureReferenceList();//track2清空
}

/* ---------------------------------------------------------------- */
void
Tracks::compute (PairwiseMatching const& matching,
    ViewportList* viewports, TrackList* tracks)
{//初始化viewports中track_ids向量为-1，
    // matching点对索引信息对应的viewports两-1的track_ids代表对应匹配点未创建track，创建一个track，匹配点对索引作为viewports下track_ids的索引，将value从未建立的-1改为tracks.size
    //创建track向量，pos为空，颜色为空，features（两个匹配组：view_id与对应feature_id）
    //合并不同视角匹配点的Track，去除合并后为空的track以及存在重复视角的track
    //正确track后根据匹配的特征点颜色给定track颜色
    /* Initialize per-viewport track IDs. */
    for (std::size_t i = 0; i < viewports->size(); ++i)//遍历视角（相机内外参，特征点位置）分配路径空间
    {
        Viewport& viewport = viewports->at(i);
        viewport.track_ids.resize(viewport.features.positions.size(), -1);
    }

    /* Propagate track IDs. */
    if (this->opts.verbose_output)
        std::cout << "Propagating track IDs..." << std::endl;
//viewports各视角的track_ids idx表示特征点索引，value表示track的索引；track索引下的value是Track结构体，包含视角与视角下特征点索引
    /* Iterate over all pairwise matchings and create tracks. */
    tracks->clear();
    for (std::size_t i = 0; i < matching.size(); ++i)
    {
        TwoViewMatching const& tvm = matching[i];//结构体匹配两视角索引，匹配点索引对向量
        Viewport& viewport1 = viewports->at(tvm.view_1_id);//根据匹配视角索引，获得相机初始内外参，初始track，特征点位置
        Viewport& viewport2 = viewports->at(tvm.view_2_id);

        /* Iterate over matches for a pair. */
        for (std::size_t j = 0; j < tvm.matches.size(); ++j)//遍历一组视角下所有特征点匹配点对
        {
            CorrespondenceIndex idx = tvm.matches[j];
            int const view1_tid = viewport1.track_ids[idx.first];//视角的track_id索引是匹配点对的匹配点索引，也是track的索引：track未建立是-1，建立后是track建立的size
            int const view2_tid = viewport2.track_ids[idx.second];
            if (view1_tid == -1 && view2_tid == -1)//都是默认-1时证明两视角间匹配点没有创建track
            {
                /* No track ID associated with the match. Create track.track_ids先放track索引，后有track填充元素 */
                viewport1.track_ids[idx.first] = tracks->size();//没构建的track_ids以该视角的匹配点索引为索引，value是构建成功的track数
                viewport2.track_ids[idx.second] = tracks->size();
                tracks->push_back(Track());//track添加一个新Track匿名结构体：三维点坐标，颜色，特征点向量（track对应的视角索引，视角下特征点索引）
                tracks->back().features.push_back(
                    FeatureReference(tvm.view_1_id, idx.first));//对track的feature下刚载入的Track结构添加特征点信息（视角索引，视角下特征点索引）
                tracks->back().features.push_back(
                    FeatureReference(tvm.view_2_id, idx.second));//track新加的Track索引正是循环前一次的tracks->size()
            }
            else if (view1_tid == -1 && view2_tid != -1)//存在一个-1时，说明路径已经建立但新添加视角下特征点未连入track
            {
                /* Propagate track ID from first to second view. */
                viewport1.track_ids[idx.first] = view2_tid;//新建视角未连入特征点索引的track_ids的value-1变为由建立好的另一个viewpoints的track_ids以匹配特征点的value（之前建立统计track）
                tracks->at(view2_tid).features.push_back(
                    FeatureReference(tvm.view_1_id, idx.first));//track的已建立Track索引正是之前建立Track的track统计
            }
            else if (view1_tid != -1 && view2_tid == -1)
            {
                /* Propagate track ID from second to first view. */
                viewport2.track_ids[idx.second] = view1_tid;
                tracks->at(view1_tid).features.push_back(
                    FeatureReference(tvm.view_2_id, idx.second));
            }
            else if (view1_tid == view2_tid)//两特征点最后遇见，之前两特征点分别建立了track，并且track名相同，说明两点已经在track内
            {
                /* Track ID already propagated. */
            }
            else//view1_tid !== view2_tid且不为-1说明两特征点分与别的视角的特征点已经建立了track
            {
                /* 如果匹配的两个特征点对应的track id不一样，则将track进行融合
                 * A track ID is already associated with both ends of a match,
                 * however, is not consistent. Unify tracks.
                 */
                unify_tracks(view1_tid, view2_tid, tracks, viewports);
            }
        }
    }

    /* Find and remove invalid tracks or tracks with conflicts. */
    if (this->opts.verbose_output)
        std::cout << "Removing tracks with conflicts..." << std::flush;

    // 删除不合理的track(同一个track,包含同一副图像中的多个特征点以及track中没有特征点）
    std::size_t const num_invalid_tracks = this->remove_invalid_tracks(viewports, tracks);
    if (this->opts.verbose_output)
        std::cout << " deleted " << num_invalid_tracks << " tracks." << std::endl;

    /* Compute color for every track. */
    if (this->opts.verbose_output)
        std::cout << "Colorizing tracks..." << std::endl;
    for (std::size_t i = 0; i < tracks->size(); ++i)//遍历Tracks
    {
        Track& track = tracks->at(i);
        math::Vec4f color(0.0f, 0.0f, 0.0f, 0.0f);
        for (std::size_t j = 0; j < track.features.size(); ++j)//遍历一个track下所有特征获得视角索引与特征点索引
        {
            FeatureReference const& ref = track.features[j];
            FeatureSet const& features = viewports->at(ref.view_id).features;
            math::Vec3f const feature_color(features.colors[ref.feature_id]);//获得视角下二维特征点颜色
            color += math::Vec4f(feature_color, 1.0f);//r,g,b,1
        }
        track.color[0] = static_cast<uint8_t>(color[0] / color[3] + 0.5f);//路径颜色是所有路径下特征点颜色均值
        track.color[1] = static_cast<uint8_t>(color[1] / color[3] + 0.5f);
        track.color[2] = static_cast<uint8_t>(color[2] / color[3] + 0.5f);
    }
}

/* ---------------------------------------------------------------- */

int
Tracks::remove_invalid_tracks (ViewportList* viewports, TrackList* tracks)
{//选出错误track，利用正确track count创建正确的track向量
    /*
     * Detect invalid tracks where a track contains no features, or
     * multiple features from a single view.
     */
    // 删除tracks没有特征点，或者是包含同一幅图像中的多个特征点
    std::vector<bool> delete_tracks(tracks->size());
    int num_invalid_tracks = 0;
    for (std::size_t i = 0; i < tracks->size(); ++i)
    {//选出之前因为两个匹配点对应track不同而合并track导致的空track
        if (tracks->at(i).features.empty()) {
            delete_tracks[i] = true;
            continue;
        }

        std::set<int> view_ids;//创建set存储视角
        for (std::size_t j = 0; j < tracks->at(i).features.size(); ++j)//遍历track内Track结构体的特征点信息
        {//选出无法向集合中添加全部视角的track
            FeatureReference const& ref = tracks->at(i).features[j];
            if (view_ids.insert(ref.view_id).second == false) {//set插入失败说明不同视角有相同视角重复
                num_invalid_tracks += 1;
                delete_tracks[i] = true;
                break;
            }
        }
    }

    /* Create a mapping from old to new track IDs. 先清除viewports中错误信息，错误track、没有建立的track索引都是-1*/
    std::vector<int> id_mapping(delete_tracks.size(), -1);
    int valid_track_counter = 0;
    for (std::size_t i = 0; i < delete_tracks.size(); ++i)
    {//错误的track索引对应value-1，正确的索引是track count【调整前是track.size(包括了错误track)】
        if (delete_tracks[i])
            continue;
        id_mapping[i] = valid_track_counter;
        valid_track_counter += 1;
    }

    /* Fix track IDs stored in the viewports.更新viewports的track_ids */
    for (std::size_t i = 0; i < viewports->size(); ++i)//遍历视角
    {
        std::vector<int>& track_ids = viewports->at(i).track_ids;
        for (std::size_t j = 0; j < track_ids.size(); ++j)//track_ids的value是去除了错误的track统计信息
            if (track_ids[j] >= 0)
                track_ids[j] = id_mapping[track_ids[j]];
    }

    /* Clean the tracks from the vector. 利用判断出错误的向量清洗track，错误的track直接清除*/
    math::algo::vector_clean(delete_tracks, tracks);

    return num_invalid_tracks;
}


SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
