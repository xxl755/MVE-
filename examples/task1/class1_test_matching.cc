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

#include "util/aligned_memory.h"
#include "util/timer.h"
#include "core/image.h"
#include "core/image_tools.h"
#include "core/image_io.h"

#include "features/surf.h"
#include "features/sift.h"
#include "features/matching.h"
#include "sfm/feature_set.h"
#include "visualizer.h"


core::ByteImage::Ptr
visualize_matching (features::Matching::Result const& matching,
    core::ByteImage::Ptr image1, core::ByteImage::Ptr image2,
    std::vector<math::Vec2f> const& pos1, std::vector<math::Vec2f> const& pos2)
{
    /* Visualize keypoints. */
    sfm::Correspondences2D2D vis_matches;
    for (std::size_t i = 0; i < matching.matches_1_2.size(); ++i)
    {
        if (matching.matches_1_2[i] < 0)
            continue;
        int const j = matching.matches_1_2[i];

        sfm::Correspondence2D2D match;
        std::copy(pos1[i].begin(), pos1[i].end(), match.p1);
        std::copy(pos2[j].begin(), pos2[j].end(), match.p2);
        vis_matches.push_back(match);
    }

    std::cout << "Drawing " << vis_matches.size() << " matches..." << std::endl;
    core::ByteImage::Ptr match_image = sfm::Visualizer::draw_matches
        (image1, image2, vis_matches);
    return match_image;
}

#define DISCRETIZE_DESCRIPTORS 0
template <typename T>
void
convert_sift_descriptors(features::Sift::Descriptors const& sift_descr,
    util::AlignedMemory<math::Vector<T, 128> >* aligned_descr)
{
    aligned_descr->resize(sift_descr.size());
    T* data_ptr = aligned_descr->data()->begin();
    for (std::size_t i = 0; i < sift_descr.size(); ++i, data_ptr += 128)
    {
        sfm::Sift::Descriptor const& d = sift_descr[i];
#if DISCRETIZE_DESCRIPTORS
        for (int j = 0; j < 128; ++j)
        {
            float value = d.data[j];
            value = math::clamp(value, 0.0f, 1.0f);
            value = math::round(value * 255.0f);
            data_ptr[j] = static_cast<unsigned char>(value);
        }
#else
        std::copy(d.data.begin(), d.data.end(), data_ptr);
#endif
    }
}

template <typename T>
void
convert_surf_descriptors(sfm::Surf::Descriptors const& surf_descr,
    util::AlignedMemory<math::Vector<T, 64> >* aligned_descr)
{
    aligned_descr->resize(surf_descr.size());
    T* data_ptr = aligned_descr->data()->begin();
    for (std::size_t i = 0; i < surf_descr.size(); ++i, data_ptr += 64)
    {
        sfm::Surf::Descriptor const& d = surf_descr[i];
#if DISCRETIZE_DESCRIPTORS
        for (int j = 0; j < 64; ++j)
        {
            float value = d.data[j];
            value = math::clamp(value, -1.0f, 1.0f);
            value = math::round(value * 127.0f);
            data_ptr[j] = static_cast<signed char>(value);
        }
#else
        std::copy(d.data.begin(), d.data.end(), data_ptr);
#endif
    }
}

void
feature_set_matching (core::ByteImage::Ptr image1, core::ByteImage::Ptr image2,std::string filename)
{
    /*FeatureSet 计算并存储一个视角的特征点，包含SIFT和SURF特征点 */
    sfm::FeatureSet::Options feature_set_opts;//默认特征点类型为sift，之后初始化sift与surf
    //feature_types设置为FEATURE_ALL表示检测SIFT和SURF两种特征点进行匹配
    feature_set_opts.feature_types = sfm::FeatureSet::FEATURE_ALL;//改变特征点类型属性
    feature_set_opts.sift_opts.verbose_output = true;
    //feature_set_opts.surf_opts.verbose_output = true;
    //feature_set_opts.surf_opts.contrast_threshold = 500.0f;

    // 计算第一幅图像的SIT和SURF特征点
    sfm::FeatureSet feat1(feature_set_opts);//构造Options struct，制定特征点类型，sift与surf属性构造
    feat1.compute_features(image1);//获得所有排序好的sift与surf特征点，并分配划线起始点与线颜色
    // 计算第二幅图像的SIFT和SURF特征点
    sfm::FeatureSet feat2(feature_set_opts);
    feat2.compute_features(image2);

    /* 对sift特征描述子进行匹配 */
    // sift 特征描述子匹配参数
    sfm::Matching::Options sift_matching_opts;//以下设置sift描述子匹配的options struct
    sift_matching_opts.lowe_ratio_threshold = 0.8f;
    sift_matching_opts.descriptor_length = 128;
    sift_matching_opts.distance_threshold = std::numeric_limits<float>::max();

#if DISCRETIZE_DESCRIPTORS
    util::AlignedMemory<math::Vec128us, 16> sift_descr1, sift_descr2;
#else
    util::AlignedMemory<math::Vec128f, 16> sift_descr1, sift_descr2;
#endif
    // 将描述子转成特定的内存格式，128维sift描述子向量
    convert_sift_descriptors(feat1.sift_descriptors, &sift_descr1);
    convert_sift_descriptors(feat2.sift_descriptors, &sift_descr2);

    // 进行双向匹配
    sfm::Matching::Result sift_matching;
    sfm::Matching::twoway_match(sift_matching_opts,
        sift_descr1.data()->begin(), sift_descr1.size(),
        sift_descr2.data()->begin(), sift_descr2.size(),
        &sift_matching);

    // 去除不一致的匹配对，匹配对feature1和feature2是一致的需要满足，feature1的最近邻居
    // 是feature2，feature2的最近邻是feature1
    sfm::Matching::remove_inconsistent_matches(&sift_matching);
    std::cout << "Consistent Sift Matches: "
        << sfm::Matching::count_consistent_matches(sift_matching)
        << std::endl;


    /*  对surf特征描述子进行匹配  */
    // surf特征匹配参数
    sfm::Matching::Options surf_matching_opts;
    // 最近邻和次近邻的比
    surf_matching_opts.lowe_ratio_threshold = 0.7f;
    // 特征描述子的维度
    surf_matching_opts.descriptor_length = 64;
    surf_matching_opts.distance_threshold = std::numeric_limits<float>::max();

#if DISCRETIZE_DESCRIPTORS
    util::AlignedMemory<math::Vec64s, 16> surf_descr1, surf_descr2;
#else
    util::AlignedMemory<math::Vec64f, 16> surf_descr1, surf_descr2;
#endif
    // 将描述子转化成特殊的格式
    convert_surf_descriptors(feat1.surf_descriptors, &surf_descr1);
    convert_surf_descriptors(feat2.surf_descriptors, &surf_descr2);

    // 进行surf描述子的双向匹配
    sfm::Matching::Result surf_matching;
    sfm::Matching::twoway_match(surf_matching_opts,
        surf_descr1.data()->begin(), surf_descr1.size(),
        surf_descr2.data()->begin(), surf_descr2.size(),
        &surf_matching);
    // 去除不一致的匹配对，匹配对feature 1 和 feature 2 互为最近邻
    sfm::Matching::remove_inconsistent_matches(&surf_matching);//matches_1_2与matches_2_1的索引与value应该相互对应
    std::cout << "Consistent Surf Matches: "
        << sfm::Matching::count_consistent_matches(surf_matching)
        << std::endl;

    // 对sift匹配的结果和surf匹配的结果进行融合，注意surf描述子索引在sift描述子后面，surf描述子的索引注意偏置+sift描述子.size
    sfm::Matching::Result matching;     //Result struct vector<int> matches_1_2与 matches_2_1
    sfm::Matching::combine_results(sift_matching, surf_matching, &matching);

    std::cout << "Consistent Matches: "
        << sfm::Matching::count_consistent_matches(matching)
        << std::endl;

    /* 特征匹配可视化 */
    /* Draw features. */
    std::cout<<"matches_1_2 size: "<<matching.matches_1_2.size()<<"feat1.size:"<<feat1.sift_descriptors.size()<<feat1.surf_descriptors.size()<<std::endl;
    std::cout<<"matches_2_1 size: "<<matching.matches_2_1.size()<<"feat2.size:"<<feat2.sift_descriptors.size()<<std::endl;
    std::vector<sfm::Visualizer::Keypoint> features1;//
    for (std::size_t i = 0; i < feat1.sift_descriptors.size(); ++i)
    {
        if (matching.matches_1_2[i] == -1)//根据匹配情况决定特征点保留情况
            continue;

        sfm::Sift::Descriptor const& descr = feat1.sift_descriptors[i];//获得匹配到特征点的特征点信息
        sfm::Visualizer::Keypoint kp;
        kp.orientation = descr.orientation;//将描述子信息给显示点
        kp.radius = descr.scale * 3.0f;
        kp.x = descr.x;
        kp.y = descr.y;
        features1.push_back(kp);//存放匹配成功的特征点信息，这里只有sift
    }

 //   std::vector<sfm::Visualizer::Keypoint> features3;//
    for (std::size_t i = 0; i < matching.matches_1_2.size(); ++i)
    {
       // if (i==432)
         //   std::cout<<matching.matches_1_2[i]-feat1.sift_descriptors.size()<<std::endl;
 //       std::cout<<matching.matches_1_2[i]<<"   "<<feat1.sift_descriptors.size()<<std::endl;
        int data=matching.matches_1_2[i]-feat1.sift_descriptors.size();//不做int转换，该减法得到一个无符号数
        if (data>0){
            std::cout<<matching.matches_1_2[i]-feat1.sift_descriptors.size()<<std::endl;


        std::cout<<"matching.matches_1_2第"<<i<<"个surf索引为："<<matching.matches_1_2[i]<<std::endl;
        sfm::Surf::Descriptor const& descr = feat1.surf_descriptors[i-feat1.sift_descriptors.size()];//获得匹配到特征点的特征点信息
            std::cout<<"surf索引为："<<i-feat1.sift_descriptors.size()<<std::endl;
        sfm::Visualizer::Keypoint kp;
        kp.orientation = descr.orientation;//将描述子信息给显示点
        kp.radius = descr.scale * 3.0f;
        kp.x = descr.x;
        kp.y = descr.y;
        features1.push_back(kp);}
    }

    std::vector<sfm::Visualizer::Keypoint> features2;
    for (std::size_t i = 0; i < feat2.sift_descriptors.size(); ++i)
    {
        if (matching.matches_2_1[i] == -1)
            continue;

        sfm::Sift::Descriptor const& descr = feat2.sift_descriptors[i];
        sfm::Visualizer::Keypoint kp;
        kp.orientation = descr.orientation;
        kp.radius = descr.scale * 3.0f;
        kp.x = descr.x;
        kp.y = descr.y;
        features2.push_back(kp);
    }
 //   std::vector<sfm::Visualizer::Keypoint> features4;//
    for (std::size_t i = 0; i < matching.matches_2_1.size(); ++i)
    {
        signed int match=matching.matches_2_1[i];//i范围超界会造成想不到的错误

        signed int data=match-(signed int)feat2.sift_descriptors.size();//不做int转换，该减法得到一个无符号数
        if (data>0){
            std::cout<<i<<std::endl;
            std::cout<<matching.matches_2_1[i]-feat2.sift_descriptors.size()<<std::endl;


            std::cout<<"matching.matches2_1第"<<i<<"个surf索引为："<<matching.matches_2_1[i]<<std::endl;
            sfm::Surf::Descriptor const& descr = feat2.surf_descriptors[i-feat2.sift_descriptors.size()];//获得匹配到特征点的特征点信息
            std::cout<<"surf索引为："<<i-feat2.sift_descriptors.size()<<std::endl;
            sfm::Visualizer::Keypoint kp;
            kp.orientation = descr.orientation;//将描述子信息给显示点
            kp.radius = descr.scale * 3.0f;
            kp.x = descr.x;
            kp.y = descr.y;
            features2.push_back(kp);}//存放匹配成功的特征点信息，这里只有sift，surf特征点需要match向量索引值减去偏置用feat1.surf_descriptors
    }

    image1 = sfm::Visualizer::draw_keypoints(image1,
        features1, sfm::Visualizer::RADIUS_BOX_ORIENTATION);
    image2 = sfm::Visualizer::draw_keypoints(image2,
        features2, sfm::Visualizer::RADIUS_BOX_ORIENTATION);

    core::ByteImage::Ptr match_image = visualize_matching(
        matching, image1, image2, feat1.positions, feat2.positions);
    std::string output_filename = filename;
    //std::string output_filename = "/home/xsun/ImageBasedModellingEduV1.0/tmp/matching_featureset.png";
    std::cout << "Saving visualization to " << output_filename << std::endl;
    core::image::save_file(match_image, output_filename);
}

int
main (int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Syntax: " << argv[0] << " image1 image2" << "output filename path"<<std::endl;
        return 1;
    }

 // 用于加速
#ifdef __SSE2__
    std::cout << "SSE2 is enabled!" << std::endl;
#endif
#ifdef __SSE3__
    std::cout << "SSE3 is enabled!" << std::endl;
#endif

    /* Regular two-view matching. */
    core::ByteImage::Ptr image1, image2;
    try
    {
        std::cout << "Loading " << argv[1] << "..." << std::endl;
        image1 = core::image::load_file(std::string(argv[1]));
        // 图像尺寸减半
        image1 = core::image::rescale_half_size<uint8_t>(image1);
        //image1 = core::image::rescale_half_size<uint8_t>(image1);
        //image1 = core::image::rotate<uint8_t>(image1, core::image::ROTATE_CCW);

        std::cout << "Loading " << argv[2] << "..." << std::endl;
        image2 = core::image::load_file(argv[2]);
        // 图像尺寸减半
        image2 = core::image::rescale_half_size<uint8_t>(image2);
        //image2 = core::image::rescale_half_size<uint8_t>(image2);
        //image2 = core::image::rotate<uint8_t>(image2, core::image::ROTATE_CCW);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // 进行特征提取和特征匹配
    feature_set_matching(image1, image2,argv[3]);

    return 0;
}
