//
// Created by xxl on 2021/5/14.
//
#include "defines.h"
#include "functions.h"
#include "sfm/bundler_common.h"
#include "sfm/bundler_features.h"
#include "sfm/bundler_matching.h"
#include "sfm/bundler_intrinsics.h"
#include "sfm/bundler_init_pair.h"
#include "sfm/bundler_tracks.h"
#include "sfm/bundler_incremental.h"
#include "core/scene.h"
#include "util/timer.h"
#include <assert.h>
#include "calibration.h"
#include <util/file_system.h>
#include <core/bundle_io.h>
#include <core/camera.h>
#include <fstream>
#include <iostream>
#include "core/image_io.h"


core::Scene::Ptr
make_scene(const std::string & image_folder_path, const std::string & scene_path){

    util::WallTimer timer;

    /*** 创建文件夹 ***/
    const std::string views_path = util::fs::join_path(scene_path, "views/");
    util::fs::mkdir(scene_path.c_str());
    util::fs::mkdir(views_path.c_str());

    /***扫描文件夹，获取所有的图像文件路径***/
    util::fs::Directory dir;
    try {dir.scan(image_folder_path);//获得文件路径与文件名
    }
    catch (std::exception&e){
        std::cerr << "Error scanning input dir: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Found " << dir.size() << " directory entries." << std::endl;

    core::Scene::Ptr scene= core::Scene::create("");

    /**** 开始加载图像 ****/
    std::sort(dir.begin(), dir.end());
    int num_imported = 0;
    for(std::size_t i=0; i< dir.size(); i++){
        // 是一个文件夹
        if(dir[i].is_dir){
            std::cout<<"Skipping directory "<<dir[i].name<<std::endl;
            continue;
        }
        util::fs::File ffile=dir[i];
        std::string fname = dir[i].name;
        std::string afname = dir[i].get_absolute_name();//路径+文件名

        // 从可交换信息文件中读取图像焦距
        std::string exif;
        core::ImageBase::Ptr image = load_any_image(afname, & exif);
        if(image == nullptr){
            continue;
        }
//初始meta_data中camera信息
        core::View::Ptr view = core::View::create();//创建新的视角地址
        view->set_id(num_imported);//设置view中meta_data的data键值对view_id:视角索引（图片索引）
        view->set_name(remove_file_extension(fname));
        //设置view中meta_data的data键值对view.name:图片名，去除后缀
        // 限制图像尺寸
        int orig_width = image->width();
        image = limit_image_size(image, MAX_PIXELS);//图片尺寸减半直到满足要求
        if (orig_width == image->width() && has_jpeg_extension(fname))
            view->set_image_ref(afname, "original");
        else
            view->set_image(image, "original");//view中的images填入图片信息，包括尺寸等

        add_exif_to_view(view, exif);//向view中blobs添加exif信息

        scene->get_views().push_back(view);//不同图片就是不同的视角

        /***保存视角信息到本地****/
        std::string mve_fname = make_image_name(num_imported);
        std::cout << "Importing image: " << fname
                  << ", writing MVE view: " << mve_fname << "..." << std::endl;
        view->save_view_as(util::fs::join_path(views_path, mve_fname));//views_path文件路径，视角信息mve_fname

        num_imported+=1;
    }

    std::cout << "Imported " << num_imported << " input images, "
              << "took " << timer.get_elapsed() << " ms." << std::endl;

    return scene;
}

core::Scene::Ptr
make_sceneary(const std::string & image_folder_path, const std::string & scene_path){
//创建文件夹存放中间结果，向scene存放图片索引、信息与图片；
    util::WallTimer timer;

    /*** 创建两个文件夹 ***/
    const std::string views_path = util::fs::join_path(scene_path, "views/");
    util::fs::mkdir(scene_path.c_str());
    util::fs::mkdir(views_path.c_str());

    /***扫描文件夹，获取所有的图像文件路径***/
    util::fs::Directory dir;
    try {dir.scan(image_folder_path);//获得文件路径与文件名，File类存放path与name
    }
    catch (std::exception&e){
        std::cerr << "Error scanning input dir: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Found " << dir.size() << " directory entries." << std::endl;

    core::Scene::Ptr scene= core::Scene::create("");

    /**** 开始加载图像 ****/
    std::sort(dir.begin(), dir.end());
    int num_imported = 0;
    for(std::size_t i=0; i< dir.size(); i++){
        // 是一个文件夹
        if(dir[i].is_dir){
            std::cout<<"Skipping directory "<<dir[i].name<<std::endl;
            continue;
        }
        util::fs::File ffile=dir[i];
        std::string fname = dir[i].name;
        std::string afname = dir[i].get_absolute_name();//路径path+文件名name

        // 从可交换信息文件中读取图像焦距
        std::string exif;
        core::ImageBase::Ptr image = core::image::load_file(afname);
        if(image == nullptr){
            continue;
        }
//初始meta_data中camera信息
        core::View::Ptr view = core::View::create();//创建新的视角地址
        view->set_id(num_imported);//设置view中meta_data存放的data键值对view.id:视角索引（图片索引）；is_dirty=true
        view->set_name(remove_file_extension(fname));//meta_data存放的data键值对view.name:图片名

        // 限制图像尺寸
        int orig_width = image->width();
        image = limit_image_size(image, MAX_PIXELS);//图片尺寸减半直到满足要求
//        if (orig_width == image->width() && has_jpeg_extension(fname))
//            view->set_image_ref(afname, "original");//view中的images填入图片信息，包括尺寸文件，绝对路径等
//        else
            view->set_image(image, "original");//view中的images填入图片信息，包括尺寸文件，绝对路径以及原图片

      //  add_exif_to_view(view, exif);//向view中blobs添加exif信息

        scene->get_views().push_back(view);//不同图片就是不同的视角，取出scene的views列表存入view

        /***保存视角信息到本地****/
        std::string mve_fname = make_image_name(num_imported);// mve_fname文件名： view+id*4.mve
        std::cout << "Importing image: " << fname
                  << ", writing MVE view: " << mve_fname << "..." << std::endl;
        view->save_view_as(util::fs::join_path(views_path, mve_fname));//views_path文件路径，视角信息mve_fname

        num_imported+=1;
    }

    std::cout << "Imported " << num_imported << " input images, "
              << "took " << timer.get_elapsed() << " ms." << std::endl;

    return scene;
}

void
features_and_matching (core::Scene::Ptr scene,
                       sfm::bundler::ViewportList* viewports,
                       sfm::bundler::PairwiseMatching* pairwise_matching){
//特征点计算：viewports获得scene的图片放在image下检测得到features【归一化position，colors，描述子】，清空scene下图片、viewports下图片
//创建bundler_matching，属性viewports传入viewports所有信息，match【放入所有视角描述子】，所有viewports描述子信息清空；两两视角描述子匹配，
//PairwiseMatching向量存放两个视角及视角下对应匹配点索引对
    /* Feature computation for the scene. */
    sfm::bundler::Features::Options feature_opts;//初始化检测图片名，尺寸初始最大，检测子初始化为sift，且sift也初始化
    feature_opts.image_embedding = "original";
    feature_opts.max_image_size = MAX_PIXELS;
    feature_opts.feature_options.feature_types = sfm::FeatureSet::FEATURE_SURF;

    std::cout << "Computing image features..." << std::endl;
    {
        util::WallTimer timer;
        sfm::bundler::Features bundler_features(feature_opts);
        bundler_features.compute(scene, viewports);//viewpoints结构相机内外参都是0，image空，features填满：position归一，描述子向量不变；；scene的image清空
//特征计算：遍历视角图片，找到图片特征点，特征点坐标归一化到-0.5，0.5间；scene中view存放各个图片meta_data(camera相机初始，data图片索引名称)images（图片信息）blobs交换文件
        std::cout << "Computing features took " << timer.get_elapsed()
                  << " ms." << std::endl;
        std::cout<<"Feature detection took " + util::string::get(timer.get_elapsed()) + "ms."<<std::endl;
    }

    /* Exhaustive matching between all pairs of views. */
    sfm::bundler::Matching::Options matching_opts;
    //匹配限制：最少特征点，ransac后特征最少点，低分辨率匹配特征数，低分辨率匹配的最小匹配数
    //matching_opts.ransac_opts.max_iterations = 1000;
    //matching_opts.ransac_opts.threshold = 0.0015;
    matching_opts.ransac_opts.verbose_output = false;
    matching_opts.use_lowres_matching = false;
    matching_opts.match_num_previous_frames = false;
    matching_opts.matcher_type = sfm::bundler::Matching::MATCHER_EXHAUSTIVE;
//特征匹配
    std::cout << "Performing feature matching..." << std::endl;
    {
        util::WallTimer timer;
        sfm::bundler::Matching bundler_matching(matching_opts);//Matching类：progress（总匹配视角数，匹配完成视角数），matcher(processed_featureset各视角，描述子向量)，viewports信息与viewpoint完全相同
        bundler_matching.init(viewports);//清除viewpoint描述子信息，保留position信息，其余不变相机内外参为0，track_id空
        //bundler_matching的matcher存放描述子向量，viewports信息与viewpoint完全相同，同时清空所有viewports类的描述子向量
        bundler_matching.compute(pairwise_matching);//任意两视角两两匹配，传入空PairwiseMatching向量（两视角索引，两匹配点id对）
        std::cout << "Matching took " << timer.get_elapsed()
                  << " ms." << std::endl;
        std::cout<< "Feature matching took "
                    + util::string::get(timer.get_elapsed()) + "ms."<<std::endl;
    }
//pairwise_matching存放TwoViewMatching结构体，包括两视角（认为一个视角一个相机）索引，匹配索引存储特征点索引对
    if (pairwise_matching->empty()) {
        std::cerr << "Error: No matching image pairs. Exiting." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
int main(int argc,char* argv[]){
    Mat innexMatrix(3,3,CV_32F,Scalar::all(0));
    getCameraMatrix("./Job/caliberation_result.txt",innexMatrix);
    float f1=(innexMatrix.at<float>(0,0)/1200+innexMatrix.at<float>(1,1)/1200)/2;
    float &f2=f1;
    std::cout<<"相机焦距： "<<f1<<endl;

//    core::Scene::Ptr scene0=make_scene(argv[1],argv[2]);
    core::Scene::Ptr scene=make_sceneary(argv[1],argv[2]);

    sfm::bundler::ViewportList viewports;//viewport结构体：焦距，畸变2，相机姿态类（内参，外参），图片，
    // 特征点类（位置，颜色，姿态，两种描述子），track索引的向量
    sfm::bundler::PairwiseMatching pairwise_matching;//存放匹配点信息：TwoViewMatching结构体，两个视角id，两视角匹配点索引pair
    features_and_matching(scene, &viewports, &pairwise_matching );//传入信息：scene包含图片及所有信息
    float* fptr;
    if (scene->get_views().size()!=viewports.size())
        throw std::invalid_argument("Viewports/scene size mismatch");
    core::Scene::ViewList const &sceneview=scene->get_views();
    for (int i = 0; i < sceneview.size(); ++i) {
        viewports[i].focal_length=f1;
    }

    /****** 开始增量的捆绑调整*****/
    util::WallTimer timer;
    /* Compute connected feature components, i.e. feature tracks. */
    sfm::bundler::TrackList tracks;//Track结构体，属性：三维点坐标、颜色，FeatureReferencelist向量（FeatureReference结构体视角索引，特征点索引id）
    {
        sfm::bundler::Tracks::Options tracks_options;
        tracks_options.verbose_output = true;

        sfm::bundler::Tracks bundler_tracks(tracks_options);
        std::cout << "Computing feature tracks..." << std::endl;
        bundler_tracks.compute(pairwise_matching, &viewports, &tracks);//传入pairwise_matching匹配视角索引与匹配内点索引向量；viewports焦距，features中特征点归一化坐标
        //利用初始图像连接图（图像两两间匹配），包含匹配的视角，特征点索引创建track；此重建的track有特征点信息没有track对应的三维点信息
//viewports向量内元素track_idx向量索引是视角下特征点索引，value是track向量索引；track内Track存放特征点视角与索引
        std::cout << "Created a total of " << tracks.size()
                  << " tracks." << std::endl;
    }

    // 计算初始的匹配对
    sfm::bundler::InitialPair::Result init_pair_result;//结构体：两个视角索引，两视角相机内外参矩阵
    sfm::bundler::InitialPair::Options init_pair_opts;
    //init_pair_opts.homography_opts.max_iterations = 1000;
    //init_pair_opts.homography_opts.threshold = 0.005f;
    init_pair_opts.homography_opts.verbose_output = false;
    init_pair_opts.max_homography_inliers = 0.8f;//单应矩阵判断内点门限
    init_pair_opts.verbose_output = true;

    // 开始计算初始的匹配对
    sfm::bundler::InitialPair init_pair(init_pair_opts);
    init_pair.initialize(viewports, tracks);//初始相机匹配对传入viewports（目前有焦距，features下特征点位置，track_ids）；tracks目前主要有Track的特征信息
    init_pair.compute_pair(&init_pair_result);//匹配点足够多，相机基线够长，不能纯旋转姿态，重建点正确量足够；得到视角索引，初始相机组内外参
    if (init_pair_result.view_1_id < 0 || init_pair_result.view_2_id < 0//几个初始匹配对标准
        || init_pair_result.view_1_id >= static_cast<int>(viewports.size())
        || init_pair_result.view_2_id >= static_cast<int>(viewports.size()))
    {
        std::cerr << "Error finding initial pair, exiting!" << std::endl;
        std::cerr << "Try manually specifying an initial pair." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Using views " << init_pair_result.view_1_id
              << " and " << init_pair_result.view_2_id
              << " as initial pair." << std::endl;

    /* Incrementally compute full bundle. */
    sfm::bundler::Incremental::Options incremental_opts;
    incremental_opts.pose_p3p_opts.max_iterations = 1000;
    incremental_opts.pose_p3p_opts.threshold = 0.005f;
    incremental_opts.pose_p3p_opts.verbose_output = false;
    incremental_opts.track_error_threshold_factor = TRACK_ERROR_THRES_FACTOR;
    incremental_opts.new_track_error_threshold = NEW_TRACK_ERROR_THRES;
    incremental_opts.min_triangulation_angle = MATH_DEG2RAD(1.0);
    incremental_opts.ba_fixed_intrinsics = false;
    //incremental_opts.ba_shared_intrinsics = conf.shared_intrinsics;
    incremental_opts.verbose_output = true;
    incremental_opts.verbose_ba = true;

    /* Initialize viewports with initial pair. */
    viewports[init_pair_result.view_1_id].pose = init_pair_result.view_1_pose;//更新初始相机对参数给viewports
    viewports[init_pair_result.view_2_id].pose = init_pair_result.view_2_pose;

    /* Initialize the incremental bundler and reconstruct first tracks. */
    sfm::bundler::Incremental incremental(incremental_opts);
    incremental.initialize(&viewports, &tracks);//类传入viewports的内外参主要是初始对，特征点位置track_ids给viewports，track的features信息给track，并将track三维点信息初始设为无效

    // 知道两相机姿态，现在对当前两个视角进行track重建，并且如果track存在外点，则将每个track的外点剥离成新的track
    incremental.triangulate_new_tracks(2);//track重建就是进行三角量测，恢复三维点坐标给incremental的track向量pos信息;传入最少相机限制2
    incremental.invalidate_large_error_tracks();

    /* Run bundle adjustment. */
    std::cout << "Running full bundle adjustment..." << std::endl;
    incremental.bundle_adjustment_full();
    int track_count= incremental.get_tracks_nums();
    int num_cameras_reconstructed=2;
    int full_ba_num_skipped = 0;//控制每隔几个视角全局捆绑调整一次
//    while (1){
//
//        /* Find suitable next views for reconstruction. */
//
//        vector<int> next_views;
//        incremental.find_next_views(&next_views);//统计可以匹配已建立三维点track但未矫正视角的匹配次数找到下一个视角
//
//        /* Reconstruct the next view. */
//        int next_viewid=-1;//重置下一个视角
//        for (int i = 0; i < next_views.size(); ++i) {
//            std::cout << std::endl;
//            std::cout << "Adding next view ID " << next_views[i]
//                      << " (" << (num_cameras_reconstructed + 1) << " of "
//                      << viewports.size() << ")..." << std::endl;
//
//            //根据建好三维点与新视角二维点信息，采用ransac的p3p方法构建新相机外参，
//            // 内点数量要大于新视角能匹配track的点数量1/3,剔除相机在track中的二维点外点与三维点联系
//            if (incremental.reconstruct_next_view(next_views[i])){
//                next_viewid=next_views[i];
//                break;
//            }
//        }
//        //next_views向量为空时next_view_id不再更新满足条件，说明没有互选视角，全部建立结束
//        if(next_viewid==-1){//所有视角都重建完成重建后，如果视角过多还要一次全局ba
//            if (full_ba_num_skipped == 0) {
//                std::cout << "No valid next view." << std::endl;
//                std::cout << "SfM reconstruction finished." << std::endl;
//                break;
//            }
//            else
//            {
//                incremental.triangulate_new_tracks(MIN_VIEWS_PER_TRACK);//track重建
//                std::cout << "Running full bundle adjustment..." << std::endl;
//                incremental.invalidate_large_error_tracks();//track滤波，全局捆绑调整
//                incremental.bundle_adjustment_full();
//                full_ba_num_skipped = 0;
//                continue;
//            }
//        }
//
//        /* Run single-camera bundle adjustment. */
//        incremental.bundle_adjustment_single_cam(next_viewid);
//        num_cameras_reconstructed+=1;
//        /* Run full bundle adjustment only after a couple of views. 新增加track重建*/
//        int const full_ba_skip_views =  std::min(100, num_cameras_reconstructed / 10);
//        if (full_ba_num_skipped<full_ba_skip_views)
//        //相机姿态小于10个视角时候每次都BA调整全局，10~20每隔增加一个视角全局调整一次，20~30每增加两个个调整一次。。。。。
//        {
//            std::cout << "Skipping full bundle adjustment (skipping "
//                      << full_ba_skip_views << " views)." << std::endl;
//            full_ba_num_skipped+=1;
//
//        }
//        else{
//            incremental.triangulate_new_tracks(2);//新视角优化后，track重建
//
//            track_count=incremental.get_tracks_nums();
//
//            std::cout << "Running full bundle adjustment..." << std::endl;
//
//            incremental.invalidate_large_error_tracks();//track滤波，去除重投影误差大的track关系
//            incremental.bundle_adjustment_full();//全局捆绑调整
//            full_ba_num_skipped=0;
//        }
//
//
//    }
    while (true)
    {
        /* Find suitable next views for reconstruction. */
        int valid_track_size= incremental.get_tracks_nums();
        std::vector<int> next_views;
        incremental.find_next_views(&next_views);//统计已建立的三维点track下并未建立相机姿态的视角能匹配track的二维点数量，将视角按能看到三维点数量排序

        /* Reconstruct the next view. */
        int next_view_id = -1;
        for (std::size_t i = 0; i < next_views.size(); ++i)
        {
            std::cout << std::endl;
            std::cout << "Adding next view ID " << next_views[i]
                      << " (" << (num_cameras_reconstructed + 1) << " of "
                      << viewports.size() << ")..." << std::endl;
            if (incremental.reconstruct_next_view(next_views[i]))//视角重建，重建成功跳出for循环，否则继续找下一合适的候选视角
            {//根据建好三维点与新视角二维点信息，采用ransac的p3p方法构建新相机外参，内点数量要大于新视角能匹配track的点数量1/3,剔除相机在track中的二维点外点与三维点联系
                next_view_id = next_views[i];
                break;
            }
        }

        if (next_view_id < 0) {//next_views向量为空时next_view_id不再更新满足条件，说明没有互选视角，全部建立结束
            if (full_ba_num_skipped == 0) {
                std::cout << "No valid next view." << std::endl;
                std::cout << "SfM reconstruction finished." << std::endl;
                break;
            }
            else
            {
                incremental.triangulate_new_tracks(MIN_VIEWS_PER_TRACK);//track重建
                std::cout << "Running full bundle adjustment..." << std::endl;
                incremental.invalidate_large_error_tracks();//track滤波，全局捆绑调整

                incremental.bundle_adjustment_full();
                full_ba_num_skipped = 0;
                continue;
            }
        }

        /* Run single-camera bundle adjustment. */
        std::cout << "Running single camera bundle adjustment..." << std::endl;
        incremental.bundle_adjustment_single_cam(next_view_id);//重建成功后，ba调整非线性优化单个相机姿态
        num_cameras_reconstructed += 1;

        /* Run full bundle adjustment only after a couple of views. 新增加track重建*/
        int const full_ba_skip_views =  std::min(100, num_cameras_reconstructed / 10);
        //相机姿态小于10个视角时候每次都BA调整全局，10~20每隔增加一个视角全局调整一次，20~30每增加两个个调整一次。。。。。
        if (full_ba_num_skipped < full_ba_skip_views)
        {
            std::cout << "Skipping full bundle adjustment (skipping "
                      << full_ba_skip_views << " views)." << std::endl;
            full_ba_num_skipped += 1;
        }
        else
        {
            incremental.triangulate_new_tracks(MIN_VIEWS_PER_TRACK);//新视角优化后，track重建
            std::cout << "Running full bundle adjustment..." << std::endl;
            valid_track_size= incremental.get_tracks_nums();

            incremental.invalidate_large_error_tracks();//track滤波，去除重投影误差大的track关系
            incremental.bundle_adjustment_full();//全局捆绑调整

            full_ba_num_skipped = 0;
        }
    }

//有效路径保存
    sfm::bundler::TrackList valid_tracks;
    for (int i = 0; i < tracks.size(); ++i) {
        if (tracks[i].is_valid()){
            valid_tracks.push_back(tracks[i]);
        }
    }

    std::cout << "SfM reconstruction took " << timer.get_elapsed()
              << " ms." << std::endl;
    std::cout<< "SfM reconstruction took "
                + util::string::get(timer.get_elapsed()) + "ms."<<std::endl;

    /***** 保存输出结果***/
    std::string filename = argv[3];
    std::ofstream out_file(filename);

    assert(out_file.is_open());
    out_file<<"ply"<<std::endl;
    out_file<<"format ascii 1.0"<<std::endl;
    out_file<<"element vertex "<<valid_tracks.size()<<std::endl;
    out_file<<"property float x"<<std::endl;
    out_file<<"property float y"<<std::endl;
    out_file<<"property float z"<<std::endl;
    out_file<<"property uchar red"<<std::endl;
    out_file<<"property uchar green"<<std::endl;
    out_file<<"property uchar blue"<<std::endl;
    out_file<<"end_header"<<std::endl;

    for (int i = 0; i < valid_tracks.size(); ++i) {
        out_file<<valid_tracks[i].pos[0]<<" "<<valid_tracks[i].pos[1]<<" "<<valid_tracks[i].pos[2]<<" "
                <<(int)valid_tracks[i].color[0]<<" "<<(int)valid_tracks[i].color[1]<<" "<<(int)valid_tracks[i].color[2]<<endl;
    }

    out_file.close();

    /* Save bundle file to scene. */
    std::cout << "Creating bundle data structure..." << std::endl;
    core::Bundle::Ptr bundle = incremental.create_bundle();
    core::save_mve_bundle(bundle, std::string(argv[2]) + "/synth_0.out");

    /* Apply bundle cameras to views. */
    core::Bundle::Cameras const& bundle_cams = bundle->get_cameras();
    core::Scene::ViewList const& views = scene->get_views();
    if (bundle_cams.size() != views.size())
    {
        std::cerr << "Error: Invalid number of cameras!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

#pragma omp parallel for schedule(dynamic,1)
    for (std::size_t i = 0; i < bundle_cams.size(); ++i) {
        core::View::Ptr view = views[i];
        core::CameraInfo const &cam = bundle_cams[i];
        if (view == nullptr)
            continue;
        if (view->get_camera().flen == 0.0f && cam.flen == 0.0f)
            continue;

        view->set_camera(cam);

        /* Undistort image. */
        if (!undistorted_name.empty()) {
            core::ByteImage::Ptr original
                    = view->get_byte_image(original_name);
            if (original == nullptr)
                continue;
            core::ByteImage::Ptr undist
                    = core::image::image_undistort_k2k4<uint8_t>
                            (original, cam.flen, cam.dist[0], cam.dist[1]);
            view->set_image(undist, undistorted_name);
        }
    }
    return 0;
}