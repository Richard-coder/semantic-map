/*************************************************************************
	> File Name: Semantic Map
	> Author: Weizhi Zhang
	> Mail: zhi_zhuce@126.com
	> Created Time: 2017年06月11日
    * 104,with beyes,show map
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include "Classifier.h"
using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

// g2o定义
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
using namespace Eigen;
// 给定index，读取一帧数据
FRAME readFrame(int index, ParameterReader &pd);
// 给定旋转矩阵和评议向量，估计帧间运动大小的函数
double normofTransform(cv::Mat rvec, cv::Mat tvec);

//对检测两个帧匹配结果的定义：enum 枚举名{ 枚举值表 };
enum CHECK_RESULT
{
    NOT_MATCHED = 0,
    TOO_FAR_AWAY,
    TOO_CLOSE,
    KEYFRAME
};
// 检测关键帧的函数声明
CHECK_RESULT checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops = false);
// 检测近距离的回环
void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);
// 随机检测回环
void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);

int main(int argc, char **argv)
{
    Matrix<float, 11, 11> M1;
    M1 << 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        0, 0, 0, 0.05, 0, 0.95, 0, 0, 0, 0, 0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        0, 0, 0, 0.05, 0, 0.95, 0, 0, 0, 0, 0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
        1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0;

    Matrix<float, 11, 1> Mt_1;
    Mt_1 << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    // 前面部分和vo是一样的
    // 读取帧的开始和结束索引
    ParameterReader pd;
    int startIndex = atoi(pd.getData("start_index").c_str());
    int endIndex = atoi(pd.getData("end_index").c_str());

    // 所有的关键帧都放在了这里
    vector<FRAME> keyframes;
    // initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex;                 // 当前索引为currIndex
    FRAME currFrame = readFrame(currIndex, pd); // 读取当前帧的索引，rgb图像，深度图像
    //计算当前帧的特征点和描述子
    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp(currFrame, detector, descriptor); //保存特征描述子和特征点到FRAME
    //根据当前帧的彩色图像、深度图像和相机参数，将当前帧的图像点转换为点云
    PointCloud::Ptr cloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);

    /******************************* 
    // 新增:有关g2o的初始化
    *******************************/
    // 初始化求解器
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer; // 最后用的就是这个东东
    globalOptimizer.setAlgorithm(solver);
    // 不要输出调试信息
    globalOptimizer.setVerbose(false);

    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); //估计为单位矩阵
    v->setFixed(true);                             //第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);
    //将第一帧加到关键帧的堆栈里
    keyframes.push_back(currFrame);
    //读取定义关键帧的一些参数
    double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    //将“是否执行闭环检测”设定为yes
    bool check_loop_closure = pd.getData("check_loop_closure") == string("yes");
    vector<int> allframe;
    allframe.push_back(-1);
    allframe.push_back(0);
    int ifkey = 0;

    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++)
    {
        //读取当前帧的数据，计算关键点和描述子，并将当前帧与关键帧里最后一帧匹配，并根据匹配结果采取不同策略
        cout << "Reading files " << currIndex << endl;
        FRAME currFrame = readFrame(currIndex, pd);                                         // 读取currFrame
        computeKeyPointsAndDesp(currFrame, detector, descriptor);                           //提取特征
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer); //匹配该帧与keyframes里最后一帧
        switch (result)                                                                     // 根据匹配结果不同采取不同策略
        {
        case NOT_MATCHED:
            //没匹配上，直接跳过
            cout << RED "Not enough inliers." << endl;
            break;
        case TOO_FAR_AWAY:
            // 太近了，也直接跳
            cout << RED "Too far away, may be an error." << endl;
            break;
        case TOO_CLOSE:
            // 太远了，可能出错了
            cout << RESET "Too close, not a keyframe" << endl;
            break;
        case KEYFRAME:
            cout << GREEN "This is a new keyframe" << endl;
            // 不远不近，刚好
            // 检测回环
            if (check_loop_closure)
            {
                //近距离回环，将当前帧与关键帧的末尾几帧匹配，以检测是否有回环
                checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                //将当前帧与随机选取的几帧关键帧匹配，以检测是否有回环
                checkRandomLoops(keyframes, currFrame, globalOptimizer);
            }
            //将当前帧压入关键帧的堆栈
            keyframes.push_back(currFrame);
            ifkey = 1;
            break;
        default:
            break;
        }
        if (ifkey)
        {
            allframe.push_back(keyframes.size() - 1);
        }
        else
        {
            allframe.push_back(-1);
        }
        ifkey = 0;
    }

    // 优化（得到相机的位姿）
    cout << RESET "optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("./result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100); //可以指定优化步数
    globalOptimizer.save("./result_after.g2o");
    cout << "Optimization done." << endl;

    double gridsize = atof(pd.getData("voxel_grid").c_str()); //分辨图可以在parameters.txt里调

    //pcl实时显示
    pcl::visualization::CloudViewer viewer("viewer");
    bool visualize = pd.getData("visualize_pointcloud") == string("yes");
    //octomap
    octomap::ColorOcTree *semMap;
    double cell_resolution_;
    double sem_ProbHit_;
    double sem_ProbMiss_;
    double prob_thres_ = 0.5;
    cell_resolution_ = gridsize;                         //octomap是吗以八叉树形式存储，这里设置地图的分辨率
    semMap = new octomap::ColorOcTree(cell_resolution_); //创建指向地图对象的指针
    semMap->setClampingThresMax(1.0);                    //在八叉树中，用概率表示一个节点是否被占据，这里设置概率的最大值和最小值
    semMap->setOccupancyThres(0);
    octomap::ColorOcTree tree(gridsize);
    //place2005test------start
    string label_file = pd.getData("label_file");
    SemanticLabel semlabel(label_file);

    ::google::InitGoogleLogging(argv[0]);

    string model_file = "/home/richard/ros-semantic-mapper/deploy.prototxt";
    string trained_file = "/home/richard/ros-semantic-mapper/places.caffemodel";
    string mean_file = "/home/richard/ros-semantic-mapper/places205CNN_mean.binaryproto";

    Classifier classifier(model_file, trained_file, mean_file, semlabel);
    
    // 拼接点云地图，（在进行这一步的时候，关键帧都已经提取完了啊。）
    cout << "saving the point cloud map..." << endl;
    PointCloud::Ptr output(new PointCloud()); //全局地图
    PointCloud::Ptr tmp(new PointCloud());
    PointCloud::Ptr semoutput(new PointCloud()); //全局地图
    PointCloud::Ptr semtmp(new PointCloud());
    PointCloud::Ptr semout2(new PointCloud());
    PointCloud::Ptr semtmp2(new PointCloud());
    //设置地图的分辨率等参数
    pcl::VoxelGrid<PointT> voxel;  // 网格滤波器，调整地图分辨率
    pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0); //4m以上就不要了

    voxel.setLeafSize(gridsize, gridsize, gridsize);
    //
    cv::namedWindow("pic+test");
    //
    int outlabel = 0;
    cv::Mat img_ori;
    Matrix<float, 11, 1> prob_res;
    Matrix<float, 11, 1> MA;
    Matrix<float, 11, 1> Mt;
    float per[semlabel.labelname.size()];
    for (size_t alli = 1; alli < allframe.size(); alli++)
    {
        if (allframe[alli] != -1)
        {
            int i = allframe[alli];
            // 从g2o优化结果中读取当前帧优化过后的相机位姿
            g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(globalOptimizer.vertex(keyframes[i].frameID));
            Eigen::Isometry3d pose = vertex->estimate();
            //将当前关键帧转换为点云                                              //该帧优化后的位姿
            PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera); //转成点云
            // 对当前地图分辨率等进行优化，滤波
            voxel.setInputCloud(newCloud);
            voxel.filter(*tmp);
            pass.setInputCloud(tmp);
            pass.filter(*newCloud);
            // 把点云变换后加入全局地图中
            pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
            //对当前场景进行预测
            cv::Mat img = keyframes[i].rgb;
            vector<Prediction> predictions = classifier.Classify(img);

            for (int idx = 0; idx < predictions.size(); idx++)
            {
                prob_res(idx, 0) = predictions[idx].second;
            }
            MA = M1 * Mt_1;
            Mt = (prob_res.array() * MA.array()).matrix();
            double psum = Mt.sum();
            Mt = Mt / psum;
            Mt_1 = Mt;
            for (int idx = 0; idx < predictions.size(); idx++)
            {
                predictions[idx].second = Mt(idx, 0);
            }
            for (int idx = 0; idx < predictions.size(); idx++)
            {
                if (predictions[outlabel].second < predictions[idx].second)
                {
                    outlabel = idx;
                }
                per[idx] = float(predictions[idx].second);
            }

            img_ori = img.clone();

            for (auto ite : tmp->points)
            {
                for (int idx = 0; idx < predictions.size(); idx++)
                {
                    octomap::point3d point(ite.x, idx, ite.z);
                    float lo = std::log(predictions[idx].second / (1 - predictions[idx].second)); //logit变换
                    semMap->updateNode(point, lo);
                    octomap::ColorOcTreeNode *cell = semMap->search(point);
                    octomap::ColorOcTreeNode::Color cS(semlabel.labelcolor[idx][0], semlabel.labelcolor[idx][1], semlabel.labelcolor[idx][2]);
                    cell->setColor(cS);
                }
            }
            for (octomap::ColorOcTree::iterator itr = semMap->begin_leafs(), end = semMap->end_leafs(); itr != end; itr++)
            {
                octomap::ColorOcTreeNode *cell = semMap->search(itr.getCoordinate());
                for (int idx = 0; idx < predictions.size(); idx++)
                {
                    octomap::ColorOcTreeNode *tcell = semMap->search(itr.getCoordinate().x(), idx, itr.getCoordinate().z());
                    if (tcell->getOccupancy() > cell->getOccupancy())
                    {
                        cell = tcell;
                    }
                }
                PointT p;
                p.x = itr.getCoordinate().x();
                p.z = itr.getCoordinate().z();
                p.y = 0;
                octomap::ColorOcTreeNode::Color c = cell->getColor();
                p.r = c.r;
                p.g = c.g;
                p.b = c.b;
                semout2->points.push_back(p);
            }
            //octomap start
            octomap::Pointcloud cloud_octo;
            for (auto p : tmp->points)
                cloud_octo.push_back(p.x, p.y, p.z);

            tree.insertPointCloud(cloud_octo,
                                  octomap::point3d(pose(0, 3), pose(1, 3), pose(2, 3)));

            for (auto p : tmp->points)
                tree.integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
            //octomap end
            *output += *tmp;
            tmp->clear();
            newCloud->clear();
            semtmp->clear();
            if (visualize == true)
            {
                viewer.showCloud(semout2);
            }
        }
        else
        {
            FRAME nokeyFrame = readFrame(alli, pd);
            img_ori = nokeyFrame.rgb.clone();
        }
        IplImage img_text_n;
        img_text_n = IplImage(img_ori);
        IplImage *img_text = &img_text_n;
        int text_x = 3;
        int text_y = 33;
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 0.5, 0.5, 0, 1, 4);
        for (int idx = 0; idx < semlabel.labelname.size(); idx++)
        {
            cvRectangle(img_text, cvPoint(text_x, text_y), cvPoint(text_x + 0 + int(100 * per[idx]), text_y - 6), cvScalar(255, 0, 0), 6);
            if (outlabel == idx)
            {
                cvPutText(img_text, semlabel.labelname[idx].c_str(), cvPoint(text_x, text_y), &font, cvScalar(0, 255, 0));
            }
            else
            {
                cvPutText(img_text, semlabel.labelname[idx].c_str(), cvPoint(text_x, text_y), &font, cvScalar(0, 0, 255));
            }
            text_y += 18;
        }
        //cv::Mat img_res;
        // cv::resize(img,img_res,cv::Size(1280,960));
        cvShowImage("pic+test", img_text);

        cv::waitKey(60);
    }
    voxel.setInputCloud(output);
    voxel.filter(*tmp);

    voxel.setInputCloud(semout2);
    voxel.filter(*semtmp2);

    pcl::io::savePCDFile("./result.pcd", *tmp);
    //pcl::io::savePCDFile("./semresult.pcd", *semtmp);
    pcl::io::savePCDFile("./2semresult.pcd", *semtmp2);
    cout << "Final map is saved." << endl;
    tree.updateInnerOccupancy();
    tree.write("map.ot");

    cout << "done." << endl;
    //
    //
}

FRAME readFrame(int index, ParameterReader &pd)
{
    //FRAME是一个结构体，定义在slambase.h保存当前帧的索引，rgb和深度图，特特征描述子等
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");
    //读取rgb图像存入FRAME
    stringstream ss;
    ss << rgbDir << index << rgbExt;
    string filename;
    ss >> filename;
    f.rgb = cv::imread(filename);
    //读取深度图像存入FRAME
    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;
    //读取索引，存入FRAME
    f.depth = cv::imread(filename, -1);
    f.frameID = index;
    return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}
//输入是两帧图像和g2o优化器
//输出是两帧图像的匹配结果：没匹配上/太近/太远/不远不近，刚好
CHECK_RESULT checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops)
{
    //PnP求解，估计这两帧图像之间的运动（肯定是要匹配的啦），可以返回内点数、旋转矩阵、平移向量
    static ParameterReader pd;
    static int min_inliers = atoi(pd.getData("min_inliers").c_str());
    static double max_norm = atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    if (result.inliers < min_inliers) //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if (is_loops == false)
    {
        if (norm >= max_norm)
            return TOO_FAR_AWAY; // too far away, may be error
    }
    else
    {
        if (norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if (norm <= keyframe_threshold)
        return TOO_CLOSE; // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }
    //如果帧间运动大小不远不近：则向g2o中添加这个顶点and与上一帧联系的边，顶点就是当前帧对应的相机姿态，边就是PnP求解的变换矩阵
    // 边部分
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex(0, opti.vertex(f1.frameID));
    edge->setVertex(1, opti.vertex(f2.frameID));
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation(information);
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
    // edge->setMeasurement( T );
    edge->setMeasurement(T.inverse());
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}
//将当前帧和前面的几个帧依次调用checkKeyframes进行匹配（is_loops=true所以只添加边，不添加新的顶点。
void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());

    // 就是把currFrame和 frames里末尾几个测一遍
    if (frames.size() <= nearby_loops)
    {
        // no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
}
//与checkNearbyLoops类似，区别在于随机与关键vector中的几个关键帧进行匹配
void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    srand((unsigned int)time(NULL));
    // 随机取一些帧进行检测

    if (frames.size() <= random_loops)
    {
        // no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
    else
    {
        // randomly check loops
        for (int i = 0; i < random_loops; i++)
        {
            int index = rand() % frames.size();
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }
}
