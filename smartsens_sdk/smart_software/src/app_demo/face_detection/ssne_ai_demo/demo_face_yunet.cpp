/*
 * @Filename: demo_face_yunet.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-01-20
 * @Copyright (c) 2025 SmartSens
 * @Description: YuNet三通道人脸检测演示程序
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include "include/utils.hpp"

using namespace std;

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

// OSD 贴图结构体
struct osdInfo {
    std::string filename; // OSD 文件名
    uint16_t x;           // 起始坐标 x
    uint16_t y;           // 起始坐标 y
};

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief YuNet人脸检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 1920;    // 输入图像宽度
    int img_height = 1080;   // 输入图像高度

    // 模型配置参数
    array<int, 2> det_shape = {160, 120};  // YuNet模型输入尺寸
    string path_det = "/app_demo/app_assets/models/yunet_160x120.m1model";  // YuNet人脸检测模型路径

    // OSD 信息
    static osdInfo osds[3] = {
        {"si.ssbmp", 10, 10},
        {"te.ssbmp", 90, 10},
        {"wei.ssbmp", 170, 10}
    };

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }

    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸 1920×1080
    array<int, 2> crop_shape = {1440, 1080};  // 裁剪尺寸（保持图像resize后比例不变）
    const int crop_offset_x = 240;  // 裁剪时X方向的偏移量（左右各裁240px）
    // 原图: 1920×1080, 模型输入图：160×120
    // 为了保证模型输入图经过resize后比例不变，需要先将原图裁剪为crop图: 1440×1080 (左右各裁240px)
    // 比例检查: 1440/1080 = 1.333, 160/120 = 1.333 ✓

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）

    // YuNet人脸检测模型初始化
    YUNET detector;
    int box_len = det_shape[0] * det_shape[1] / 64 * 10;  // 计算最大检测框数量
    detector.Initialize(path_det, &crop_shape, &det_shape, true, box_len);  // 初始化检测器（支持关键点）

    // 人脸检测结果初始化
    FaceDetectionResult* det_result = new FaceDetectionResult;

    // OSD可视化器初始化（用于绘制检测框）
    VISUALIZER visualizer;
    //visualizer.Initialize(img_shape);  // 初始化可视化器（配置图像尺寸）
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");  // 初始化可视化器（配置图像尺寸和位图LUT）
    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  // 等待系统稳定

    uint16_t num_frames = 0;  // 帧计数器
    uint8_t osd_index = 0; // osd 贴图 index
    ssne_tensor_t img_sensor;  // 图像tensor定义

    // OSD 贴图
    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    //循环50000帧后推出，循环次数可以修改，也可以改成while(true)
    while (!check_exit_flag()) {

        // 从sensor获取图像（裁剪图）
        processor.GetImage(&img_sensor);

        // YuNet人脸检测模型推理（置信度阈值0.6）
        detector.Predict(&img_sensor, det_result, 0.6f);

        /**********************************************************************************
         * 3.1 判断是否有检测到人脸
         **********************************************************************************/
        if (det_result->boxes.size() > 0) {
            /**********************************************************************************
             * 3.2 坐标转换：将crop图坐标转换为原图坐标
             **********************************************************************************/
            std::vector<std::array<float, 4>> boxes_original_coord;  // 存储转换后的原图坐标

            // 遍历所有检测框进行坐标转换
            for (size_t i = 0; i < det_result->boxes.size(); i++) {
                // 原始crop图坐标（基于1440×1080裁剪图）
                float x1_crop = det_result->boxes[i][0];  // 左上角x
                float y1_crop = det_result->boxes[i][1];  // 左上角y
                float x2_crop = det_result->boxes[i][2];  // 右下角x
                float y2_crop = det_result->boxes[i][3];  // 右下角y

                // 转换到原图坐标（x坐标加上裁剪偏移，y坐标不变）
                float x1_orig = x1_crop + crop_offset_x;  // 加上裁剪偏移量240
                float y1_orig = y1_crop;
                float x2_orig = x2_crop + crop_offset_x;
                float y2_orig = y2_crop;

                // 保存原图坐标用于OSD绘制
                boxes_original_coord.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
            }

            /**********************************************************************************
             * 3.3 OSD绘图：使用原图坐标在OSD上绘制人脸检测框
             **********************************************************************************/
            visualizer.Draw(boxes_original_coord);
        }
        else {
            // 未检测到人脸，清除OSD上的检测框
            cout << "[INFO] No face detected" << endl;
            std::vector<std::array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);  // 传入空向量清除显示
        }

        num_frames += 1;  // 帧计数器递增

        // OSD 贴图
        osd_index = (num_frames / 10) % 3;
        visualizer.DrawBitmap(osds[osd_index].filename, "shared_colorLUT.sscl", osds[osd_index].x, osds[osd_index].y, 2);
    }

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/

    delete det_result;  // 释放检测结果
    detector.Release();  // 释放检测器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}

