/*
 * @Filename: demo_mouse_face.cpp
 * @Description: 小鼠面部检测主程序
 *
 * 串行流水线: 图像采集 → 质量评估 → 面部检测 → 特征打分 → OSD 可视化
 *
 * 三个模型:
 *   frame_quality_lite.m1model  — 图像质量评估 (0~1 分值)
 *   face_cropper_lite.m1model   — 面部边界框检测 [x1,y1,x2,y2,conf]
 *   grimace_scorer_lite.m1model — 5 种面部特征打分 (各 4 子分值)
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <unistd.h>
#include "include/utils.hpp"

using namespace std;

// ============================================================
//  全局退出标志
// ============================================================
bool g_exit_flag = false;
std::mutex g_mtx;

void keyboard_listener() {
    std::string input;
    std::cout << "Keyboard listener started. Press 'q' to exit..." << std::endl;
    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "Exit requested, notifying main thread..." << std::endl;
            break;
        }
    }
}

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}


// ============================================================
//  OSD 位图信息
// ============================================================
struct osdInfo {
    std::string filename;
    uint16_t x;
    uint16_t y;
};


// ============================================================
//  main
// ============================================================
int main() {
    // ---- 1. 参数配置 ----
    int img_width  = 1920;
    int img_height = 1080;

    // 模型路径 (运行时位于 /app_demo/app_assets/models/)
    string path_quality = "/app_demo/app_assets/models/frame_quality_lite.m1model";
    string path_detect  = "/app_demo/app_assets/models/face_cropper_lite.m1model";
    string path_score   = "/app_demo/app_assets/models/grimace_scorer_lite.m1model";

    // OSD 位图
    static osdInfo osds[3] = {
        {"si.ssbmp", 10, 10},
        {"te.ssbmp", 90, 10},
        {"wei.ssbmp", 170, 10}
    };

    // ---- 2. SSNE 初始化 ----
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }
    printf("[INFO] SSNE initialized\n");

    // ---- 3. 图像采集器初始化 ----
    array<int, 2> img_shape = {img_width, img_height};
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    // ---- 4. 小鼠面部流水线初始化 (加载 3 个模型) ----
    MOUSE_FACE_PIPELINE pipeline;
    pipeline.Initialize(&img_shape, path_quality, path_detect, path_score);

    // ---- 5. 检测结果 ----
    MouseFaceResult det_result;

    // ---- 6. OSD 可视化器初始化 ----
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

    // ---- 7. 系统稳定等待 ----
    cout << "Waiting 0.2s for system stability..." << endl;
    usleep(200000);

    // 初始 OSD 位图
    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    // ---- 8. 启动键盘监听线程 ----
    std::thread listener_thread(keyboard_listener);

    // ---- 9. 主循环 ----
    uint16_t num_frames = 0;
    uint8_t  osd_index  = 0;
    ssne_tensor_t img_sensor;

    while (!check_exit_flag()) {

        // 9a. 获取传感器帧
        processor.GetImage(&img_sensor);

        // 9b. 执行三模型流水线
        bool detected = pipeline.ProcessFrame(&img_sensor, &det_result);

        // 9c. OSD 可视化
        if (detected && det_result.valid) {
            // 绘制检测框 (Layer 0)
            vector<array<float, 4>> boxes;
            boxes.push_back({det_result.x1, det_result.y1,
                             det_result.x2, det_result.y2});
            visualizer.Draw(boxes);

            // 绘制特征档位指示器 (Layer 1), 仅当打分成功时
            bool has_scores = (det_result.grimace_levels[0] >= 0);
            if (has_scores) {
                visualizer.DrawGrimaceScores(det_result.grimace_levels);
            }

            // 打印到控制台 (调试用)
            if (has_scores) {
                printf("[FRAME %d] quality=%.3f conf=%.3f bbox=[%.0f,%.0f,%.0f,%.0f] levels=[%d,%d,%d,%d,%d]\n",
                       num_frames,
                       det_result.quality_score,
                       det_result.confidence,
                       det_result.x1, det_result.y1, det_result.x2, det_result.y2,
                       det_result.grimace_levels[0],
                       det_result.grimace_levels[1],
                       det_result.grimace_levels[2],
                       det_result.grimace_levels[3],
                       det_result.grimace_levels[4]);
            } else {
                printf("[FRAME %d] quality=%.3f conf=%.3f bbox=[%.0f,%.0f,%.0f,%.0f] levels=skip(crop_invalid)\n",
                       num_frames,
                       det_result.quality_score,
                       det_result.confidence,
                       det_result.x1, det_result.y1, det_result.x2, det_result.y2);
            }

        } else {
            // 没有检测到 → 清除 Layer 0 和 Layer 1
            vector<array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);
        }

        num_frames++;

        // 9d. OSD 位图切换 (每 10 帧)
        osd_index = (num_frames / 10) % 3;
        visualizer.DrawBitmap(osds[osd_index].filename, "shared_colorLUT.sscl",
                              osds[osd_index].x, osds[osd_index].y, 2);
    }

    // ---- 10. 资源释放 ----
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    pipeline.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    printf("[INFO] Program exited normally\n");
    return 0;
}
