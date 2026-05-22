/*

 * @Description: 鼠脸检测三模型流水线演示程序
 *   pipeline: quality → cropper → grimace scorer
 *
 * 模型1 - frame_quality_lite:  224×224 灰度输入, 输出 1 个质量分数 (0~1)
 * 模型2 - face_cropper_lite:   256×256 BGR 输入, 输出 [conf, x1, y1, x2, y2] (归一化)
 * 模型3 - grimace_scorer_lite: 224×224 BGR 输入, 输出 20 个 logits (5器官×4分类)
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
#include <cmath>
#include <termios.h>
#include "include/utils.hpp"

using namespace std;

// 全局退出标志（线程安全）
bool g_exit_flag = false;
std::mutex g_mtx;

// 5 个器官名称（与训练脚本 ORGAN_COLUMNS 一致）
static const char* ORGAN_NAMES[] = {"eye", "nose", "cheek", "ear", "whisker"};
static const int NUM_ORGANS = 5;
static const int NUM_CLASSES = 4;  // 每器官 0~3 分

// UART 通信配置（GPIO2 复用为 UART TX，与下位机通信）
static const char* UART_DEVICE = "/dev/uartdev";  // GPIO2 复用的 UART 设备节点
static const int UART_BAUD = B115200;            // 波特率 115200
static const int GRIMACE_ALERT_THRESHOLD = 7;    // 表情总分 ≥ 7 时触发报警信号

/**
 * @brief 初始化 UART 通信（GPIO2 复用为 UART）
 * @return UART 文件描述符，失败返回 -1
 */
static int uart_init() {
    int fd = open(UART_DEVICE, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        fprintf(stderr, "[UART] Failed to open %s: %s\n", UART_DEVICE, strerror(errno));
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, UART_BAUD);
    cfsetospeed(&options, UART_BAUD);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= CLOCAL | CREAD;

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 1;

    tcsetattr(fd, TCSANOW, &options);
    tcflush(fd, TCIOFLUSH);

    printf("[UART] Initialized: %s (GPIO2), baud=%d\n", UART_DEVICE, 115200);
    return fd;
}

/**
 * @brief 通过 UART 发送信号给下位机
 * @param fd UART 文件描述符
 */
static void uart_send_signal(int fd) {
    if (fd < 0) return;
    const char* signal = "1";
    ssize_t written = write(fd, signal, 1);
    if (written < 0) {
        fprintf(stderr, "[UART] Write failed: %s\n", strerror(errno));
    }
}

/**
 * @brief 键盘监听线程，输入 'q' 退出程序
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;
    while (true) {
        std::cin >> input;
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
 * @brief 线程安全地检查退出标志
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief 对 logits 数组做 softmax（原地）
 * @param arr 长度为 len 的浮点数组
 */
static void softmax_inplace(float* arr, int len) {
    float max_val = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        arr[i] = expf(arr[i] - max_val);
        sum += arr[i];
    }
    for (int i = 0; i < len; i++) {
        arr[i] /= sum;
    }
}

/**
 * @brief 鼠脸检测三模型流水线主函数
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    // 图像尺寸配置
    int img_width = 1920;
    int img_height = 1080;
    array<int, 2> img_shape = {img_width, img_height};

    // 模型路径
    string path_quality = "/app_demo/app_assets/models/frame_quality_lite.m1model";
    string path_cropper = "/app_demo/app_assets/models/face_cropper_lite.m1model";
    string path_grimace = "/app_demo/app_assets/models/grimace_scorer_lite.m1model";

    // 阈值配置
    const float QUALITY_THRESHOLD = 4.0f / 6.0f;  // 质量阈值 (对应原始分数 4/6 ≈ 0.667)
    const float CROPPER_CONF_THRESHOLD = 0.5f;     // 检测置信度阈值

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    // SSNE 引擎初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    // 图像处理器初始化（获取完整 1920×1080 图像）
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    // 加载三个模型
    uint16_t model_quality = ssne_loadmodel(
        const_cast<char*>(path_quality.c_str()), SSNE_STATIC_ALLOC);
    uint16_t model_cropper = ssne_loadmodel(
        const_cast<char*>(path_cropper.c_str()), SSNE_STATIC_ALLOC);
    uint16_t model_grimace = ssne_loadmodel(
        const_cast<char*>(path_grimace.c_str()), SSNE_STATIC_ALLOC);

    printf("[INFO] Models loaded: quality=%d cropper=%d grimace=%d\n",
           model_quality, model_cropper, model_grimace);

    // 创建模型输入 tensor
    // quality: 224×224 灰度 (SSNE_Y_8)
    ssne_tensor_t input_quality = create_tensor(224, 224, SSNE_Y_8, SSNE_BUF_AI);
    // cropper: 256×256 BGR
    ssne_tensor_t input_cropper = create_tensor(256, 256, SSNE_BGR, SSNE_BUF_AI);
    // grimace: 224×224 BGR
    ssne_tensor_t input_grimace = create_tensor(224, 224, SSNE_BGR, SSNE_BUF_AI);

    // 模型输出 tensor（声明即可，ssne_getoutput 会填充；只在程序结束时释放）
    ssne_tensor_t output_quality[1];
    ssne_tensor_t output_cropper[1];
    ssne_tensor_t output_grimace[1];

    // 创建 AI 预处理管道（每个模型独立管道）
    AiPreprocessPipe pipe_quality = GetAIPreprocessPipe();
    AiPreprocessPipe pipe_cropper = GetAIPreprocessPipe();
    AiPreprocessPipe pipe_grimace = GetAIPreprocessPipe();

    // 从模型加载归一化参数
    SetNormalize(pipe_quality, model_quality);
    SetNormalize(pipe_cropper, model_cropper);
    SetNormalize(pipe_grimace, model_grimace);

    // quality 模型预处理：中心裁方 → resize 224×224
    // 1920×1080 中心正方形 = 1080×1080, 起点 x=(1920-1080)/2=420
    SetCrop(pipe_quality, 420, 0, 1500, 1080);

    // cropper 模型预处理：全图 → resize 256×256
    // 喂入完整 1920×1080 图像，让模型在全局检测鼠脸
    SetCrop(pipe_cropper, 0, 0, 1920, 1080);

    // grimace 模型预处理：初始全图裁剪（运行时每帧根据检测结果动态更新）
    SetCrop(pipe_grimace, 0, 0, 1920, 1080);

    // OSD 可视化器初始化
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

    // UART 初始化（GPIO2 复用为 UART，与下位机通信）
    int uart_fd = uart_init();

    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);

    // 帧计数器
    uint32_t num_frames = 0;
    uint32_t num_quality_pass = 0;
    uint32_t num_face_detected = 0;

    // 图像 tensor
    ssne_tensor_t img_sensor;

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    while (!check_exit_flag()) {

        // 从 sensor 获取完整图像（1920×1080 YUV422）
        processor.GetImage(&img_sensor);

        // ============================================================
        // Step 1: 帧质量检测 (frame_quality_lite)
        // ============================================================
        int ret_q = RunAiPreprocessPipe(pipe_quality, img_sensor, input_quality);
        if (ret_q != 0) {
            fprintf(stderr, "[ERROR] quality preprocess failed! ret=%d\n", ret_q);
            continue;
        }
        if (ssne_inference(model_quality, 1, &input_quality)) {
            fprintf(stderr, "[ERROR] quality inference failed!\n");
            continue;
        }

        ssne_getoutput(model_quality, 1, output_quality);
        float quality_score = ((float*)get_data(output_quality[0]))[0];

        bool frame_usable = (quality_score >= QUALITY_THRESHOLD);
        printf("[Frame %u] Quality=%.4f %s\n",
               num_frames, quality_score, frame_usable ? "PASS" : "REJECT");

        if (!frame_usable) {
            num_frames++;
            // 清除 OSD 检测框
            std::vector<std::array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);
            continue;
        }
        num_quality_pass++;

        // ============================================================
        // Step 2: 鼠脸检测 (face_cropper_lite)
        // ============================================================
        int ret_c = RunAiPreprocessPipe(pipe_cropper, img_sensor, input_cropper);
        if (ret_c != 0) {
            fprintf(stderr, "[ERROR] cropper preprocess failed! ret=%d\n", ret_c);
            num_frames++;
            continue;
        }
        if (ssne_inference(model_cropper, 1, &input_cropper)) {
            fprintf(stderr, "[ERROR] cropper inference failed!\n");
            num_frames++;
            continue;
        }

        ssne_getoutput(model_cropper, 1, output_cropper);
        float* cropper_data = (float*)get_data(output_cropper[0]);
        float conf = cropper_data[0];
        float x1_norm = cropper_data[1];   // 左上 x（归一化）
        float y1_norm = cropper_data[2];   // 左上 y（归一化）
        float x2_norm = cropper_data[3];   // 右下 x（归一化）
        float y2_norm = cropper_data[4];   // 右下 y（归一化）

        // 模型输出角点可能未排序，取 min/max 确保 x1<x2, y1<y2
        if (x1_norm > x2_norm) { float tmp = x1_norm; x1_norm = x2_norm; x2_norm = tmp; }
        if (y1_norm > y2_norm) { float tmp = y1_norm; y1_norm = y2_norm; y2_norm = tmp; }

        printf("[Frame %u] Cropper: conf=%.4f bbox_xyxy=[%.3f, %.3f, %.3f, %.3f]\n",
               num_frames, conf, x1_norm, y1_norm, x2_norm, y2_norm);

        if (conf < CROPPER_CONF_THRESHOLD) {
            printf("[Frame %u] No face detected (conf=%.4f < %.2f)\n",
                   num_frames, conf, CROPPER_CONF_THRESHOLD);
            num_frames++;
            std::vector<std::array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);
            continue;
        }
        num_face_detected++;

        // xyxy 归一化 → 像素坐标（1920×1080）
        float face_x1 = x1_norm * img_width;
        float face_y1 = y1_norm * img_height;
        float face_x2 = x2_norm * img_width;
        float face_y2 = y2_norm * img_height;

        // 边界裁剪 & 确保有效性
        face_x1 = fmaxf(0.0f, fminf(face_x1, (float)img_width - 1));
        face_y1 = fmaxf(0.0f, fminf(face_y1, (float)img_height - 1));
        face_x2 = fmaxf(face_x1 + 1.0f, fminf(face_x2, (float)img_width));
        face_y2 = fmaxf(face_y1 + 1.0f, fminf(face_y2, (float)img_height));

        // 确保裁剪宽度能被 8 整除（预处理管线要求）
        float face_w = face_x2 - face_x1;
        float face_h = face_y2 - face_y1;
        int face_w_aligned = ((int)face_w / 8) * 8;
        if (face_w_aligned < 8) face_w_aligned = 8;
        face_x2 = face_x1 + face_w_aligned;
        if (face_x2 > img_width) {
            face_x2 = img_width;
            face_x1 = face_x2 - face_w_aligned;
        }

        // 在 OSD 上绘制检测框
        std::vector<std::array<float, 4>> boxes = {
            {face_x1, face_y1, face_x2, face_y2}
        };
        visualizer.Draw(boxes);

        // ============================================================
        // Step 3: 鼠脸表情打分 (grimace_scorer_lite)
        // ============================================================
        // 动态设置裁剪区域为检测到的鼠脸
        SetCrop(pipe_grimace,
                (uint16_t)face_x1, (uint16_t)face_y1,
                (uint16_t)face_x2, (uint16_t)face_y2);
        int ret_g = RunAiPreprocessPipe(pipe_grimace, img_sensor, input_grimace);
        if (ret_g != 0) {
            fprintf(stderr, "[ERROR] grimace preprocess failed! ret=%d\n", ret_g);
            num_frames++;
            continue;
        }

        if (ssne_inference(model_grimace, 1, &input_grimace)) {
            fprintf(stderr, "[ERROR] grimace inference failed!\n");
            num_frames++;
            continue;
        }

        ssne_getoutput(model_grimace, 1, output_grimace);
        float* grimace_data = (float*)get_data(output_grimace[0]);

        // 输出形状 [1, 20, 1, 1] → reshape 为 [5, 4] → argmax
        int total_score = 0;
        printf("[Frame %u] Grimace scores: ", num_frames);
        for (int i = 0; i < NUM_ORGANS; i++) {
            // 对每个器官的 4 个 logits 做 softmax，然后取 argmax
            float organ_logits[4];
            for (int j = 0; j < NUM_CLASSES; j++) {
                organ_logits[j] = grimace_data[i * NUM_CLASSES + j];
            }
            softmax_inplace(organ_logits, NUM_CLASSES);

            int max_class = 0;
            float max_prob = organ_logits[0];
            for (int j = 1; j < NUM_CLASSES; j++) {
                if (organ_logits[j] > max_prob) {
                    max_prob = organ_logits[j];
                    max_class = j;
                }
            }
            total_score += max_class;
            printf("%s=%d(%.2f) ", ORGAN_NAMES[i], max_class, max_prob);
        }
        printf("| TOTAL=%d\n", total_score);

        // 总分 ≥ 7 时：串口打印报警 + UART 发信号给下位机
        if (total_score >= GRIMACE_ALERT_THRESHOLD) {
            printf("[ALERT] TOTAL=%d >= %d, stimulate evacuation!\n",
                   total_score, GRIMACE_ALERT_THRESHOLD);
            uart_send_signal(uart_fd);
        }

        num_frames++;
    }

    // 等待监听线程退出
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/

    printf("\n[STATS] Total frames: %u | Quality pass: %u | Face detected: %u\n",
           num_frames, num_quality_pass, num_face_detected);

    // 释放输入 tensor
    release_tensor(input_quality);
    release_tensor(input_cropper);
    release_tensor(input_grimace);

    // 释放输出 tensor
    release_tensor(output_quality[0]);
    release_tensor(output_cropper[0]);
    release_tensor(output_grimace[0]);

    // 释放 AI 预处理管道
    ReleaseAIPreprocessPipe(pipe_quality);
    ReleaseAIPreprocessPipe(pipe_cropper);
    ReleaseAIPreprocessPipe(pipe_grimace);

    // 释放可视化器
    visualizer.Release();

    // 关闭 UART
    if (uart_fd >= 0) {
        close(uart_fd);
        printf("[UART] Closed\n");
    }

    // 释放图像处理器
    processor.Release();

    // 释放 SSNE 引擎
    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    printf("[INFO] Program exited successfully.\n");
    return 0;
}

