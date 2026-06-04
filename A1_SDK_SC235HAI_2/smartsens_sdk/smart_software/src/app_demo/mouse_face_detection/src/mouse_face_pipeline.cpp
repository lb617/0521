/*
 * @Filename: mouse_face_pipeline.cpp
 * @Description: 小鼠面部检测三模型串行流水线实现
 *
 * 流水线: 质量评估 → 面部检测 → 特征打分
 *
 * 模型规格:
 *   frame_quality:  1×224×224 → 1 float (0~1)
 *   face_cropper:   3×256×256 → 5 floats [x1,y1,x2,y2,conf]
 *   grimace_scorer: 1×256×224 → 20 floats (5 特征各 4 个分值)
 */
#include "../include/common.hpp"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

// IMAGEPROCESSOR 实现位于 pipeline_image.cpp，此处不重复定义。

// 辅助函数: 检查文件是否存在
static bool file_exists(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0);
}

// ---------- MOUSE_FACE_PIPELINE ----------

void MOUSE_FACE_PIPELINE::Initialize(
    std::array<int, 2>* in_img_shape,
    const std::string& quality_model,
    const std::string& detect_model,
    const std::string& score_model)
{
    img_width  = (*in_img_shape)[0];
    img_height = (*in_img_shape)[1];

    printf("[INFO] MOUSE_FACE_PIPELINE: initializing with img=%dx%d\n", img_width, img_height);

    // ---------- 1. 加载三个模型 ----------

    if (!file_exists(quality_model.c_str())) {
        fprintf(stderr, "[ERROR] Model file not found: %s\n", quality_model.c_str());
        return;
    }
    model_quality = ssne_loadmodel(const_cast<char*>(quality_model.c_str()), SSNE_STATIC_ALLOC);
    printf("[INFO] Loaded frame_quality model, id=%u\n", model_quality);

    if (!file_exists(detect_model.c_str())) {
        fprintf(stderr, "[ERROR] Model file not found: %s\n", detect_model.c_str());
        return;
    }
    model_detect = ssne_loadmodel(const_cast<char*>(detect_model.c_str()), SSNE_STATIC_ALLOC);
    printf("[INFO] Loaded face_cropper model, id=%u\n", model_detect);

    if (!file_exists(score_model.c_str())) {
        fprintf(stderr, "[ERROR] Model file not found: %s\n", score_model.c_str());
        return;
    }
    model_score = ssne_loadmodel(const_cast<char*>(score_model.c_str()), SSNE_STATIC_ALLOC);
    printf("[INFO] Loaded grimace_scorer model, id=%u\n", model_score);

    // ---------- 2. 创建输入 tensor ----------

    // frame_quality: 1-ch 灰阶 224×224
    input_quality = create_tensor(224, 224, SSNE_Y_8, SSNE_BUF_AI);
    // face_cropper: 3-ch RGB 256×256
    input_detect  = create_tensor(256, 256, SSNE_RGB, SSNE_BUF_AI);
    // grimace_scorer: 1-ch 灰阶 256×224
    input_score   = create_tensor(256, 224, SSNE_Y_8, SSNE_BUF_AI);

    // ---------- 3. 创建预处理管道 + 从模型加载归一化参数 ----------

    pipe_quality = GetAIPreprocessPipe();
    if (SetNormalize(pipe_quality, model_quality) != 0) {
        printf("[WARN] frame_quality SetNormalize failed, using default\n");
    }

    pipe_detect = GetAIPreprocessPipe();
    if (SetNormalize(pipe_detect, model_detect) != 0) {
        printf("[WARN] face_cropper SetNormalize failed, using default\n");
    }

    pipe_score = GetAIPreprocessPipe();
    if (SetNormalize(pipe_score, model_score) != 0) {
        printf("[WARN] grimace_scorer SetNormalize failed, using default\n");
    }

    printf("[INFO] MOUSE_FACE_PIPELINE: initialization complete\n");
}


bool MOUSE_FACE_PIPELINE::ProcessFrame(ssne_tensor_t* img, MouseFaceResult* result)
{
    result->Clear();

    // ============================================================
    // Step 1: 图像质量评估
    //   输入: 传感器 YUV422_16 帧
    //   预处理: YUV→Y8 灰阶, resize 224×224
    //   推理: 1 float 质量分 (0~1)
    // ============================================================
    int ret = RunAiPreprocessPipe(pipe_quality, *img, input_quality);
    if (ret != 0) {
        printf("[ERROR] frame_quality preprocess failed! ret=%d\n", ret);
        return false;
    }

    if (ssne_inference(model_quality, 1, &input_quality) != 0) {
        fprintf(stderr, "[ERROR] frame_quality inference failed!\n");
        return false;
    }

    ssne_getoutput(model_quality, 1, &output_quality);
    float quality_score = *(float*)get_data(output_quality);
    result->quality_score = quality_score;

    // 质量检查 (调试: 始终打印分值)
    static int q_skip_count = 0;
    if (quality_score < quality_threshold) {
        if (++q_skip_count % 30 == 1)
            printf("[DEBUG] quality=%.4f < threshold(%.2f), skip (x%d)\n",
                   quality_score, quality_threshold, q_skip_count);
        result->valid = false;
        return false;
    }

    // ============================================================
    // Step 2: 面部边界框检测
    //   输入: 传感器 YUV422_16 帧
    //   预处理: YUV→RGB, resize 256×256
    //   推理: 5 floats [x1, y1, x2, y2, confidence]
    // ============================================================
    ret = RunAiPreprocessPipe(pipe_detect, *img, input_detect);
    if (ret != 0) {
        printf("[ERROR] face_cropper preprocess failed! ret=%d\n", ret);
        return false;
    }

    if (ssne_inference(model_detect, 1, &input_detect) != 0) {
        fprintf(stderr, "[ERROR] face_cropper inference failed!\n");
        return false;
    }

    ssne_getoutput(model_detect, 1, &output_detect);
    float* det = (float*)get_data(output_detect);

    float x1 = det[0];
    float y1 = det[1];
    float x2 = det[2];
    float y2 = det[3];
    float det_conf = det[4];

    // 调试: 每 300 帧打印一次原始数值
    static int dbg_count = 0;
    dbg_count++;
    if (dbg_count % 300 == 1) {
        printf("[DEBUG] quality=%.4f det_raw=[%.3f,%.3f,%.3f,%.3f] det_conf=%.4f\n",
               quality_score, x1, y1, x2, y2, det_conf);
    }

    // 置信度检查
    if (det_conf < det_confidence_threshold) {
        // 每 5 次通过质量的帧里打印一次低置信度信息
        static int dc_skip = 0;
        if (++dc_skip % 5 == 1)
            printf("[DEBUG] det_conf=%.4f < threshold(%.2f), bbox_raw=[%.3f,%.3f,%.3f,%.3f] (skip x%d)\n",
                   det_conf, det_confidence_threshold, x1, y1, x2, y2, dc_skip);
        result->valid = false;
        return false;
    }

    // 坐标转换: 模型输出 0~1 归一化坐标 → 传感器像素坐标
    // 注意: face_cropper 的输出是直接归一化值 (0~1), 不是相对于 256 的像素值
    float fx1 = x1 * static_cast<float>(img_width);
    float fy1 = y1 * static_cast<float>(img_height);
    float fx2 = x2 * static_cast<float>(img_width);
    float fy2 = y2 * static_cast<float>(img_height);

    // 确保 x1<x2, y1<y2 (模型可能输出顺序反转)
    if (fx1 > fx2) std::swap(fx1, fx2);
    if (fy1 > fy2) std::swap(fy1, fy2);

    // 边界 clamp
    if (fx1 < 0.0f) fx1 = 0.0f;
    if (fy1 < 0.0f) fy1 = 0.0f;
    if (fx2 > static_cast<float>(img_width))  fx2 = static_cast<float>(img_width);
    if (fy2 > static_cast<float>(img_height)) fy2 = static_cast<float>(img_height);

    // 填充检测结果 (打分可能不成功，先填入现在已拿到的检测数据)
    result->x1         = fx1;
    result->y1         = fy1;
    result->x2         = fx2;
    result->y2         = fy2;
    result->confidence = det_conf;

    // ============================================================
    // Step 3: 面部特征打分
    //   输入: 传感器 YUV422_16 帧 (裁剪到面部区域)
    //   预处理: SetCrop(face_bbox), YUV→Y8 灰阶, resize 256×224
    //   推理: 20 floats (5 特征 × 4 档置信度)
    // ============================================================

    // 设置裁剪区域 (基于传感器原图坐标)
    uint16_t crop_x1 = static_cast<uint16_t>(fx1);
    uint16_t crop_y1 = static_cast<uint16_t>(fy1);
    uint16_t crop_x2 = static_cast<uint16_t>(fx2);
    uint16_t crop_y2 = static_cast<uint16_t>(fy2);

    // 确保裁剪区域有效，无效则跳过打分支 (检测数据仍然有效)
    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1 ||
        (crop_x2 - crop_x1) < 4 || (crop_y2 - crop_y1) < 4) {
        printf("[WARN] Invalid crop region: (%d,%d)-(%d,%d), skip grimace scoring\n",
               crop_x1, crop_y1, crop_x2, crop_y2);
        // 跳过打分但检测框有效
        for (int i = 0; i < 5; i++) result->grimace_levels[i] = -1;
        result->valid = true;
        return true;
    }

    SetCrop(pipe_score, crop_x1, crop_y1, crop_x2, crop_y2);

    ret = RunAiPreprocessPipe(pipe_score, *img, input_score);
    if (ret != 0) {
        printf("[ERROR] grimace_scorer preprocess failed! ret=%d\n", ret);
        return false;
    }

    if (ssne_inference(model_score, 1, &input_score) != 0) {
        fprintf(stderr, "[ERROR] grimace_scorer inference failed!\n");
        return false;
    }

    ssne_getoutput(model_score, 1, &output_score);
    float* raw_scores = (float*)get_data(output_score);

    // 20 个输出 = 5 特征 × 4 档(0,1,2,3)置信度
    // 每个特征取置信度最高的那一档作为结果
    for (int i = 0; i < 5; i++) {
        int best_level = 0;
        float best_conf = raw_scores[i * 4];
        for (int j = 1; j < 4; j++) {
            if (raw_scores[i * 4 + j] > best_conf) {
                best_conf = raw_scores[i * 4 + j];
                best_level = j;
            }
        }
        result->grimace_levels[i] = best_level;
    }

    result->valid = true;
    return true;
}


void MOUSE_FACE_PIPELINE::Release()
{
    printf("[INFO] MOUSE_FACE_PIPELINE: releasing resources...\n");

    // 释放输入 tensor
    release_tensor(input_quality);
    release_tensor(input_detect);
    release_tensor(input_score);

    // 释放输出 tensor
    release_tensor(output_quality);
    release_tensor(output_detect);
    release_tensor(output_score);

    // 释放预处理管道
    ReleaseAIPreprocessPipe(pipe_quality);
    ReleaseAIPreprocessPipe(pipe_detect);
    ReleaseAIPreprocessPipe(pipe_score);

    printf("[INFO] MOUSE_FACE_PIPELINE: released\n");
}
