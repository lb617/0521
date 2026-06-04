/*
 * @Filename: common.hpp
 * @Description: Mouse face detection pipeline — data structures and model classes
 */
#pragma once

#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <math.h>
#include "smartsoc/ssne_api.h"

/*!
 * @brief 小鼠面部检测完整结果
 */
struct MouseFaceResult {
    float x1, y1, x2, y2;        // 面部边界框 (传感器原图坐标)
    float confidence;             // 面部检测置信度
    float quality_score;          // 图像质量评分 (0~1)
    int grimace_levels[5];        // 5 种特征的判定档位 (0~3), 取每特征 4 档中置信度最高者
    bool valid;                   // 此帧是否通过了质量检查 + 检测

    MouseFaceResult() {
        x1 = y1 = x2 = y2 = 0.0f;
        confidence = 0.0f;
        quality_score = 0.0f;
        for (int i = 0; i < 5; i++) grimace_levels[i] = 0;
        valid = false;
    }

    void Clear() {
        x1 = y1 = x2 = y2 = 0.0f;
        confidence = 0.0f;
        quality_score = 0.0f;
        for (int i = 0; i < 5; i++) grimace_levels[i] = 0;
        valid = false;
    }
};

/*!
 * @brief 图像采集器 — 从传感器 Online Pipeline 获取帧
 *
 * 与现有人脸检测 demo 中的 IMAGEPROCESSOR 完全相同。
 */
class IMAGEPROCESSOR {
public:
    void Initialize(std::array<int, 2>* in_img_shape);
    void GetImage(ssne_tensor_t* img_sensor);
    void Release();

    std::array<int, 2> img_shape;

private:
    uint8_t format_online;
};


/*!
 * @brief 小鼠面部检测流水线
 *
 * 串行执行三个模型的推理：
 *   1. frame_quality  → 图像质量评估 (灰阶 224×224 → 1 float)
 *   2. face_cropper    → 面部边界框检测 (RGB 256×256 → 5 floats)
 *   3. grimace_scorer  → 面部特征打分 (灰阶 256×224 → 20 floats)
 */
class MOUSE_FACE_PIPELINE {
public:
    MOUSE_FACE_PIPELINE()
        : model_quality(0), model_detect(0), model_score(0)
        , pipe_quality(nullptr), pipe_detect(nullptr), pipe_score(nullptr)
        , img_width(1920), img_height(1080)
        , quality_threshold(0.5f), det_confidence_threshold(0.6f)
    {}

    /*!
     * @brief 初始化流水线：加载模型、创建 tensor、创建预处理管道
     * @param in_img_shape   传感器原图尺寸 [width, height]
     * @param quality_model  图像质量模型路径
     * @param detect_model   面部检测模型路径
     * @param score_model    特征打分模型路径
     */
    void Initialize(std::array<int, 2>* in_img_shape,
                    const std::string& quality_model,
                    const std::string& detect_model,
                    const std::string& score_model);

    /*!
     * @brief 处理一帧图像
     * @param img    传感器帧 (YUV422_16, 1920×1080)
     * @param result 输出结果
     * @return true  如果检测到了有效的小鼠面部
     */
    bool ProcessFrame(ssne_tensor_t* img, MouseFaceResult* result);

    /*! @brief 释放所有资源 */
    void Release();

private:
    // 三个模型 ID
    uint16_t model_quality;
    uint16_t model_detect;
    uint16_t model_score;

    // 输入 tensor (每个模型一个)
    ssne_tensor_t input_quality;
    ssne_tensor_t input_detect;
    ssne_tensor_t input_score;

    // 输出 tensor
    ssne_tensor_t output_quality;
    ssne_tensor_t output_detect;
    ssne_tensor_t output_score;

    // 预处理管道 (每个模型一个)
    AiPreprocessPipe pipe_quality;
    AiPreprocessPipe pipe_detect;
    AiPreprocessPipe pipe_score;

    // 传感器原图尺寸
    int img_width;
    int img_height;

    // 可配置阈值
    float quality_threshold;
    float det_confidence_threshold;
};
