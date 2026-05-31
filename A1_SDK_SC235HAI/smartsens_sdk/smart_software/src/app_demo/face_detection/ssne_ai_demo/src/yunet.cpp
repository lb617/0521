/*
 * @Filename: yunet.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-01-20
 * @Copyright (c) 2025 SmartSens
 * @Description: YuNet三通道人脸检测实现文件
 */
#include <assert.h>
#include "../include/utils.hpp"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

// YuNet模型每个anchor输出17维: loc(14) + conf(2) + iou(1)
static const int OUTPUT_DIM_PER_ANCHOR = 17;
static const int LOC_DIM = 14;      // bbox(4) + landmarks(10)
static const int CONF_DIM = 2;      // 二分类
static const int IOU_DIM = 1;       // iou预测

// 保存图像使用
static ssne_tensor_t g_last_img;
static ssne_tensor_t g_last_pipe_input;
static bool g_has_frame = false;

/**
 * @brief 生成YuNet的先验框(priors)
 * @description 根据特征图尺寸和anchor配置生成所有先验框
 */
void YUNET::GeneratePriors() { 
    int w = det_shape[0]; // 检测图像宽度 
    int h = det_shape[1]; // 检测图像高度 

    // 计算各层特征图尺寸 
    // feature_map_2th = (h+1)/2/2, (w+1)/2/2 
    int feature_map_2th_h = ((h + 1) / 2) / 2; 
    int feature_map_2th_w = ((w + 1) / 2) / 2; 
    
    // feature_map_3th = feature_map_2th / 2 
    int feature_map_3th_h = feature_map_2th_h / 2; 
    int feature_map_3th_w = feature_map_2th_w / 2; 
    
    // feature_map_4th = feature_map_3th / 2 
    int feature_map_4th_h = feature_map_3th_h / 2; 
    int feature_map_4th_w = feature_map_3th_w / 2; 
    
    // feature_map_5th = feature_map_4th / 2 
    int feature_map_5th_h = feature_map_4th_h / 2; 
    int feature_map_5th_w = feature_map_4th_w / 2; 
    
    // feature_map_6th = feature_map_5th / 2 
    int feature_map_6th_h = feature_map_5th_h / 2; 
    int feature_map_6th_w = feature_map_5th_w / 2; 
    
    // YuNet使用4层特征图: 3th, 4th, 5th, 6th 
    std::vector<std::array<int, 2>> feature_maps = {
         {feature_map_3th_h, feature_map_3th_w}, 
         {feature_map_4th_h, feature_map_4th_w}, 
         {feature_map_5th_h, feature_map_5th_w}, 
         {feature_map_6th_h, feature_map_6th_w} 
        }; 
        
        priors.clear(); 
        
        // 遍历每层特征图 
        for (size_t k = 0; k < feature_maps.size(); k++) {
            int fh = feature_maps[k][0]; // 特征图高度 
            int fw = feature_maps[k][1]; // 特征图宽度 
             
            // 遍历特征图的每个位置 (i, j)  
            for (int i = 0; i < fh; i++) { 
                for (int j = 0; j < fw; j++) { 
                    // 遍历该层的所有min_size 
                    for (size_t m = 0; m < min_sizes[k].size(); m++) { 
                        int min_size = min_sizes[k][m]; 
                        
                        // 计算归一化的anchor尺寸 
                        float s_kx = static_cast<float>(min_size) / static_cast<float>(w); 
                        float s_ky = static_cast<float>(min_size) / static_cast<float>(h); 
                        
                        // 计算归一化的anchor中心坐标 
                        float cx = (j + 0.5f) * steps[k] / static_cast<float>(w); 
                        float cy = (i + 0.5f) * steps[k] / static_cast<float>(h); 
                        
                        // 添加先验框 [cx, cy, s_kx, s_ky] 
                        priors.push_back({cx, cy, s_kx, s_ky}); 
                    } 
                } 
            } 
        } 
        num_priors = static_cast<int>(priors.size()); 
        printf("[INFO] YuNet generated %d priors\n", num_priors); 
    }


/**
 * @brief 对2维的分类分数进行softmax
 * @param input 输入数组 (长度为2)
 * @param length 固定为2
 */
void YUNET::Softmax(float* input, int length) {
    // 找最大值防止溢出
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    
    // 归一化
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

/**
 * @brief 从4个head输出恢复loc/conf/iou
 * @description 将4个head_conv输出按照YuNet格式转换为统一的loc/conf/iou数组
 * 
 * YuNet输出格式（模型转换后为NHWC）：
 * - 每个head输出形状: [N, H, W, C], 其中C = num_anchors_per_position * 17
 * - 遍历顺序: H -> W -> anchor -> 17维输出
 * - Slice: loc(0:14), conf(14:16)->Softmax, iou(16:17)
 */
void YUNET::RestoreOutputsFromHeads(
    const float* head0, const float* head1,
    const float* head2, const float* head3,
    int h0, int w0, int h1, int w1,
    int h2, int w2, int h3, int w3,
    std::vector<float>& loc,
    std::vector<float>& conf,
    std::vector<float>& iou)
{
    constexpr int DIM_PER_ANCHOR = 17;
    constexpr int LOC_DIM  = 14;
    constexpr int CONF_DIM = 2;
    constexpr int IOU_DIM  = 1;

    // 每个 head 的 channel 数（由模型结构决定）
    const int c0 = 51; // 3 * 17
    const int c1 = 34; // 2 * 17
    const int c2 = 34; // 2 * 17
    const int c3 = 51; // 3 * 17

    std::vector<float> flat;
    flat.reserve(
        h0 * w0 * c0 +
        h1 * w1 * c1 +
        h2 * w2 * c2 +
        h3 * w3 * c3
    );

    auto flatten_head = [&](const float* head, int H, int W, int C) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                const int base = i * W * C + j * C;
                for (int c = 0; c < C; ++c) {
                    flat.push_back(head[base + c]);
                }
            }
        }
    };

    flatten_head(head0, h0, w0, c0);
    flatten_head(head1, h1, w1, c1);
    flatten_head(head2, h2, w2, c2);
    flatten_head(head3, h3, w3, c3);

    // === Step 2: reshape(-1, 17) ===
    if (flat.size() % DIM_PER_ANCHOR != 0) {
        fprintf(stderr,
                "[ERROR] RestoreOutputsFromHeads: flat size %zu not divisible by 17\n",
                flat.size());
        return;
    }

    const int num_anchors = static_cast<int>(flat.size() / DIM_PER_ANCHOR);

    loc.resize(num_anchors * LOC_DIM);
    conf.resize(num_anchors * CONF_DIM);
    iou.resize(num_anchors * IOU_DIM);

    // === Step 3: slice + softmax（与 Python 完全一致） ===
    for (int k = 0; k < num_anchors; ++k) {
        const float* p = &flat[k * DIM_PER_ANCHOR];

        // loc: [0:14]
        for (int i = 0; i < LOC_DIM; ++i) {
            loc[k * LOC_DIM + i] = p[i];
        }

        // conf logits: [14:16] → softmax
        float logits[2] = { p[14], p[15] };

        // softmax
        float maxv = std::max(logits[0], logits[1]);
        float e0 = std::exp(logits[0] - maxv);
        float e1 = std::exp(logits[1] - maxv);
        float sum = e0 + e1;
        conf[k * 2 + 0] = e0 / sum;
        conf[k * 2 + 1] = e1 / sum;

        // iou: [16]
        iou[k] = p[16];
    }

}


/**
 * @brief 解码检测结果
 * @description 将网络输出的loc/conf/iou解码为bbox、landmarks和scores
 * 
 * 解码公式（参考Python代码）:
 * - score = sqrt(cls_score * clamp(iou_score, 0, 1))
 * - bbox_center = prior_center + loc[0:2] * variance[0] * prior_size
 * - bbox_wh = prior_size * exp(loc[2:4] * variance)
 * - bbox = [center_x - w/2, center_y - h/2, w, h]
 * - landmarks = prior_center + loc[4:14] * variance[0] * prior_size
 */
void YUNET::Decode(std::vector<float>& loc, std::vector<float>& conf, std::vector<float>& iou,
                   std::vector<std::array<float, 4>>& bboxes, 
                   std::vector<std::array<float, 10>>& landmarks,
                   std::vector<float>& scores) {
    
    int w = det_shape[0];
    int h = det_shape[1];
    
    bboxes.resize(num_priors);
    landmarks.resize(num_priors);
    scores.resize(num_priors);
    
    for (int i = 0; i < num_priors; i++) {
        // 获取先验框参数
        float cx = priors[i][0];      // 归一化中心x
        float cy = priors[i][1];      // 归一化中心y
        float s_kx = priors[i][2];    // 归一化宽度
        float s_ky = priors[i][3];    // 归一化高度
        
        // 计算分数: score = sqrt(cls_score * clamp(iou_score, 0, 1))
        float cls_score = conf[i * CONF_DIM + 1];  // 取正类分数
        float iou_score = iou[i];

        // Clamp iou_score to [0, 1]
        if (iou_score < 0.0f) iou_score = 0.0f;
        if (iou_score > 1.0f) iou_score = 1.0f;

        scores[i] = sqrtf(cls_score * iou_score);

        
        // 解码bbox (归一化坐标)
        float* loc_ptr = &loc[i * LOC_DIM];
        
        // center = prior_center + loc[0:2] * variance[0] * prior_size
        float pred_cx = cx + loc_ptr[0] * variance[0] * s_kx;
        float pred_cy = cy + loc_ptr[1] * variance[0] * s_ky;
        
        // wh = prior_size * exp(loc[2:4] * variance)
        float pred_w = s_kx * expf(loc_ptr[2] * variance[1]);
        float pred_h = s_ky * expf(loc_ptr[3] * variance[1]);
        
        // 转换为绝对坐标 [xmin, ymin, xmax, ymax]
        float xmin = (pred_cx - pred_w / 2.0f) * w;
        float ymin = (pred_cy - pred_h / 2.0f) * h;
        float xmax = (pred_cx + pred_w / 2.0f) * w;
        float ymax = (pred_cy + pred_h / 2.0f) * h;
        
        // 边界裁剪
        xmin = fmaxf(0.0f, xmin);
        ymin = fmaxf(0.0f, ymin);
        xmax = fminf(static_cast<float>(w), xmax);
        ymax = fminf(static_cast<float>(h), ymax);
        
        bboxes[i] = {xmin, ymin, xmax, ymax};
        
        // 解码5个关键点 (每个关键点2维坐标)
        for (int j = 0; j < 5; j++) {
            float lm_x = (cx + loc_ptr[4 + j * 2] * variance[0] * s_kx) * w;
            float lm_y = (cy + loc_ptr[4 + j * 2 + 1] * variance[0] * s_ky) * h;
            landmarks[i][j * 2] = lm_x;
            landmarks[i][j * 2 + 1] = lm_y;
        }
    }
}

/**
 * @brief 后处理函数
 * @description 对解码后的检测结果进行置信度过滤、NMS和尺度恢复
 */
void YUNET::Postprocess(std::vector<std::array<float, 4>>* boxes, 
                        std::vector<std::array<float, 10>>* landmarks,
                        std::vector<float>* scores, 
                        FaceDetectionResult* result, float* conf_threshold) {
    
    size_t num_res = boxes->size();
    
    result->Clear();
    result->landmarks_per_face = 5;  // YuNet输出5个关键点
    result->Reserve(num_res);
    
    int res_count = 0;
    
    // 置信度过滤
    for (size_t i = 0; i < num_res; i++) {
        float score = scores->at(i);
   
        if (score <= *conf_threshold) {
            continue;
        }

        result->boxes.emplace_back(boxes->at(i));
        result->scores.push_back(score);
        // 添加5个关键点
        for (int j = 0; j < 5; j++) {
            std::array<float, 2> lm = {
                landmarks->at(i)[j * 2],
                landmarks->at(i)[j * 2 + 1]
            };
            result->landmarks.emplace_back(lm);
        }
        res_count++;
    }
    result->Resize(res_count);
    
    // 执行NMS
    utils::NMS(result, nms_threshold, top_k);
    
    // 尺度恢复：将检测框坐标从检测尺寸恢复到原始图像尺寸
    res_count = static_cast<int>(result->boxes.size());
    result->Resize(std::min(res_count, keep_top_k));
    
    for (size_t i = 0; i < result->boxes.size(); i++) {
        result->boxes[i][0] *= w_scale;  // x1
        result->boxes[i][1] *= h_scale;  // y1
        result->boxes[i][2] *= w_scale;  // x2
        result->boxes[i][3] *= h_scale;  // y2
    }
    
    // 关键点也需要尺度恢复
    for (size_t i = 0; i < result->landmarks.size(); i++) {
        result->landmarks[i][0] *= w_scale;  // x
        result->landmarks[i][1] *= h_scale;  // y
    }
    
}

/**
 * @brief 初始化YuNet检测器
 */
void YUNET::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, 
                       std::array<int, 2>* in_det_shape, bool in_use_kps,
                       int in_box_len) {
    
    // 设置NMS和top-k参数
    nms_threshold = 0.3f;   // NMS的IoU阈值
    keep_top_k = 30;       // 最终保留的检测框数量
    top_k = 150;           // NMS前保留的检测框数量
    
    img_shape = *in_img_shape;  // 原始图像尺寸
    det_shape = *in_det_shape;  // 检测输入尺寸
    use_kps = in_use_kps;       // 是否使用关键点
    box_len = in_box_len;

    // 计算宽高缩放比例
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);

    // 设置YuNet特有的anchor参数
    min_sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    steps = {8, 16, 32, 64};
    variance = {0.1f, 0.2f};

    // 生成先验框
    GeneratePriors();
    
    // 加载模型
    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    // 创建模型输入tensor (RGB三通道)
    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_RGB, SSNE_BUF_AI);
    
    printf("[INFO] YuNet initialized with input shape [%d, %d]\n", det_shape[0], det_shape[1]);
}

/**
 * @brief 执行人脸检测预测
 */
void YUNET::Predict(ssne_tensor_t* img, FaceDetectionResult* result, float conf_threshold) {
    
    // offline图像tensor初始化：对输入图像进行预处理（resize等）
    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        printf("[ERROR] Failed to run AI preprocess pipe!\n");
        printf("ret: %d\n", ret);
        return;
    }

    // 保存当前帧的图像（每次调用都更新，最终保留最后一帧）
    g_last_img = *img;
    g_last_pipe_input = inputs[0];
    g_has_frame = true;
    
    // 前向推理：在NPU上执行模型推理
    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "ssne inference fail!\n");
        return;
    }

    // 获取模型输出：4个输出tensor (head_conv_0, head_conv_1, head_conv_2, head_conv_3)
    ssne_getoutput(model_id, 4, outputs);
    
    // 获取各层输出数据和尺寸
    // 注意: 输出格式为 NHWC
    float* head0 = (float*)get_data(outputs[0]);
    float* head1 = (float*)get_data(outputs[1]);
    float* head2 = (float*)get_data(outputs[2]);
    float* head3 = (float*)get_data(outputs[3]);
    
    // 计算各层特征图尺寸
    int w = det_shape[0];
    int h = det_shape[1];
    
    int feature_map_2th_h = ((h + 1) / 2) / 2;
    int feature_map_2th_w = ((w + 1) / 2) / 2;
    int h0 = get_height(outputs[0]);
    int w0 = get_width(outputs[0]);
    int h1 = get_height(outputs[1]);
    int w1 = get_width(outputs[1]);
    int h2 = get_height(outputs[2]);
    int w2 = get_width(outputs[2]);
    int h3 = get_height(outputs[3]);
    int w3 = get_width(outputs[3]);


    
    int num_anchors_0 = h0 * w0 * 3;  // 3 anchors per position for head0
    int num_anchors_1 = h1 * w1 * 2;  // 2 anchors per position for head1
    int num_anchors_2 = h2 * w2 * 2;  // 2 anchors per position for head2
    int num_anchors_3 = h3 * w3 * 3;  // 3 anchors per position for head3
    

    
    // 从4个head输出恢复loc/conf/iou
    std::vector<float> loc, conf, iou;
    RestoreOutputsFromHeads(head0, head1, head2, head3,
                            h0, w0, h1, w1, h2, w2, h3, w3,
                            loc, conf, iou);
    
    // 解码检测结果
    std::vector<std::array<float, 4>> bboxes;
    std::vector<std::array<float, 10>> landmarks;
    std::vector<float> scores;
    Decode(loc, conf, iou, bboxes, landmarks, scores);
    
    // 后处理
    Postprocess(&bboxes, &landmarks, &scores, result, &conf_threshold);
}

/**
 * @brief 释放资源
 */
void YUNET::Release() {
    // 保存最后一帧的图像（如果有的话）
    if (g_has_frame) {
        printf("[INFO] Saving last frame images...\n");
        save_tensor(g_last_img, "dbg_in.raw");
        save_tensor(g_last_pipe_input, "dbg_in_pipe.raw");
        printf("[INFO] Last frame saved successfully!\n");
    }

    // 释放输入tensor
    release_tensor(inputs[0]);
    // 释放输出tensor
    release_tensor(outputs[0]);
    release_tensor(outputs[1]);
    release_tensor(outputs[2]);
    release_tensor(outputs[3]);
    // 释放预处理管道
    ReleaseAIPreprocessPipe(pipe_offline);
}

/* debug */
/**
 * @brief 保存图像数据到二进制文件（调试用）
 */
void YUNET::saveImageBin(const void* data, int w, int h, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&w, sizeof(int), 1, file);
        fwrite(&h, sizeof(int), 1, file);
        fwrite(data, sizeof(char), w * h * 3, file);  // RGB三通道
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    }
    else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}

/**
 * @brief 保存浮点数组到二进制文件（调试用）
 */
void YUNET::saveFloatBin(const float* data, int length, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&length, sizeof(int), 1, file);
        fwrite(data, sizeof(float), length, file);
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    }
    else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}
