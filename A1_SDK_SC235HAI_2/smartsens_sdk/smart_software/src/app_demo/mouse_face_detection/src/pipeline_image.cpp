/*
 * @Filename: pipeline_image.cpp
 * @Description: 图像采集器 — 从传感器 Online Pipeline 获取全分辨率帧
 *
 * 与现有人脸检测 demo 不同：此处使用全分辨率 1920×1080，不裁剪。
 * 模型预处理阶段各自做 resize，不需要 Pipeline 层面裁剪。
 */

#include "../include/common.hpp"
#include <iostream>
#include <unistd.h>

void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape)
{
    img_shape = *in_img_shape;

    uint16_t img_width  = static_cast<uint16_t>(img_shape[0]);
    uint16_t img_height = static_cast<uint16_t>(img_shape[1]);
    format_online = SSNE_YUV422_16;

    // 全分辨率输出，不做裁剪
    OnlineSetCrop(kPipeline0, 0, img_width, 0, img_height);
    OnlineSetOutputImage(kPipeline0, format_online, img_width, img_height);

    int res0 = OpenOnlinePipeline(kPipeline0);
    if (res0 != 0) {
        printf("[ERROR] Failed to open online pipeline!\n");
        printf("ret: %d\n", res0);
        return;
    }
    printf("[INFO] IMAGEPROCESSOR: online pipe0 opened, %dx%d (full frame)\n", img_width, img_height);
}

void IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor)
{
    int ret = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR] Get Invalid Image from kPipeline0!\n");
    }
}

void IMAGEPROCESSOR::Release()
{
    CloseOnlinePipeline(kPipeline0);
    printf("[INFO] IMAGEPROCESSOR: OnlinePipe closed!\n");
}

