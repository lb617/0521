/*
 * @Filename: utils.hpp
 * @Description: OSD 可视化器 + 工具函数声明
 */
#pragma once

#include "osd-device.hpp"
#include "common.hpp"
#include <algorithm>

/*!
 * @brief OSD 可视化器
 *
 * Layer 0 — 面部检测框 (每帧更新, 空心矩形)
 * Layer 1 — 特征分数区域 (固定正方形, 实心彩色块)
 * Layer 2 — 位图 logo (每 10 帧切换)
 */
class VISUALIZER {
public:
    void Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path = "");
    void Release();

    /*! @brief 绘制检测框 (OSD Layer 0) */
    void Draw(const std::vector<std::array<float, 4>>& boxes);

    /*! @brief 绘制 grimace 档位指示器 (OSD Layer 1), 每特征 4 格点亮对应档位 */
    void DrawGrimaceScores(const int levels[5]);

    /*! @brief 绘制位图 (OSD Layer 2) */
    void DrawBitmap(const std::string& bitmap_path, const std::string& lut_path = "",
                    int pos_x = 0, int pos_y = 0, int layer_id = 2);

    static const int DETECTION_LAYER_ID = 0;
    static const int SCORE_LAYER_ID     = 1;
    static const int BITMAP_LAYER_ID    = 2;

private:
    sst::device::osd::OsdDevice osd_device;
    int m_width;
    int m_height;
    std::string m_bitmap_lut_path_full;
};
