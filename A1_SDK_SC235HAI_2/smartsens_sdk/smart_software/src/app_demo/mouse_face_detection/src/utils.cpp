/*
 * @Filename: utils.cpp
 * @Description: OSD 可视化器实现 — 检测框绘制 + grimace 分数色条绘制
 */
#include "../include/utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>

// ============================================================
//  VISUALIZER
// ============================================================

void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path)
{
    m_width  = in_img_shape[0];
    m_height = in_img_shape[1];

    const char* lut_path = nullptr;
    if (!bitmap_lut_path.empty()) {
        m_bitmap_lut_path_full = "/app_demo/app_assets/" + bitmap_lut_path;
        lut_path = m_bitmap_lut_path_full.c_str();
    }
    osd_device.Initialize(m_width, m_height, lut_path);
    printf("[VISUALIZER] Initialized %dx%d\n", m_width, m_height);
}


void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes)
{
    // OsdDevice::Draw 需要非 const 引用，这里做一次拷贝
    std::vector<std::array<float, 4>> boxes_copy(boxes);

    if (boxes_copy.empty()) {
        // 清除 Layer 0 检测框
        osd_device.Draw(boxes_copy, 0, DETECTION_LAYER_ID,
                        fdevice::TYPE_HOLLOW, fdevice::TYPE_ALPHA75, 2);
        return;
    }

    printf("[VISUALIZER] Drawing %zu detection boxes\n", boxes_copy.size());
    osd_device.Draw(boxes_copy, 3, DETECTION_LAYER_ID,
                    fdevice::TYPE_HOLLOW, fdevice::TYPE_ALPHA75, 2);
}


void VISUALIZER::DrawGrimaceScores(const int levels[5])
{
    // 每个特征一行，每行 4 个小格代表档位 0~3
    // 命中档位的格子用实心亮色填充，其余用空心暗色表示

    const int cell_size   = 24;   // 每格像素
    const int cell_margin = 4;    // 格间距
    const int row_spacing = 36;   // 行间距
    const int base_x      = 20;   // 左上角 X
    const int base_y      = 60;   // 左上角 Y

    for (int trait = 0; trait < 5; trait++) {
        int level = levels[trait];
        if (level < 0) level = 0;
        if (level > 3) level = 3;

        for (int lv = 0; lv < 4; lv++) {
            int x1 = base_x + lv * (cell_size + cell_margin);
            int y1 = base_y + trait * row_spacing;
            int x2 = x1 + cell_size;
            int y2 = y1 + cell_size;

            std::vector<std::array<float, 4>> cell;
            cell.push_back({
                static_cast<float>(x1), static_cast<float>(y1),
                static_cast<float>(x2), static_cast<float>(y2)
            });

            if (lv == level) {
                // 命中档位: 实心亮色 (颜色索引 2~6 对应 5 个特征)
                osd_device.Draw(cell, 0, SCORE_LAYER_ID,
                                fdevice::TYPE_SOLID, fdevice::TYPE_ALPHA100, trait + 2);
            } else {
                // 非命中档位: 空心暗色 (颜色索引 1)
                osd_device.Draw(cell, 1, SCORE_LAYER_ID,
                                fdevice::TYPE_HOLLOW, fdevice::TYPE_ALPHA50, 1);
            }
        }
    }
}


void VISUALIZER::DrawBitmap(const std::string& bitmap_path, const std::string& lut_path,
                            int pos_x, int pos_y, int layer_id)
{
    std::string full_bitmap_path = "/app_demo/app_assets/" + bitmap_path;
    const char* full_lut_path = nullptr;
    osd_device.DrawTexture(full_bitmap_path.c_str(), full_lut_path, layer_id, pos_x, pos_y);
}


void VISUALIZER::Release()
{
    osd_device.Release();
    printf("[VISUALIZER] Released\n");
}
