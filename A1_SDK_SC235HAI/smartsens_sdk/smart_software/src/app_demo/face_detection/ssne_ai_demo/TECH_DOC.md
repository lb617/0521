# SmartSens M1Pro 鼠脸表情检测系统技术文档

**版本：** v1.0 | **日期：** 2026-05-07 | **平台：** M1Pro / SC235HAI / SSNE NPU

---

## 1. 系统架构

```
帧质量检测 (frame_quality_lite)
       ↓ quality ≥ 0.667
鼠脸检测 (face_cropper_lite)
       ↓ conf ≥ 0.5
表情打分 (grimace_scorer_lite)
       ↓ TOTAL ≥ 7
触发 UART 告警 + 串口打印
```

---

## 2. 硬件与软件环境

| 项目 | 规格 |
|------|------|
| 开发板 | SmartSens M1Pro |
| 传感器 | SC235HAI (1920×1080) |
| CPU / NPU | ARM Cortex-A7 (NEON) / SSNE |
| 操作系统 | Buildroot 2022.02 |
| 通信接口 | GPIO2 复用为 UART (/dev/uartdev, 115200 8N1) |

---

## 3. 模型规格

| 模型 | 输入格式/尺寸 | 输出 | 阈值 |
|------|--------------|------|------|
| frame_quality_lite | SSNE_Y_8, 224×224 | 1 sigmoid (0~1) | 0.667 |
| face_cropper_lite | SSNE_BGR, 256×256 | 5 sigmoid [conf,cx,cy,w,h] | conf ≥ 0.5 |
| grimace_scorer_lite | SSNE_BGR, 224×224 | 20 logits (5器官×4类) | TOTAL ≥ 7 |

---

## 4. 代码修改

### 4.1 mgs.cpp（新建，替代 demo_face_yunet.cpp）

核心逻辑：
- **Step 1**：质量检测，quality < 0.667 → REJECT + 清除 OSD
- **Step 2**：cropper 输出 cxcywh → 转换为像素坐标（**宽度 8 像素对齐**），conf < 0.5 → 无鼠脸
- **Step 3**：grimace 20 个 logits → softmax → 各器官取 argmax → 5 项之和为 TOTAL，≥ 7 触发告警

```cpp
// 宽度 8 像素对齐（防止 SSNE 报错：width not divisible by 8）
int face_w = (int)((cw * 1920) / 8) * 8;
int face_h = (int)(ch * 1080);

// 表情打分
int eye_score   = argmax(eye_logit, 4);
int nose_score  = argmax(nose_logit, 4);
int cheek_score = argmax(cheek_logit, 4);
int ear_score   = argmax(ear_logit, 4);
int whisker_score = argmax(whisker_logit, 4);
int total = eye_score + nose_score + cheek_score + ear_score + whisker_score;

if (total >= 7) {
    printf("[ALERT] TOTAL=%d >= 7, stimulate evacuation!\n", total);
    uart_send_signal(uart_fd, '1');
}
```

### 4.2 pipeline_image.cpp

```cpp
// 修改前：OnlineSetCrop(kPipeline0, 240, 1680, 0, 1080);  // 输出 1440×1080
// 修改后：OnlineSetCrop(kPipeline0, 0, img_width, 0, img_height);  // 输出 1920×1080
```

### 4.3 CMakeLists.txt

```cmake
# 修改前：add_executable(demo_face_yunet demo_face_yunet.cpp)
# 修改后：add_executable(mgs mgs.cpp)
```

---

## 5. UART/GPIO 告警

GPIO2 复用为 UART，通过 `/dev/uartdev` 向下位机发送告警字符 `'1'`。

```cpp
int uart_init(const char* device) {
    uart_fd = open(device, O_WRONLY | O_NOCTTY | O_NONBLOCK);
    tcgetattr(uart_fd, &oldtty);
    cfsetospeed(&newtty, B115200);
    newtty.c_cflag = CS8 | CREAD | CLOCAL;
    tcsetattr(uart_fd, TCSANOW, &newtty);
    return 0;
}
```

---

## 6. 编译与部署

```bash
cd /home/pdd/data/A1_SDK_SC235HAI/smartsens_sdk
make clean
make smartsens_sc235hai_ai_demo_defconfig
make -j$(nproc)
# 二进制：output/target/usr/bin/mgs
```

模型文件部署到 `/app_assets/models/`：`frame_quality_lite.m1model`、`face_cropper_lite.m1model`、`grimace_scorer_lite.m1model`

---

## 7. 串口输出示例

| 场景 | 输出 |
|------|------|
| 质量差 | `[Frame N] Quality=0.3421 REJECT` |
| 无鼠脸 | `[Frame N] Quality=0.7823 PASS` → `[Frame N] No face detected (conf=0.34 < 0.50)` |
| 正常打分 | `[Frame N] Quality=... PASS` → `Cropper: conf=... bbox_cxcywh=[...]` → `Grimace scores: ... TOTAL=6` |
| 触发告警 | 以上 + `[ALERT] TOTAL=8 >= 7, stimulate evacuation!` + `[UART] Signal sent: 1` |

---

*文档结束*
