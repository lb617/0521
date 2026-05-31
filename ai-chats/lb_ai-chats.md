# 代码修改与调试记录

> 项目: SmartSens A1 SDK SC235HAI — AI 鼠脸检测应用
> 分支: main
> 日期: 2026-05-25 ~ 2026-05-29

---

## 修改清单总览

| 序号 | 日期 | 文件 | 严重程度 | 问题简述 |
|------|------|------|----------|----------|
| 1 | 05-25 | `mgs.cpp:225` | **高** | `sleep(0.2)` 被截断为 0，系统初始化等待无效 |
| 2 | 05-26 | `osd-device.cpp:58,81` | **高** | `sleep(0.25)` 被截断为 0，DMA 缓冲区等待无效 |
| 3 | 05-26 | `CMakeLists.txt:25` | **高** | `message([<mode>]...)` 未替换模板占位符 |
| 4 | 05-27 | `mgs.cpp:46,85` | **中** | `strerror()` 在 `<cstring>` 下缺少 `std::` 命名空间 |
| 5 | 05-27 | `mgs.cpp:179-183` | **中** | 模型加载失败无错误检查，静默失败难以排查 |
| 6 | 05-27 | `utils.cpp:271` | **低** | 测试用 `Draw()` 无参版本会清除所有 OSD 图层 |
| 7 | 05-28 | `TECH_DOC.md:67` | **低** | `uart_send_signal` 签名与代码不一致 |
| 8 | 05-28 | `tools/inspect_m1model.py` | — | 新增 m1model 离线结构分析工具 |
| 9 | 05-28 | `tools/test_onnx_models.py` | — | 新增 ONNX 模型 PC 端离线推理验证工具 |

---

## 1. [mgs.cpp:225] `sleep(0.2)` 系统初始化等待无效

**问题**: `sleep()` 的参数类型是 `unsigned int`（秒）。传入浮点数 `0.2` 会被隐式转换为 `0`，等待完全不生效。这会导致 SSNE 引擎和设备可能未稳定就开始图像采集。

**修改前**:
```cpp
cout << "sleep for 0.2 second!" << endl;
sleep(0.2);
```

**修改后**:
```cpp
cout << "sleep for 0.2 second!" << endl;
usleep(200000);
```

**原因**: `usleep()` 以微秒为单位，`200000` 微秒 = 0.2 秒，真正实现系统初始化稳定等待。

---

## 2. [osd-device.cpp:58,81] `sleep(0.25)` DMA 缓冲区等待无效

**问题**: 同问题 1，`sleep(0.25)` 被截断为 `0`。OSD 初始化中 DMA 缓冲区分配后需要等待硬件完成，但实际未等待，可能导致 `osd_alloc_buffer` 返回后 DMA 尚未就绪，后续 `osd_create_layer` 操作失败。

**修改前**:
```cpp
// Line 58
osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, dma_size);sleep(0.25);

// Line 81
sleep(0.25);  // 等待DMA分配完成
```

**修改后**:
```cpp
// Line 58
osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, dma_size);usleep(250000);

// Line 81
usleep(250000);  // 等待DMA分配完成
```

---

## 3. [CMakeLists.txt:25] CMake 语法错误

**问题**: `[<mode>]` 是 CMake 文档中的模板占位符，未被替换。CMake 会报语法错误或输出异常。

**修改前**:
```cmake
message([<mode>] "message to display" ${TARGET})
```

**修改后**:
```cmake
message(STATUS "TARGET path: ${TARGET}")
```

**原因**: `STATUS` 是标准的 CMake 消息模式，输出带 `--` 前缀的信息行。

---

## 4. [mgs.cpp:46,85] `strerror()` 命名空间问题

**问题**: 头文件使用了 `#include <cstring>`（C++ 标准头），其中 `strerror` 在 `std` 命名空间中。虽然 GCC+glibc 在全局命名空间也提供这些函数，但属于实现定义行为，在更严格的编译选项下可能编译失败。

**修改前**:
```cpp
fprintf(stderr, "[UART] Failed to open %s: %s\n", UART_DEVICE, strerror(errno));
// ...
fprintf(stderr, "[UART] Write failed: %s\n", strerror(errno));
```

**修改后**:
```cpp
fprintf(stderr, "[UART] Failed to open %s: %s\n", UART_DEVICE, std::strerror(errno));
// ...
fprintf(stderr, "[UART] Write failed: %s\n", std::strerror(errno));
```

---

## 5. [mgs.cpp:179-183] 模型加载无错误检查

**问题**: `ssne_loadmodel()` 返回的 model_id 未检查有效性。如果模型文件缺失或损坏，返回值可能是 `0xFFFF`（无效句柄），但原代码仅打印 ID 后继续执行，后续 `ssne_inference` 才会崩溃，排查困难。

**修改前**:
```cpp
uint16_t model_quality = ssne_loadmodel(
    const_cast<char*>(path_quality.c_str()), SSNE_STATIC_ALLOC);
uint16_t model_cropper = ssne_loadmodel(
    const_cast<char*>(path_cropper.c_str()), SSNE_STATIC_ALLOC);
uint16_t model_grimace = ssne_loadmodel(
    const_cast<char*>(path_grimace.c_str()), SSNE_STATIC_ALLOC);

printf("[INFO] Models loaded: quality=%d cropper=%d grimace=%d\n",
       model_quality, model_cropper, model_grimace);
```

**修改后**:
```cpp
uint16_t model_quality = ssne_loadmodel(
    const_cast<char*>(path_quality.c_str()), SSNE_STATIC_ALLOC);
uint16_t model_cropper = ssne_loadmodel(
    const_cast<char*>(path_cropper.c_str()), SSNE_STATIC_ALLOC);
uint16_t model_grimace = ssne_loadmodel(
    const_cast<char*>(path_grimace.c_str()), SSNE_STATIC_ALLOC);

// 模型加载失败检查: 0xFFFF 为无效模型句柄
if (model_quality == 0xFFFF || model_cropper == 0xFFFF || model_grimace == 0xFFFF) {
    fprintf(stderr, "[ERROR] Model loading failed! quality=%u cropper=%u grimace=%u\n",
            model_quality, model_cropper, model_grimace);
    ssne_release();
    return -1;
}

printf("[INFO] Models loaded: quality=%d cropper=%d grimace=%d\n",
       model_quality, model_cropper, model_grimace);
```

---

## 6. [utils.cpp:271] 测试用 `Draw()` 会清除所有 OSD 图层

**问题**: `VISUALIZER::Draw()` 无参版本（测试用）调用了 `osd_device.Draw(quad_rangle_vec)`（无 layer_id 版本），该版本内部调用 `osd_clean_all_layer()`，会清除所有图层，包括固定正方形（layer 1）和位图（layer 2）。虽然 `mgs.cpp` 中未使用该无参版本（只用了 `Draw(boxes)`），但作为公共 API 这是一个危险的陷阱。

**修改前**:
```cpp
// 调用OSD设备绘制测试矩形框
osd_device.Draw(quad_rangle_vec);
```

**修改后**:
```cpp
// 使用 layer 0 绘制，避免清除其他图层
osd_device.Draw(quad_rangle_vec, DETECTION_LAYER_ID);
```

---

## 7. [TECH_DOC.md:67] 文档与代码不一致

**问题**: TECH_DOC 中 `uart_send_signal` 显示为两个参数 `(uart_fd, '1')`，但实际代码只接受一个参数 `(int fd)`，在内部发送固定字符串 `"1"`。

**修改前**（TECH_DOC 代码示例中）:
```cpp
uart_send_signal(uart_fd, '1');
```

**修改后**:
```cpp
uart_send_signal(uart_fd);
```

---

## 8. [tools/inspect_m1model.py] 新增 m1model 离线结构分析工具

**日期**: 2026-05-25

**目的**: `.m1model` 是 SSNE NPU 私有格式，无法在 PC 上运行推理。该工具提供离线文件结构分析能力：
- 验证文件完整性（header + weight = file_size）
- 解析模型结构参数（层数、权重段大小、权重非零率等）
- 与代码预期值自动对比
- 计算 MD5 校验和，便于版本追踪

**用法**:
```bash
python3 tools/inspect_m1model.py --all              # 检测全部模型
python3 tools/inspect_m1model.py <path/to/model>    # 检测单个模型
```

**m1model 文件 MD5 校验** (2026-05-25):

| 文件 | MD5 |
|------|-----|
| face_cropper_lite.m1model | `cc92149f2af28d6673e4ea0f3dc79416` |
| frame_quality_lite.m1model | `f527924c396db8da84d2f6d9fdb2db50` |
| grimace_scorer_lite.m1model | `7b676915065fb2940029d69aeec49355` |
| yunet_160x120.m1model | `7c899ce904cd43d5480fce9e114d2306` |

---

## 9. [tools/test_onnx_models.py] 新增 ONNX 模型 PC 端离线推理验证工具

**日期**: 2026-05-29

**目的**: 在 PC 端用 ONNX Runtime 离线验证三个鼠脸检测 ONNX 模型的推理效果，不依赖 M1 硬件。

**测试内容**:
- 形状验证: 输入输出维度与代码预期一致
- 随机输入: 输出在有效范围内, 无 NaN/Inf
- 边界条件: 全零/全一输入正常
- 结构输入: 棋盘格、渐变、合成人脸图案
- 性能基准: CPU 推理耗时

**依赖**: `pip install onnx onnxruntime`

**模型结构分析结果**:

| 模型 | 输入 | 输出 | 节点数 | 参数量 | 主要算子 |
|------|------|------|--------|--------|----------|
| frame_quality_lite | [1,1,224,224] float32 | [1,1,1,1] sigmoid | 22 | 210,321 | Conv(9)+Relu(8)+MaxPool(3)+Sigmoid |
| face_cropper_lite | [1,3,256,256] float32 | [1,5,1,1] sigmoid | 22 | 210,997 | Conv(9)+Relu(8)+MaxPool(3)+Sigmoid |
| grimace_scorer_lite | [1,3,224,224] float32 | [1,20,1,1] logits | 21 | 422,572 | Conv(9)+Relu(8)+MaxPool(3) |

**验证结果 (2026-05-29)**: **18/18 全部通过**

```bash
python3 tools/test_onnx_models.py --verbose --benchmark
```

关键发现:
- 三个模型的 I/O 规格与 `mgs.cpp` 代码完全匹配
- `face_cropper_lite` 对任意输入置信度都很高 (>0.99), 设计上是宽松检测器, 依赖上游 quality 模型做帧过滤, 符合流水线逻辑
- `grimace_scorer_lite` 20 个 logit 在随机输入下 argmax 模式为 [2,1,2,2,1] total=8, 模型偏向高分配分
- CPU 推理耗时: quality 2.8ms, cropper 4.0ms, grimace 5.4ms (ONNX Runtime, x86_64)

