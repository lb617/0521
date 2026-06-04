# A1 部署调试日志

> **维护人**：（填写）
> **起始日期**：2026-05-21（局部晴朗回忆交接 → 此后按 [EXPERIMENT_RULES.md](EXPERIMENT_RULES.md) 持续维护）
> **板卡**：飞凌微 A1（SmartSens SC235HAI）

---

## 使用说明

- 一条问题一条记录。**已解决**和**待解决**两节，状态变了就把整条剪过去。
- **已解决**至少写：现象 / 试过什么 / 最后怎么搞定。
- **待解决**至少写：现象 / 试过哪些办法各自啥结果 / 怀疑是哪里。
- 改代码前先在对应条目里加假设（见 R1）；改完更新结果（见 R2、R3）。

---

## §1 已解决问题

### 1. OSD 绘图检测框：cxcywh vs xyxy 误判

**错误代码**（假设输出是 cxcywh）：
```c
float cx_norm = cropper_data[1];
float cy_norm = cropper_data[2];
float bw_norm = cropper_data[3];
float bh_norm = cropper_data[4];

float face_x1 = (cx_norm - bw_norm / 2.0f) * img_width;
float face_y1 = (cy_norm - bh_norm / 2.0f) * img_height;
float face_x2 = (cx_norm + bw_norm / 2.0f) * img_width;
float face_y2 = (cy_norm + bh_norm / 2.0f) * img_height;
```

模型实际输出是 xyxy，cxcywh→xyxy 这段转换多余，导致检测框坐标错误。

**修正**：
```c
float* cropper_data = (float*)get_data(output_cropper[0]);
float conf    = cropper_data[0];
float x1_norm = cropper_data[1];
float y1_norm = cropper_data[2];
float x2_norm = cropper_data[3];
float y2_norm = cropper_data[4];

float face_x1 = x1_norm * img_width;
float face_y1 = y1_norm * img_height;
float face_x2 = x2_norm * img_width;
float face_y2 = y2_norm * img_height;
```

**根因**：模型契约不清——cropper 输出格式没有写明，部署侧默认按 YOLOv8 原始约定 cxcywh 处理。
**长期修复**：在 `models/CONTRACTS/face_cropper_lite.md` 写清楚输出格式。

---

### 2. 编译链问题

基本套用源码编译路即可。有一些小问题，把编译报错反复扔给 AI 改即可。

> ⚠️ **风险标注**：这一项严格说不算"已解决"——是"AI 暴力试错通过"。
> 下次再出问题需要重新依赖 AI。建议在下次踩到时，把 AI 给的解释也写进 DEVLOG，逐步建立编译链的心智模型。

---

### 3. 检测框绘制太小不打分

模型前期输入与板上图像获取到的图形尺寸不匹配，添加裁剪对齐即可。

> **TODO**：补具体的尺寸对应（模型输入 224×224 vs 图像 1920×1080 → 裁剪/缩放比例多少？）。
> 下次复发时务必把数字记下来。

---

## §2 待解决问题

### 1. 串口打印显示 wrong input 和 inference failed

> ⚠️ 改了很多版本，但是会话记录被误删了几条。这里记录最近一次尝试。
> **核心问题**：尝试了多个修复，板上行为仍未改变 → 不确定是修复没起作用，还是这些 fix 处理的不是真因。

**问题 1：pipe_cropper 和 pipe_grimace 从未调用 SetCrop()**

- **根因**：内部裁剪区域默认 (0,0,0,0)，导致 S1AIPreprocess 管线内部断言失败，静默跳过处理，输出 tensor 为未初始化状态 → 送入 `ssne_inference` 触发 `[SSNE] Wrong input tensor!`
- **修复**：为三个 pipe 均设置初始裁剪区域
  ```c
  SetCrop(pipe_quality, 420, 0, 1500, 1080);   // 中心裁方
  SetCrop(pipe_cropper, 0, 0, 1920, 1080);      // 全图 ← 新增
  SetCrop(pipe_grimace, 0, 0, 1920, 1080);      // 全图（循环内动态覆盖）← 新增
  ```

**问题 2：RunAiPreprocessPipe 返回值从未检查**

- **现象**：管线失败后仍把脏 tensor 送入推理，无法定位真正出错环节
- **根因**：参考实现 `yunet.cpp` 检查了返回值并在失败时 return。`mgs.cpp` 直接忽略了返回值。
- **修复**：3 处调用全部改为检查返回值
  ```c
  // 修复前
  RunAiPreprocessPipe(pipe_xxx, img_sensor, input_xxx);
  // 修复后
  int ret = RunAiPreprocessPipe(pipe_xxx, img_sensor, input_xxx);
  if (ret != 0) {
      fprintf(stderr, "[ERROR] xxx preprocess failed! ret=%d\n", ret);
      continue;
  }
  ```

**问题 3：grimace 推理失败——裁剪宽度不能被 8 整除**

- **根因**：`face_cropper_lite` 模型输出的两个角点顺序不保证（x1 > x2 或 y1 > y2 可能）。代码直接按 x1,y1,x2,y2 使用时，x1>x2 导致裁剪宽度仅 ~8 像素，resize 到 224×224 时宽度无法被 8 整除，管线跳过。
- **修复**：在坐标转换前 swap
  ```c
  if (x1_norm > x2_norm) { float tmp = x1_norm; x1_norm = x2_norm; x2_norm = tmp; }
  if (y1_norm > y2_norm) { float tmp = y1_norm; y1_norm = y2_norm; y2_norm = tmp; }
  ```

**当前怀疑**：上述三处都改完后，串口行为仍未恢复正常 → 怀疑还有其他根因，可能与下面 §2.2 是同一个 root cause（模型转换坏了导致一切下游解析都失效）。

**下一步打算**：跑 `scripts/verify_model.py` 验证模型转换是否正常，然后再回头看这些修复是不是其实已经起作用、只是被另一个问题盖住了。

---

### 2. 检测鼠脸不准，检测框基本只会出现在中心，非鼠脸也画框

**怀疑**：模型转了以后有问题。

**下一步打算**：跑 `scripts/verify_model.py` 验证 —— 用同一组测试图比较 PC 端 ONNX 推理输出 vs 板上 a1model 推理输出。如果数值差异显著（详见 [EXPERIMENT_RULES.md](EXPERIMENT_RULES.md) R5 的阈值表），则锁定是转换问题，需要回头改算法侧的导出/校准。

**阻塞**：这是当前最大悬念。在没验证之前，所有上层修复都可能是徒劳。**优先级 P0**。

---

## §3 待规划项目

来自交接时"想写的"：

- [ ] **写 `scripts/verify_model.py`**：验证模型转换出来之后输出是否正确。这是当前最大悬念，先做这个。
- [ ] **考虑"从底重写一份"**：如果验证出模型转换没问题，但代码逻辑积累了太多 patch 难以维护，评估"基于 yunet.cpp 范本重写 mgs.cpp"的成本和收益。
- [ ] 项目整体的必要信息的缺失：1. 恺哥当时在群里面整理了全量的部署代码，需要整理到model文件夹
- [ ] 原版SDK代码，以及相关的示例模型等你在部署中用到的官方信息
---

## 相关参考

- [实验规则 EXPERIMENT_RULES.md](EXPERIMENT_RULES.md)
- [板上状态 BOARD_STATE.md](BOARD_STATE.md)
- 部署 pipeline 速查见 vault：`workbench/A1-board-deployment-reference.md`



李犇/第13周delog；（详见ai——chats）
1. [mgs.cpp:225] `sleep(0.2)` 系统初始化等待无效
usleep()` 以微秒为单位，`200000` 微秒 = 0.2 秒，真正实现系统初始化稳定等待。
2. [osd-device.cpp:58,81] `sleep(0.25)` DMA 缓冲区等待无效
3. [CMakeLists.txt:25] CMake 语法错误
原因**: `STATUS` 是标准的 CMake 消息模式，输出带 `--` 前缀的信息行。
4. [mgs.cpp:46,85] `strerror()` 命名空间问题
5. [mgs.cpp:179-183] 模型加载无错误检查
*问题**: `ssne_loadmodel()` 返回的 model_id 未检查有效性。如果模型文件缺失或损坏，返回值可能是 `0xFFFF`（无效句柄），但原代码仅打印 ID 后继续执行，后续 `ssne_inference` 才会崩溃，排查困难。
6. [utils.cpp:271] 测试用 `Draw()` 会清除所有 OSD 图层
7. [TECH_DOC.md:67] 文档与代码不一致
 8. [tools/inspect_m1model.py] 新增 m1model 离线结构分析工具
 .m1model` 是 SSNE NPU 私有格式，无法在 PC 上运行推理
 9. [tools/test_onnx_models.py] 新增 ONNX 模型 PC 端离线推理验证工具
 在 PC 端用 ONNX Runtime 离线验证三个鼠脸检测 ONNX 模型的推理效果，不依赖 M1 硬件。
 **模型结构分析结果**:

| 模型 | 输入 | 输出 | 节点数 | 参数量 | 主要算子 |
|------|------|------|--------|--------|----------|
| frame_quality_lite | [1,1,224,224] float32 | [1,1,1,1] sigmoid | 22 | 210,321 | Conv(9)+Relu(8)+MaxPool(3)+Sigmoid |
| face_cropper_lite | [1,3,256,256] float32 | [1,5,1,1] sigmoid | 22 | 210,997 | Conv(9)+Relu(8)+MaxPool(3)+Sigmoid |
| grimace_scorer_lite | [1,3,224,224] float32 | [1,20,1,1] logits | 21 | 422,572 | Conv(9)+Relu(8)+MaxPool(3) |

**验证结果 (2026-05-29)**: **18/18 全部通过**



下一步应该会重写构建一个新的项目文件，目前正在逐步解决编译链问题


李犇/第14周debug
1.build_app.sh 它只重编了旧包，不会编译新的 mouse_face_demo
写入：rm -rf output/build/mouse_face_demo/  make ssne_ai_demo-rebuild mouse_face_demo-rebuild
2.现在只有 一个镜像文件，两个 app 都会被打包进同一个 rootfs，也就是说板子上会同时跑两个程序
defconfig — 把旧人脸检测编译开关注释掉，只保留小鼠检测：BR2_PACKAGE_MOUSE_FACE_DEMO=y
build_app.sh — 只重编小鼠包：rm -rf output/build/mouse_face_demo/   make mouse_face_demo-rebuild
3.编译过程会拉取下载一些源码，由于docker容器网络环境隔离，经常卡住，尝试开代理，直连或者tun模式都不行，docker容器内部配置代理也不行
解决办法：不断ctrl+C，不断重启电脑，然后挂着梯子编译
第一遍编译完成后，之后的编译都会很流畅，不用挂梯子
4.编译报错：error: binding reference of type 'std::vector<std::array<float, 4>>&' 
       to 'const std::vector<std::array<float, 4>>' discards qualifiers
       把 const 引用传给非 const 引用参数
utils.cpp 第 33 行 加了一行本地拷贝：void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes)
{
    // 拷贝一份，OsdDevice::Draw 需要非 const 引用
    std::vector<std::array<float, 4>> boxes_copy(boxes);
    
    osd_device.Draw(boxes_copy, ...);   // 传拷贝，不是 const 原值
5.IMAGEPROCESSOR 的三个函数在编译时出现了两份定义:
  编译流程是：


    1. g++ pipeline_image.cpp      → pipeline_image.o    (含 Initialize, GetImage, Release)
    2. g++ mouse_face_pipeline.cpp → mouse_face_pipeline.o (含 Initialize, GetImage, Release)  ← 重复！
    3. ld  把两个 .o 链接在一起    → 💥 multiple definition，不知道该用哪份
    更改：把 mouse_face_pipeline.cpp 里那几十行重复的 IMAGEPROCESSOR 实现代码删掉，替换成一行注释
6.更改文件后，编译报同样错误，
    清除旧的编译缓存：rm -rf output/build/mouse_face_demo/ && make mouse_face_demo-rebuild
7.对于打分模型输出的20个数认知改正，实现：
    特征1:  [■] [□] [□] [□]   ← level=0, 第0格实心
    特征2:  [□] [□] [■] [□]   ← level=2, 第2格实心
    特征3:  [□] [■] [□] [□]   ← level=1, 第1格实心
8.run.sh没有可执行权限，上板以后没有自动跑起来
      在 mouse_face_demo.mk 的安装步骤里加了一行：


    chmod +x $(TARGET_DIR)/app_demo/scripts/run.sh
    这样 Buildroot 打包镜像前，run.sh 就会被赋予执行权限。
9.串口打印卡住，原因移动了模型文件路径，Segfault — 加载失败后没有检查ssne_loadmodel() 失败时返回 0，代码没有检查就直接用了。需要加防御代码。
    手动添加模型文件；
        防御性检查（解决 segfault）
        ssne_loadmodel() 打开文件失败后，model_id 是无效值，后续 SetNormalize 传入无效 id 直接崩溃。现在加了三处文件存在性检查：


        if (!file_exists(quality_model.c_str())) {
            fprintf(stderr, "[ERROR] Model file not found: %s\n", quality_model.c_str());
            return;  // 提前退出，不会崩溃
        }
10.帧质量或者未检测到小鼠面部时串口打印保持静默，不利于判断和调试：增加每 30 帧会自动打印一行调试信息

11.face_cropper 输出的坐标是 0~1 归一化值，不是像素坐标
直接当 0~1 归一化值处理 × img_width / × img_height，同时加 std::swap 处理可能反转的 x1/x2、y1/y2


之前: fx1 = 1.0 × 7.5 = 7px      → 无效
之后: fx1 = 1.0 × 1920 = 1920px  → 合理
      fx2 = 0.8 × 1920 = 1536px
      fx1 > fx2 → swap → fx1=1536, fx2=1920 ✓

12.grimace_scorer 因为无效裁剪区域崩溃
链式反应：
face_cropper 输出噪声坐标 → 缩放后框无效 → 回退到全图 
→ 全图 1920×1080 YUV422_16 → 预处理到 256×224 SSNE_Y_8 
→ 模型期望的输入格式/数据类型不匹配 → [SSNE] Wrong input tensor!
crop 无效时设置 grimace_levels[i] = -1，跳过 grimace_scorer 推理，但检测框仍然有效并绘制。主程序检测到 levels[0] < 0 时只画框不画分。

13.帧质量总是卡在0.498，小鼠面部置信度卡在0.5-0.55之间
    先降低两个阈值：
            [ERROR] grimace_scorer preprocess failed! ret=504
            1970-01-01 00:12:49.297574 ERROR  63:3069214080: [S1AIPreprocess]output image dtype=1 but width=105(not divisible by 8); skip run offline pipe!
            报错刷屏
            pipeline_image.cpp 里继承了人脸检测 demo 的裁剪设置，传感器输出被裁剪到了1440×1080（左右各裁了 240px）。但代码里坐标是用 1920 来缩放的，结果 SetCrop 传入的坐标超出了 1440 的图像宽度，硬件算出了垃圾值 65288。
            [ERROR] grimace_scorer preprocess failed! ret=504
            1970-01-01 00:12:50.983025 ERROR  63:3069169024: [S1AIPreprocess]output image dtype=1 but width=53(not divisible by 8); skip run offline pipe!
            再次刷屏了
            在 mouse_face_pipeline.cpp 的 SetCrop 调用前，对坐标做 8 对齐：
            // x1,y1 向下对齐到 8, x2,y2 向上对齐到 8
            crop_x1 = (x1 / 8) * 8;       // 例: 1782 → 1782/8=222, 222*8=1776
            crop_y1 = (y1 / 8) * 8;       // 例: 541  → 541/8=67,   67*8=536
            crop_x2 = ((x2 + 7) / 8) * 8; // 例: 1918 → 1925/8=240, 240*8=1920
            crop_y2 = ((y2 + 7) / 8) * 8; // 例: 783  → 790/8=98,   98*8=784
        
        传感器数据正常，Pipeline 存在双缓冲交替问题
        1. pipeline_image.cpp 还留着旧的裁剪参数
        复制过来后没改，Pipeline 实际输出的是 1440×1080（左右各裁 240px），但代码里存的 img_width=1920。坐标 ×1920 比实际需要的 ×1440 放大了 1.33 倍，导致 bbox 偏位。

        修复：改为全分辨率输出 1920×1080，同时加了 OnlineSetFrameDrop(kPipeline0, 3, 0) 丢弃前 3 帧让双缓冲稳定。

        2. 检测置信度失败时静默跳过
        之前 det_conf < 0.6 时没有任何打印，所以根本不知道是质量不过还是检测不过。

        修复：加了调试打印，每 5 次质量通过的帧里打印一次低置信度信息。

        3. 数据本身正常
        传感器确实在工作——同一个模型看到了同一片区域（图像右下部），坐标在帧间有合理漂移。问题只是：

        一半帧是静态缓冲（quality=0.498），加了 FrameDrop 应该能解决
        det_conf=0.50 说明镜头里可能确实没有小鼠，或者小鼠不在那个位置
        
        
        还是quality卡0.4980，置信度卡0.5013
        问题不在传感器，也不在模型加载。问题在 Preprocess Pipe 的双缓冲机制。
        每一个 RunAiPreprocessPipe 调用内部可能使用了双缓冲，两个 buffer 交替返回——一个 buffer 包含真实帧，另一个包含未更新的旧数据。三个模型各有一条独立的 pipe，它们各自的 buffer 交替不同步，导致 quality 取到了"旧 buffer 的默认数据"（0.498）或"真实图像"（0.996）。

        OnlineSetFrameDrop 只影响 Online Pipeline 的帧速率，不影响 Preprocess Pipe 的内部缓冲。

        修复方案：改用 copy_tensor + RunAiPreprocessPipe 直接拿到干净数据
        问题根源是每次 RunAiPreprocessPipe 的输出 tensor（input_quality/input_detect/input_score）在推理后被"污染"了。正确做法是每次 Preprocess 完后立即推理，不做任何依赖前一帧的操作，同时确保 input tensor 每次都是干净状态。

        实际上当前代码逻辑上已经是一帧一推理的顺序，问题更可能是 create_tensor 创建的是静态 tensor，而 Preprocess Pipe 的行为依赖于 tensor 的内部状态。

        最简单的尝试：把 input 格式从 Y8 改成 BYTES。因为 Y8 格式可能触发预处理管道的特定行为（只读亮度通道），而 BYTES 格式让模型内部的预处理自行处理格式转换。
        rameDrop 不是全部问题。在加 FrameDrop 之前，quality 已经是二值的了（0.4980 vs 0.9961）。FrameDrop 只是让坏帧比例从 50% 变成了 70%。

        真正的问题很可能在预处理管的格式转换。现有 demo 使用的是 YUV→RGB 转换，一直正常工作。而我们的 quality 和 scorer 模型用的是 YUV→Y8（灰阶）。硬件预处理管可能对 SSNE_Y_8 输出格式的支持有 bug，导致输出 tensor 的数据状态不稳定。
        复尝试：把灰阶改为 RGB
        模型实际接受 3 通道 RGB 和 1 通道灰阶是等效的（网络第一层会自适应通道数权重）——但如果模型文件本身确实是 1 通道，改 RGB 可能不兼容。更安全的办法是用 SSNE_BYTES 替代 SSNE_Y_8。

        