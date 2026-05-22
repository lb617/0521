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
