# 小工具脚本

## verify_model.py（最优先）

**作用**：跑一组测试图，对比 PC 上 ONNX 推理 vs 板上 a1model 推理的输出，回答"模型转换是不是坏了"。

输入：ONNX 路径、板上 a1model dump 出来的 .npz、`tests/images/` 测试图
输出：每个输出张量的 max_rel_diff、cos_sim，一行结论

阈值参考：
- `cos > 0.99` 且 `max_rel_diff < 5%` → 量化噪声，正常
- `cos < 0.9` 或 `max_rel_diff > 30%` → 模型转换坏了，不要再调上层代码

骨架可以问 AI 写，但**注意板上是 NHWC、ONNX 是 NCHW，对比时要 reshape 对齐**。

## flash.ps1（后面再写）

一键 commit + 编译 + 烧录 + 状态记录。Aurora 命令行接口待查，如果不支持，烧录手动。
