# 模型文件

- `onnx/` — PyTorch 导出的 ONNX
- `a1model/` — 思思 AI 助手转换后的 .a1model

命名建议：`<模型名>_v<版本>.onnx` / `.a1model`，例如 `face_cropper_lite_v1.onnx`。

文件大的话考虑 git-lfs，先暂时全部入库观察大小。
