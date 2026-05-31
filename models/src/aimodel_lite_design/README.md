# aimodel_lite_design

AIMODEL / 板端部署参考代码。

重点文件：

- `preprocess_reference.py`：CPU 侧预处理/后处理参考
- `export_aimodel_onnx.py`：部署参考版 ONNX 导出
- `check_onnx_ops.py`：ONNX 算子检查
- `aimodel_lite_models.py`：部署参考版模型定义

更多说明见：`docs/05_DEPLOYMENT_NOTES.md`。
