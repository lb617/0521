#!/usr/bin/env python3
"""
m1model 离线检测与结构解析工具

用途: 在不依赖 M1 硬件的情况下, 离线分析 .m1model 文件的内部结构,
     验证模型文件完整性, 提取网络元数据, 检测与代码预期值的一致性。

用法:
    python3 tools/inspect_m1model.py <model_file>
    python3 tools/inspect_m1model.py --all   # 检测所有模型

m1model 文件布局:
    [0x00-0x7F]  固定头部 (magic=0x01, 填充区)
    [0x80-0x12C] 层描述符区 (每层结构描述)
    [0x12C-end]  权重数据区
"""

import struct
import sys
import os
import hashlib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / \
    "smartsens_sdk/smart_software/src/app_demo/face_detection/ssne_ai_demo/app_assets/models"

# 代码预期值 (来自 mgs.cpp)
EXPECTED_SPECS = {
    "frame_quality_lite": {"input": "224x224 GRAY",  "output": "1 float (quality score)"},
    "face_cropper_lite":  {"input": "256x256 BGR",   "output": "5 float [conf,cx,cy,w,h]"},
    "grimace_scorer_lite": {"input": "224x224 BGR",  "output": "20 float (5organs x 4classes)"},
    "yunet_160x120":      {"input": "160x120 RGB",   "output": "4 heads ([H,W,C])"},
}


def parse_m1model(data: bytes, filepath: str = "") -> dict:
    """解析 m1model 文件结构"""
    size = len(data)

    if size < 0x130:
        return {"error": "file too small"}

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 1:
        return {"error": f"bad magic: 0x{magic:08x}, expected 0x00000001"}

    hdr = struct.unpack_from("<16I", data, 0x80)

    info = {
        "file_size": size,
        "magic": magic,
        "resolution_w": hdr[0],
        "resolution_h": hdr[1],
        "tensor_w": hdr[2],
        "tensor_h": hdr[3],
        "param_a": hdr[4],
        "param_b": hdr[5],
        "param_c": hdr[6],
        "param_d": hdr[7],
        "header_size": hdr[8],
        "layer_count": hdr[9],
        "total_header_end": hdr[10],
        "weight_size": hdr[11],
        "op_count": hdr[12],
        "reserved": hdr[13],
        "channel_info": hdr[14],
        "depth_info": hdr[15],
    }

    # 校验: 权重段 + 头部 = 文件大小
    estimated = info["weight_size"] + info["total_header_end"]
    info["size_match"] = (estimated == size)
    info["size_diff"] = estimated - size

    # MD5
    info["md5"] = hashlib.md5(data).hexdigest()

    # 权重数据基本信息
    weight_start = info["total_header_end"]
    weight_data = data[weight_start:]
    nonzero = sum(1 for b in weight_data if b != 0)
    info["weight_nonzero_pct"] = 100.0 * nonzero / max(len(weight_data), 1)
    info["weight_zero_pct"] = 100.0 - info["weight_nonzero_pct"]

    return info


def check_expected(name: str, info: dict) -> list:
    """与代码预期值对比, 返回警告列表"""
    warnings = []
    expected = EXPECTED_SPECS.get(name)

    if expected is None:
        return warnings

    # 对每个模型做针对性检查
    if name == "frame_quality_lite":
        if info["layer_count"] != 50:
            warnings.append(f"layer_count={info['layer_count']}, expected ~50")
        if info["weight_size"] < 200_000:
            warnings.append("weight_size suspiciously small for quality model")

    elif name == "face_cropper_lite":
        if info["layer_count"] != 50:
            warnings.append(f"layer_count={info['layer_count']}, expected ~50")

    elif name == "grimace_scorer_lite":
        if info["layer_count"] != 50:
            warnings.append(f"layer_count={info['layer_count']}, expected ~50")
        if info["channel_info"] < 100:
            warnings.append("channel_info looks wrong for 20-output model")

    elif name == "yunet_160x120":
        if info["layer_count"] != 74:
            warnings.append(f"layer_count={info['layer_count']}, expected ~74")
        if info["tensor_w"] * info["tensor_h"] < 10000:
            warnings.append("tensor dims too small for 160x120 input")

    return warnings


def print_report(filepath: str, info: dict, warnings: list):
    """打印检测报告"""
    name = os.path.basename(filepath)
    if "error" in info:
        print(f"\033[31m[ERROR]\033[0m {name}: {info['error']}")
        return

    status = "\033[32mOK\033[0m" if info["size_match"] else "\033[31mCORRUPT\033[0m"
    print(f"\n{'='*60}")
    print(f"  模型文件: {name}")
    print(f"  完整性:   {status}  (size={info['file_size']:,})")
    print(f"  MD5:      {info['md5']}")
    print(f"{'='*60}")
    print(f"  结构参数:")
    print(f"    分辨率:      {info['resolution_w']}x{info['resolution_h']}")
    print(f"    Tensor:      {info['tensor_w']}x{info['tensor_h']}")
    print(f"    Params:      {info['param_a']}, {info['param_b']}, {info['param_c']}, {info['param_d']}")
    print(f"    层数:        {info['layer_count']}")
    print(f"    操作数:      {info['op_count']} (0x{info['op_count']:x})")
    print(f"    权重段大小:  {info['weight_size']:,} bytes")
    print(f"    描述符段:    {info['header_size']} bytes")
    print(f"    总头部:      {info['total_header_end']} bytes")
    print(f"    权重非零率:  {info['weight_nonzero_pct']:.1f}%")
    print(f"    channel_info: {info['channel_info']} (0x{info['channel_info']:x})")

    # 与代码预期对比
    model_key = next((k for k in EXPECTED_SPECS if k in name), None)
    if model_key:
        spec = EXPECTED_SPECS[model_key]
        print(f"\n  代码预期:")
        print(f"    输入: {spec['input']}")
        print(f"    输出: {spec['output']}")

    if warnings:
        print(f"\n  \033[33m[警告]\033[0m")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\n  \033[32m[通过] 未发现异常\033[0m")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\n可用的模型文件:")
        if MODELS_DIR.exists():
            for f in sorted(MODELS_DIR.glob("*.m1model")):
                print(f"  {f}")
        sys.exit(1)

    targets = []
    if sys.argv[1] == "--all":
        if MODELS_DIR.exists():
            targets = sorted(MODELS_DIR.glob("*.m1model"))
        else:
            print(f"模型目录不存在: {MODELS_DIR}")
            sys.exit(1)
    else:
        targets = [Path(sys.argv[1])]

    all_ok = True
    for filepath in targets:
        filepath = str(filepath)
        if not os.path.exists(filepath):
            print(f"[ERROR] 文件不存在: {filepath}")
            all_ok = False
            continue

        with open(filepath, "rb") as f:
            data = f.read()

        info = parse_m1model(data, filepath)
        name = os.path.basename(filepath)
        model_key = name.replace(".m1model", "")
        warnings = check_expected(model_key, info)
        print_report(filepath, info, warnings)

        if "error" in info:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
