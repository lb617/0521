
=======
# A1 部署工作区

集创赛 · 小鼠行为学项目 · 飞凌微 A1（SmartSens SC235HAI）板上推理部署。

## 上手

```powershell
git clone https://github.com/lb617/0521 A1_deploy
cd A1_deploy
```

编译：
```bash
cd A1_SDK_SC235HAI/smartsens_sdk
./scripts/a1_sc235hai_build.sh
# 产物：output/images/zImage.smartsens-m1-evb
```

烧录用 Aurora + CH347 接口。详细 pipeline 见 vault `workbench/A1-board-deployment-reference.md`。

## 目录

```
README.md              # 本文件
EXPERIMENT_RULES.md    # 几条小约定（防忘）
DEVLOG.md              # 问题日志，碰到什么就往里加
.gitignore

A1_SDK_SC235HAI/       # SDK fork（不要乱改）
    └── ...smart_software/src/app_demo/face_detection/ssne_ai_demo/
        ├── mgs.cpp / include/ / src/ / ...

models/
├── onnx/              # PyTorch 导出的 ONNX
└── a1model/           # 思思 AI 助手转换后的 .a1model

scripts/               # verify_model.py 等小工具
tests/images/          # 5 张固定测试图
ai-chats/              # AI 调代码会话存档（每次结束扔一份）
reports/               # 每周一份简短同步
```

## 几个不会变的约定

- 改完 commit 一下，message 写啥都行，**当天有就好**
- 和 AI 调代码完，把对话存到 `ai-chats/YYYY-MM-DD-话题.md` 再 commit
- 任何模型改动前后跑一次 `scripts/verify_model.py`，确认数值没崩
- 每周日晚发一份 `reports/周报-YYYY-MM-DD.md`，3 个问题各一两句话即可

详见 `EXPERIMENT_RULES.md`。
>>>>>>> 7fc82cee8db2b1f2f2bb27b93c32c2edbb251c7a
