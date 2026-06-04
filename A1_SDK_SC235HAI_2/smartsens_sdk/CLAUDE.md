# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

这是 **Buildroot 2022.02**，针对 **SmartSens SC235HAI (M1 Pro)** SoC 进行了定制 —— 一款带有自研神经网络引擎 SSNE (SmartSens Neural Engine) 的 ARM Cortex-A7 嵌入式平台。本仓库构建一套完整的嵌入式 Linux 系统：交叉编译工具链、Linux 5.15.24 内核、基于 Busybox 的根文件系统、预编译的 M1 SDK 用户空间库，以及一个 AI 人脸检测演示程序。

## 端到端构建流程

构建由 shell 脚本编排，这些脚本封装了 Buildroot 基于 make 的构建系统。以下是各个环节如何串联的：

### 1. 依赖下载（`scripts/build_dl.sh`）

将三个压缩包下载到 `cache/` 目录，并进行 MD5 校验：
- **Linux 内核 5.15.24** → `cache/linux-5.15.24.tar.gz`（来自 kernel.org 镜像）
- **预编译工具链** → `cache/glibc-ssp-cpp.tar.gz`（来自 gitee.com）
- **预编译软件包** → `cache/package.tar.gz`（来自 gitee.com）—— 包含 `output/opt/m1_sdk/`（M1 SDK 库 + 内核模块）和 `output/build/` 编译产物

### 2. 完整构建（`scripts/build_release_sdk.sh`）

这是主构建脚本，按顺序执行以下操作：

1. **调用 `build_dl.sh`** 确保所有压缩包已下载
2. **解压工具链** 到 `smart_software/toolchain/glibc-ssp-cpp/` —— 提供 `arm-smartsens-linux-gnueabihf_sdk-buildroot/`（GCC 10，glibc，Cortex-A7，硬浮点 NEON/VFPv4）
3. **解压预编译软件包** 到顶层 `package/` 目录 —— 提供缓存的 Buildroot 下载产物，使构建不需要联网
4. **解压内核源码** 到 `smart_software/src/linux-5.15.24/`，然后应用 `patch/0001-Add-support-of-A1-sc235hai.patch` 补丁以添加 SMARTSOC 平台支持
5. **配置 Buildroot**：`make BR2_EXTERNAL=./smart_software smartsens_m1pro_release_defconfig`
6. **编译所有内容**：`make -j32`

构建成功后，最终产物位于 `output/images/` —— 主要是合并后的 zImage（`zImage.smartsens-m1-evb`），它将内核 + DTB + initramfs CPIO 打包成一个可直接启动的镜像。

### 3. 增量应用重编译（`scripts/build_app.sh`）

用于修改 AI 演示源码后的快速迭代：
```bash
rm -rf output/build/ssne_ai_demo/
make ssne_ai_demo-rebuild
```
这会删除包的构建目录，强制 Buildroot 只重新配置/编译这一个包，然后重新打包根文件系统。

### 4. 编排脚本（`scripts/a1_sc235hai_build.sh`）

顶层封装脚本：
1. 检查内核 zImage 是否存在（如果已存在则跳过首次完整构建）
2. 如果需要，运行 `build_release_sdk.sh`（首次构建）
3. 运行 `build_app.sh` 重编译 AI 演示程序
4. 再次运行 `build_release_sdk.sh` 生成最终镜像
5. 将 zImage 复制到 `/home/a1_output/`（CI/部署目录）

## Buildroot 外部层机制

`smarter_software/` 目录是一个 **Buildroot 外部树**（`BR2_EXTERNAL`）。Buildroot 的顶层 `Makefile`（约第 190 行）使用 `support/scripts/br2-external` 来发现外部层，读取 `external.desc` 获取名称（`SMART_SOFTWARE`），然后包含 `external.mk`。

### 外部层入口点

```
smart_software/
├── external.desc          ← "name: SMART_SOFTWARE" — 标识外部层
├── external.mk            ← 被 Buildroot 包含；设置 S1SRC，包含所有包的 .mk 文件
├── local.mk               ← 通过 BR2_PACKAGE_OVERRIDE_FILE 被包含；覆盖 LINUX_OVERRIDE_SRCDIR
├── Config.in              ← 被 Buildroot 的 menuconfig 包含；引用各包的 Config.in 文件
├── configs/               ← Buildroot defconfig + busybox 配置
├── board/m1pro/           ← 内核 defconfig + 根文件系统 overlay
├── package/               ← 每个包的构建定义
├── toolchain/             ← 预编译的交叉编译器
└── src/                   ← 内核源码树（已打补丁）+ 应用演示源码
```

### `external.mk` 如何串联各包

```makefile
export MYBOOTDIR=$(BR2_EXTERNAL_SMART_SOFTWARE_PATH)/package/boot
export S1SRC=$(BR2_EXTERNAL_SMART_SOFTWARE_PATH)/src
include $(sort $(wildcard $(BR2_EXTERNAL_SMART_SOFTWARE_PATH)/package/boot/*/*.mk))
include $(sort $(wildcard $(BR2_EXTERNAL_SMART_SOFTWARE_PATH)/package/*/*.mk))
```

这段代码通过 glob 模式包含 `package/` 下的所有 `.mk` 文件，因此添加新包只需创建一个带有 `.mk` 和 `Config.in` 的新子目录 —— 无需其他注册步骤。

### `local.mk` 如何覆盖内核源码路径

```makefile
LINUX_OVERRIDE_SRCDIR=$(CONFIG_DIR)/smart_software/src/linux-5.15.24
```

这告诉 Buildroot 使用本地已打补丁的内核树，而不是下载新的 tarball。在 `build_release_sdk.sh` 执行期间，内核被解压到此处并打补丁。Buildroot 的 `linux.mk` 会识别 `LINUX_OVERRIDE_SRCDIR` 并从该目录构建。

## 包构建系统详解

### `m1_sdk_lib` —— 预编译 SDK 库

**源路径**：`output/opt/m1_sdk/`（从 `cache/package.tar.gz` 解压得到）

此包将预编译的二进制文件复制到目标根文件系统中。该目录包含：
- `usr/lib/` —— 共享库：`libssne.so`（神经网络引擎运行时）、`libcmabuffer.so`（CMA 缓冲区管理）、`libosd.so`（屏幕显示）、`libemb.so`（嵌入式运行时）、`libzlog.so`/`libsszlog.so`（日志）、`libgpio.so`、`libuart.so`，以及系统库（`libstdc++`、`libfreetype`、elfutils）
- `usr/include/smartsoc/` —— C/C++ 头文件：`ssne_api.h`、`cmabuffer.h`、`osd_lib_api.h`、`libemb_api.h`、`zlog.h` 等
- `extra/` —— 外部内核模块：`ddr_mmap.ko`、`ocm.ko`、`emb.ko`、`preoffline.ko`、`preonpipe.ko`、`lnpu.ko`、`osd_kmod.ko`、`isp_debug.ko`、`axi_dma.ko`、`aec.ko`、`uart_kmod.ko`、`cmdset.ko`、`gpio_kmod.ko`

`.mk` 文件的复制操作：
- `usr/lib/*` → `$(TARGET_DIR)/usr/lib`
- `usr/include/smartsoc/*` → `$(TARGET_DIR)/usr/include/smartsoc/`
- `extra/*.ko` → `$(TARGET_DIR)/lib/modules/5.15.24/extra/`

### `ssne_ai_demo` —— AI 人脸检测演示程序

**源路径**：`smart_software/src/app_demo/face_detection/ssne_ai_demo/`（具体的应用由 defconfig 中的 `BR2_PACKAGE_SSNE_AI_DEMO_APP` 配置变量选择，当前设为 `"face_detection"`）

**构建类型**：`cmake-package` —— Buildroot 的 cmake 基础设施处理交叉编译

**编译流程**：
1. Buildroot 的 cmake-package 基础设施设置交叉编译环境（CC、CXX、sysroot 等）
2. `CMakeLists.txt` 包含 `cmake_config/Paths.cmake`，通过 `$ENV{BASE_DIR}/$ENV{EXPORT_LIB_M1_SDK_ROOT_PATH}` 解析 SDK 路径 —— 指向 `output/opt/m1_sdk/usr`
3. 通过 glob 收集源文件（`src/*.cpp`）并与主文件 `demo_face_yunet.cpp` 一起编译
4. 链接库：`libssne.so`、`libcmabuffer.so`、`libosd.so`、`libsszlog.so`、`libzlog.so`、`libemb.so`
5. 输出的二进制文件 `ssne_ai_demo` 安装到 `$(TARGET_DIR)/app_demo/`
6. `app_assets/` 目录（模型文件、颜色 LUT）和 `scripts/run.sh` 也复制到 `/app_demo/`

**应用架构**（YuNet 人脸检测流水线）：
- `demo_face_yunet.cpp` —— 主循环：初始化 SSNE，创建 `IMAGEPROCESSOR` + `YUNET` 检测器 + `VISUALIZER`，循环处理传感器帧
- `IMAGEPROCESSOR`（来自 `common.hpp`）—— 打开 SSNE 在线管线 0，配置裁剪（1920→1440×1080），从传感器获取 YUV422_16 格式帧
- `YUNET`（来自 `common.hpp`）—— 创建 SSNE 离线预处理管线（YUV422→RGB 缩放至 160×120），加载 `.m1model`，运行 NPU 推理，获取 4 个输出头（NHWC 布局的 loc/conf/iou），解码 + NMS 后处理
- `VISUALIZER`（来自 `utils.hpp`）—— 使用 `OsdDevice`（来自 `osd_lib_api.h`）在 OSD 图层上绘制边界框和位图
- 键盘监听线程等待输入 'q' 来干净退出

### 启动引导包

`smarter_software/package/boot/` 目录当前为空（没有自定义引导加载程序包）。系统使用内核内置的 initramfs 加上追加的 DTB —— 此外部层不构建单独的引导加载程序包。

## 内核构建过程

### 内核配置

defconfig `smartsens_m1pro_release_defconfig` 中的设置：
```
BR2_LINUX_KERNEL_CUSTOM_VERSION_VALUE="5.15.24"
BR2_LINUX_KERNEL_CUSTOM_CONFIG_FILE="...board/m1pro/smartsens_demo_linux_defconfig"
BR2_LINUX_KERNEL_APPENDED_ZIMAGE=y
BR2_LINUX_KERNEL_INTREE_DTS_NAME="smartsens-m1-evb"
```

关键选项：
- `BR2_LINUX_KERNEL_APPENDED_ZIMAGE=y` —— 将 DTB 和 initramfs CPIO 直接追加到内核 zImage 后面，生成单一可启动镜像
- `BR2_LINUX_KERNEL_INTREE_DTS_NAME="smartsens-m1-evb"` —— 将树内 DTS（由补丁添加）编译为 DTB 并追加

### 内核补丁内容（`patch/0001-Add-support-of-A1-sc235hai.patch`）

向内核树添加了 21 个文件：

| 文件 | 用途 |
|------|------|
| `arch/arm/mach-smartsens/` | 新机器平台（`ARCH_SMARTSOC`） |
| `arch/arm/boot/dts/smartsens-m1.dtsi` | SoC 级 DTS 包含文件（GIC、外设、CMA 池） |
| `arch/arm/boot/dts/smartsens-m1-evb.dts` | 评估板 DTS（64MB RAM，4MB OSD CMA @ 0x5800000，36MB AI CMA @ 0x5c00000） |
| `arch/arm/boot/dts/smartsens-m1-fpga.dts` | FPGA 板 DTS（128MB，不同的 CMA 布局） |
| `drivers/spi/spi-smarts1-qspi.c` | 自定义 QSPI Flash 控制器驱动（769 行） |
| `drivers/tty/serial/smarts1-uart.c` | 自定义 UART 驱动（768 行） |
| `arch/arm/include/debug/smarts1.S` | 底层调试 UART 例程 |
| `drivers/dma-buf/heaps/cma_heap.c` | CMA heap 小修改 |
| `drivers/mtd/spi-nor/gigadevice.c` | 添加 Flash 芯片 ID |
| `drivers/char/random.c` | 删除 2 行（可能是构建修复） |

内核配置（`smartsens_demo_linux_defconfig`）启用了：
- `CONFIG_ARCH_SMARTSOC=y` —— 激活新机器平台
- `CONFIG_SERIAL_SMARTS1=y` —— 自定义 UART
- `CONFIG_CMA=y`、`CONFIG_DMA_CMA=y`、`CONFIG_DMABUF_HEAPS=y` —— 为 SSNE 和 OSD 提供 CMA/DMABUF
- `CONFIG_ARM_APPENDED_DTB=y` —— 追加 DTB
- `CONFIG_INITRAMFS_SOURCE="${BR_BINARIES_DIR}/rootfs.cpio"` —— 嵌入式 initramfs

## 根文件系统构建与启动顺序

### 根文件系统 Overlay

来自 `smart_software/board/m1pro/rootfs_overlay/` 的文件会直接覆盖到 Busybox 生成的根文件系统之上：

| Overlay 文件 | 作用 |
|-------------|------|
| `etc/inittab` | Busybox init：系统初始化时运行 `rcS`，在控制台启动 ash，关机时运行 `rcK` |
| `etc/fstab` | 挂载 proc、sysfs、dev（tmpfs）、tmp、run、var 为 tmpfs |
| `etc/hostname` | 设置主机名为 `smartsens_s1` |
| `etc/init.d/rcS` | 启动脚本：挂载所有文件系统，启动 mdev（设备管理器），然后执行 `smartsoc_start.sh` |
| `etc/init.d/rcK` | 关机脚本：按逆序停止所有 init 脚本 |
| `usr/smartsoc/smartsoc_start.sh` | **主应用启动器** —— insmod 加载所有内核模块，然后 `cd app_demo && ./scripts/run.sh` |
| `usr/bin/image_dump` | 预编译 ARM 二进制 —— DDR 内存调试工具（通过 `/dev/mmapiomem` 读写物理内存） |
| `usr/debug.user.ini` | 图像预处理调试参数（裁剪、填充、归一化配置） |

### 运行时启动顺序

```
内核启动 → initramfs 解包 → /sbin/init (Busybox)
  → /etc/inittab: sysinit:/etc/init.d/rcS
    → mount -a（根据 fstab 挂载）
    → mdev -s（冷插拔设备扫描）+ mdev -d（热插拔守护进程）
    → /usr/smartsoc/smartsoc_start.sh
      → insmod ddr_mmap.ko, ocm.ko, emb.ko, preoffline.ko, preonpipe.ko
      → insmod lnpu.ko（NPU 驱动）, osd_kmod.ko, isp_debug.ko
      → insmod axi_dma.ko, aec.ko, uart_kmod.ko
      → cd /app_demo && ./scripts/run.sh
        → ./ssne_ai_demo（YuNet 人脸检测主循环）
  → console::askfirst:/bin/ash（串口控制台可用）
```

### CMA 内存布局（EVB 评估板）

来自设备树：
- **OSD CMA**：4MB，位于 `0x05800000` —— 帧缓冲/叠加层
- **AI CMA**：36MB，位于 `0x05c00000` —— SSNE NPU 工作内存（模型权重、张量、预处理缓冲区）
- 总 RAM：64MB（剩余约 24MB 用于 Linux 内核 + 用户空间）

## 关键关联关系总览

```
build_dl.sh ──下载──→ cache/（内核、工具链、软件包压缩包）
                                │
build_release_sdk.sh ──解压──→ smart_software/toolchain/（交叉编译器）
                              ├→ smart_software/src/linux-5.15.24/（已打补丁的内核）
                              ├→ package/（缓存的 Buildroot 下载文件）
                              └→ output/opt/m1_sdk/（预编译 SDK 库 + .ko 模块）
                                      │
make BR2_EXTERNAL=./smart_software  │
  ├─ 读取 .defconfig → smart_software/configs/smartsens_m1pro_release_defconfig
  ├─ 包含 smart_software/external.mk → glob 匹配 package/*/*.mk
  ├─ 包含 smart_software/local.mk → LINUX_OVERRIDE_SRCDIR 指向已打补丁的内核
  ├─ 构建主机工具（cmake、DTC 等）
  ├─ 构建工具链封装器
  ├─ 从已打补丁的源码构建 linux-custom（zImage + DTB + 追加的 CPIO）
  ├─ 构建 m1_sdk_lib（将预编译文件从 output/opt/m1_sdk/ 复制到 target）
  ├─ 构建 ssne_ai_demo（cmake 交叉编译，链接 SDK 库）
  ├─ 生成 rootfs.cpio.gz（Busybox + overlay + 各包）
  └─ 生成 output/images/zImage.smartsens-m1-evb（最终可启动镜像）
```

## 手动 Buildroot 命令

```bash
# 配置
make BR2_EXTERNAL=./smart_software smartsens_m1pro_release_defconfig

# 完整构建
make -j32

# 仅重编译 AI 演示程序（修改 smart_software/src/app_demo/... 后）
rm -rf output/build/ssne_ai_demo/ && make ssne_ai_demo-rebuild

# 仅重编译内核
make linux-rebuild

# 内核配置
make linux-menuconfig

# Buildroot 配置
make menuconfig

# 列出所有可用的 defconfig
make list-defconfigs
```

## 代码风格

- `.clang-format` —— Linux 内核风格（从 Linux 5.15.6 导入），用于树内 C 文件
- `.flake8` —— Buildroot 支持脚本的 Python 代码检查配置
- 内核空间代码遵循 Linux 内核编码风格
- 用户空间 C++（AI 演示程序）使用 C++11 标准配合 `-O2` 优化，通过 CMake 编译，附加 `-Wno-psabi` 选项

## 仓库目录布局（仅重要目录）

```
smartsens_sdk/
├── scripts/                 # 构建编排 shell 脚本
├── smart_software/          # Buildroot 外部层（核心定制点）
│   ├── configs/             # Buildroot + busybox defconfig
│   ├── board/m1pro/         # 内核 defconfig + 根文件系统 overlay
│   ├── package/             # 每个包的 Buildroot 定义
│   ├── toolchain/           # 解压后的预编译交叉编译器
│   └── src/                 # 内核源码（解压+补丁）+ 应用演示源码
├── patch/                   # 内核平台补丁
├── cache/                   # 下载的压缩包（内核、工具链、软件包）
├── output/                  # Buildroot 输出目录
│   ├── build/               # 每个包的构建目录
│   ├── images/              # 最终镜像（追加了 DTB+initramfs 的 zImage）
│   ├── target/              # 生成 CPIO 前的临时根文件系统
│   ├── host/                # 主机工具（交叉编译封装器、cmake、DTC 等）
│   └── opt/m1_sdk/          # 预编译 SDK 库和内核模块
├── Makefile                 # Buildroot 顶层 makefile
├── Config.in                # Buildroot 顶层 Kconfig
└── .clang-format            # C 代码格式化规则
```
