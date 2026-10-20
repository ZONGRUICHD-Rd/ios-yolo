# iOS YOLO26 实时检测

在 iPhone（面向 iPhone 18 Pro / A20 Pro）上用相机实时运行 Ultralytics 最新的 **YOLO26** 目标检测模型，可在 **NPU（Apple Neural Engine）/ GPU / CPU / Auto** 之间即时切换，并内置三种计算单元的跑分对比。IPA 由 GitHub Actions 构建，不签名。

## 功能

- 后置摄像头 1080p@60fps 实时检测，绘制检测框、类别与置信度
- 计算单元切换（Core ML `MLComputeUnits`）：
  | 选项 | Core ML 设置 | 说明 |
  | --- | --- | --- |
  | NPU | `.cpuAndNeuralEngine` | Apple Neural Engine，CPU 仅承担 ANE 不支持的算子 |
  | GPU | `.cpuAndGPU` | Metal GPU（Core ML 没有"纯 GPU"选项，这是最接近的） |
  | CPU | `.cpuOnly` | 仅 CPU |
  | Auto | `.all` | 由系统调度 |
- 模型切换（CI 默认打包 `yolo26n`、`yolo26s`）、置信度阈值调节
- 实时显示 FPS、推理耗时、后处理耗时、目标数
- **跑分对比**：依次在 NPU / GPU / CPU 上测 50 次推理的平均/中位/最快耗时，并用 `MLComputePlan`（iOS 17.4+）显示各设备实际承担的算子占比，验证模型确实跑在所选单元上
- 同时支持 YOLO26 的 NMS-free 端到端输出（`[1,300,6]`）和 YOLO11/YOLOv8 的原始输出（`[1,84,8400]`，App 内做 NMS），可直接换成自己训练的模型

## 构建 IPA（GitHub Actions）

工作流：`.github/workflows/build-ipa.yml`，运行在 `macos-26`（Xcode 26）。

1. push 到 `main`/`master`、提 PR 或在 Actions 页面手动运行 **Build unsigned IPA**
2. 手动运行时可填写要打包的模型（如 `yolo26n yolo26s yolo26m`）和权重精度（FP16 / INT8 / FP32）
3. 在运行详情页的 **Artifacts** 下载 `YOLOApp-unsigned-ipa`
4. 推送 `v*` 标签（如 `v1.0.0`）时，IPA 会自动附加到 GitHub Release

流程：`pip install ultralytics` → `export_models.py` 导出 `.mlpackage`（`nms=False` 保留 YOLO26 的端到端头）→ `coremlcompiler` 编译为 `.mlmodelc` → XcodeGen 生成工程 → `xcodebuild`（`CODE_SIGNING_ALLOWED=NO`）→ 打包 `Payload/` 为 `.ipa`。

## 安装

IPA 未签名，需要用 Sideloadly、AltStore、SideStore、TrollStore 或自己的证书（如 `zsign` / Xcode）重签后安装。首次打开需要授予相机权限。

## 本地开发（macOS）

```bash
brew install xcodegen
pip install -r scripts/requirements.txt
python scripts/export_models.py yolo26n yolo26s --out Models   # .mlpackage 会被 Xcode 当作文件夹拷贝，需要先编译：
for p in Models/*.mlpackage; do xcrun coremlcompiler compile "$p" Models/ && rm -rf "$p"; done
xcodegen generate && open YOLOApp.xcodeproj
```

在 Xcode 里选择自己的开发团队即可真机调试。

使用自定义模型：把训练好的 `best.pt` 放进仓库，手动运行工作流时 `models` 填 `best.pt` 即可（类别名从模型元数据读取）。

## 代码结构

```
YOLOApp/Sources/
  YOLODetector.swift          Core ML 加载、Vision 推理、输出解码 / NMS、跑分
  ComputePlanInspector.swift  MLComputePlan 分析每个算子被分配到 NPU/GPU/CPU
  CameraManager.swift         AVCaptureSession，竖屏 BGRA 帧
  DetectorViewModel.swift     模型加载/切换、逐帧处理、跑分调度
  ContentView.swift           主界面（预览、HUD、控制面板）
  DetectionOverlay.swift      检测框绘制
  BenchmarkView.swift         跑分结果页
scripts/export_models.py      YOLO → Core ML 导出
project.yml                   XcodeGen 工程定义
```

## 说明

- 首次以 NPU 模式加载某个模型时，系统需要为 ANE 编译，可能耗时数秒；之后会被缓存。
- 实时模式的"推理"耗时包含 Vision 的缩放预处理；跑分页是纯 `MLModel.prediction` 耗时，更适合比较计算单元。
