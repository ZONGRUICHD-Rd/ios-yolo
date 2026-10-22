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
- **AprilTag 追踪与 3D 坐标系**（tag36h11）：画出每个 Tag 的轮廓、ID 和轨迹，并按 ID 跨帧追踪（短暂丢失时保留 0.3 秒）。以每个 Tag 的**左上角**为原点建立右手坐标系：X 沿上边向右，Y 向上，Z 垂直指出 Tag（朝向相机），与 OpenCV/ArUco 的标记坐标系方向一致。实时显示原点在相机坐标系下的位置 (cm) 和距离
- 检测模式：YOLO / AprilTag / 同时
- 同时支持 YOLO26 的 NMS-free 端到端输出（`[1,300,6]`）和 YOLO11/YOLOv8 的原始输出（`[1,84,8400]`，App 内做 NMS），可直接换成自己训练的模型

## 构建 IPA（GitHub Actions）

工作流：`.github/workflows/build-ipa.yml`，运行在 `macos-26`（Xcode 26）。

1. push 到 `main`/`master`、提 PR 或在 Actions 页面手动运行 **Build unsigned IPA**
2. 手动运行时可填写要打包的模型（如 `yolo26n yolo26s yolo26m`）和权重精度（FP16 / INT8 / FP32）
3. 在运行详情页的 **Artifacts** 下载 `YOLOApp-unsigned-ipa`
4. 发版：手动运行时在 `release_tag` 填版本号（如 `v1.0.1`），或推送 `v*` 标签，都会创建 GitHub Release 并附上 IPA；发版说明取自 `docs/release-notes/<版本号>.md`（没有则用 GitHub 自动生成的）

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

## AprilTag 说明

- **Tag 边长**：在 App 里设置 Tag 外圈黑色正方形的边长（不含白边）。默认 6.9 cm，是按 tag36h11 ID 0–5 那张 A4 图 100% 打印估算的；位姿和距离的单位以此为准，请用尺子量一下实际打印的尺寸。
- **图案方向**：那张 A4 图里的图案相对 AprilTag 官方参考图（`tag36_11_00000.png` 等）旋转了 180°，两者的"左上角"是对角。默认"打印图"对应那张 A4（ID 文字在 Tag 下方时的左上角）；如果打印的是官方 PNG，切到"官方图"。
- **相机内参**：优先使用 AVFoundation 每帧给出的内参矩阵；拿不到时按当前格式的视场角估算（HUD 显示"FOV 估算"），此时距离会有几个百分点的误差。
- **稳定性**（`AprilTag/AprilTagTracker.c`）：
  - *平面位姿二义性*：Tag 较小或接近正对相机时，两个位姿解的误差几乎一样，逐帧取误差最小的那个会来回翻转（Z 轴乱跳）。现在保留 AprilTag 给出的两个候选解，按以下顺序选择：
    1. **板平面约束**：≥3 个 Tag 在同一平面上时（如那张 A4），用各 Tag 中心拟合平面（位置不受二义性影响），选法向与平面一致的解，平面法向再做时间平滑。思路来自 Ch'ng et al., *Resolving Marker Pose Ambiguity by Robust Rotation Averaging with Clique Constraints* (ICRA 2020)；
    2. **时间一致性**：没有板平面时，沿用离上一帧位姿最近的解，误差持续明显更小才切换，类似 ArUco 的 `MarkerPoseTracker`；
  - *抖动*：对位置和旋转（四元数 slerp）做 [One Euro 滤波](https://gery.casiez.net/1euro/)（Casiez et al., CHI 2012），静止时强平滑、快速移动时自动放开，App 里"平滑"滑块可调。
  - 在仿真里（把那张 A4 按已知位姿渲染到 iPhone 相机，加噪声，和真值比较），1.8–2.4 m 远处选错解的比例从 25–34% 降到 1–10%，静止时抖动降低 6–10 倍。
- 检测用的是 [AprilTag 3](https://github.com/AprilRobotics/apriltag) C 库（v3.4.5，BSD-2，`ThirdParty/apriltag`），直接读取相机 420f 帧的亮度平面，在 CPU 多线程运行；位姿由 `estimate_tag_pose` 求解。

## 代码结构

```
YOLOApp/Sources/
  YOLODetector.swift          Core ML 加载、Vision 推理、输出解码 / NMS、跑分
  ComputePlanInspector.swift  MLComputePlan 分析每个算子被分配到 NPU/GPU/CPU
  CameraManager.swift         AVCaptureSession，竖屏 420f 帧 + 每帧相机内参
  DetectorViewModel.swift     模型加载/切换、逐帧处理、跑分调度
  ContentView.swift           主界面（预览、HUD、控制面板）
  DetectionOverlay.swift      检测框绘制
  BenchmarkView.swift         跑分结果页
  AprilTag/AprilTagBridge.c   AprilTag C 库的精简封装（检测 + 位姿）
  AprilTag/AprilTagTracker.c  位姿二义性消解（板平面 + 时间一致性）+ One Euro 滤波
  AprilTag/AprilTagDetector.swift  左上角坐标系换算、轨迹、投影
ThirdParty/apriltag/          AprilTag 3 源码（仅 tag36h11）
scripts/export_models.py      YOLO → Core ML 导出
project.yml                   XcodeGen 工程定义
```

## 说明

- 首次以 NPU 模式加载某个模型时，系统需要为 ANE 编译，可能耗时数秒；之后会被缓存。
- 实时模式的"推理"耗时包含 Vision 的缩放预处理；跑分页是纯 `MLModel.prediction` 耗时，更适合比较计算单元。
