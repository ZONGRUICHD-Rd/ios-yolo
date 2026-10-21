import Combine
import CoreGraphics
import CoreVideo
import Foundation
import QuartzCore

struct FrameStats {
    var fps: Double = 0
    var inferenceMs: Double = 0
    var postprocessMs: Double = 0
    var objectCount = 0
    var tagMs: Double = 0
    var tagCount = 0
    var intrinsicsFromCamera = false
}

enum DetectionMode: String, CaseIterable, Identifiable {
    case yolo
    case aprilTag
    case both

    var id: String { rawValue }

    var title: String {
        switch self {
        case .yolo: return "YOLO"
        case .aprilTag: return "AprilTag"
        case .both: return "同时"
        }
    }

    var runsYOLO: Bool { self != .aprilTag }
    var runsAprilTag: Bool { self != .yolo }
}

struct BenchmarkResult: Identifiable {
    let mode: ComputeMode
    var loadMs: Double?
    var stats: BenchmarkStats?
    var breakdown: DeviceBreakdown?
    var error: String?

    var id: String { mode.id }
}

final class DetectorViewModel: ObservableObject {
    let models = ModelInfo.bundled()
    let camera = CameraManager()

    @Published var selectedModel: ModelInfo? {
        didSet { if selectedModel != oldValue { loadModel() } }
    }
    @Published var computeMode: ComputeMode = .neuralEngine {
        didSet { if computeMode != oldValue { loadModel() } }
    }
    @Published var confidence: Double = 0.25 {
        didSet {
            let value = Float(confidence)
            camera.videoQueue.async { self.confidenceThreshold = value }
        }
    }

    @Published var detectionMode: DetectionMode = .yolo {
        didSet {
            let mode = detectionMode
            camera.videoQueue.async {
                self.activeMode = mode
                if !mode.runsAprilTag { self.tagTracker.reset() }
            }
        }
    }
    /// Edge length of the tag's outer black square, in centimeters.
    @Published var tagSizeCM: Double = 6.9 {
        didSet {
            let meters = tagSizeCM / 100
            camera.videoQueue.async { self.activeTagSize = meters }
        }
    }
    @Published var tagOrientation: TagPatternOrientation = .rotated180 {
        didSet {
            let orientation = tagOrientation
            camera.videoQueue.async {
                self.activeTagOrientation = orientation
                self.tagTracker.reset()
            }
        }
    }

    @Published private(set) var detections: [Detection] = []
    @Published private(set) var tags: [TagOverlay] = []
    @Published private(set) var imageSize = CGSize(width: 1080, height: 1920)
    @Published private(set) var stats = FrameStats()
    @Published private(set) var isLoading = false
    @Published private(set) var statusMessage: String?
    @Published private(set) var errorMessage: String?

    @Published private(set) var benchmarkResults: [BenchmarkResult] = []
    @Published private(set) var isBenchmarking = false

    private let loadQueue = DispatchQueue(label: "yolo.model.load", qos: .userInitiated)
    private var loadGeneration = 0
    private var hasStarted = false

    // Confined to camera.videoQueue.
    private var detector: YOLODetector?
    private var confidenceThreshold: Float = 0.25
    private var isPaused = false
    private var activeMode: DetectionMode = .yolo
    private var activeTagSize = 0.069
    private var activeTagOrientation: TagPatternOrientation = .rotated180
    private lazy var tagDetector = AprilTagDetector()
    private let tagTracker = AprilTagTracker()
    private var lastFrameTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

    init() {
        selectedModel = models.first
        camera.onFrame = { [weak self] frame in self?.process(frame) }
    }

    func start() {
        camera.start { [weak self] error in
            if let error { self?.errorMessage = error }
        }
        if !hasStarted {
            hasStarted = true
            loadModel()
        }
    }

    // MARK: - Model loading

    private func loadModel() {
        guard let model = selectedModel else {
            errorMessage = "App 内没有模型，请检查构建流程中的模型导出步骤"
            return
        }
        let mode = computeMode
        loadGeneration += 1
        let generation = loadGeneration
        isLoading = true
        statusMessage = nil

        loadQueue.async {
            let result = Result { try YOLODetector(info: model, computeMode: mode) }
            if case let .success(detector) = result {
                self.camera.videoQueue.async {
                    self.detector = detector
                    self.smoothedFPS = 0
                }
            }
            DispatchQueue.main.async {
                guard generation == self.loadGeneration else { return }
                self.isLoading = false
                switch result {
                case let .success(detector):
                    self.errorMessage = nil
                    self.statusMessage = String(format: "%@ 已加载到 %@，用时 %.0f ms", model.name, mode.title, detector.loadTime * 1000)
                case let .failure(error):
                    self.errorMessage = "模型加载失败：\(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Per-frame processing (camera.videoQueue)

    private func process(_ frame: CameraFrame) {
        guard !isPaused else { return }
        let pixelBuffer = frame.pixelBuffer
        let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        var stats = FrameStats()
        var detections: [Detection] = []
        var tags: [TagOverlay] = []
        var failure: String?

        if activeMode.runsYOLO, let detector {
            do {
                let result = try detector.detect(pixelBuffer: pixelBuffer, confidence: confidenceThreshold)
                detections = result.detections
                stats.inferenceMs = result.inferenceMs
                stats.postprocessMs = result.postprocessMs
                stats.objectCount = result.detections.count
            } catch {
                failure = "推理失败：\(error.localizedDescription)"
            }
        }

        if activeMode.runsAprilTag {
            let start = CACurrentMediaTime()
            let poses = tagDetector.detect(
                pixelBuffer: pixelBuffer,
                intrinsics: frame.intrinsics,
                tagSize: activeTagSize,
                orientation: activeTagOrientation
            )
            stats.tagMs = (CACurrentMediaTime() - start) * 1000
            stats.tagCount = poses.count
            stats.intrinsicsFromCamera = frame.intrinsics.isFromCamera
            tags = tagTracker.update(with: poses, intrinsics: frame.intrinsics, imageSize: size, at: start)
        }

        let now = CACurrentMediaTime()
        if lastFrameTime > 0 {
            let instant = 1 / max(now - lastFrameTime, 1e-6)
            smoothedFPS = smoothedFPS == 0 ? instant : smoothedFPS * 0.9 + instant * 0.1
        }
        lastFrameTime = now
        stats.fps = smoothedFPS

        DispatchQueue.main.async {
            self.detections = detections
            self.tags = tags
            self.imageSize = size
            self.stats = stats
            if let failure { self.errorMessage = failure }
        }
    }

    // MARK: - Benchmark

    /// Loads the selected model on NPU, GPU and CPU in turn and times raw predictions.
    func runBenchmark(iterations: Int = 50) {
        guard let model = selectedModel, !isBenchmarking else { return }
        isBenchmarking = true
        benchmarkResults = []
        camera.videoQueue.async {
            self.isPaused = true
            self.tagTracker.reset()
            DispatchQueue.main.async {
                self.detections = []
                self.tags = []
            }
        }

        Task.detached(priority: .userInitiated) {
            for mode in [ComputeMode.neuralEngine, .gpu, .cpu] {
                var result = BenchmarkResult(mode: mode)
                do {
                    let detector = try YOLODetector(info: model, computeMode: mode)
                    result.loadMs = detector.loadTime * 1000
                    result.stats = try detector.benchmark(iterations: iterations, warmup: 5)
                } catch {
                    result.error = error.localizedDescription
                }
                result.breakdown = await ComputePlanInspector.breakdown(modelURL: model.url, mode: mode)
                let finished = result
                await MainActor.run { self.benchmarkResults.append(finished) }
            }
            await MainActor.run { self.isBenchmarking = false }
            self.camera.videoQueue.async {
                self.isPaused = false
                self.lastFrameTime = 0
            }
        }
    }
}
