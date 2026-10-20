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

    @Published private(set) var detections: [Detection] = []
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
    private var lastFrameTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

    init() {
        selectedModel = models.first
        camera.onFrame = { [weak self] buffer in self?.process(buffer) }
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

    private func process(_ pixelBuffer: CVPixelBuffer) {
        guard !isPaused, let detector else { return }
        do {
            let result = try detector.detect(pixelBuffer: pixelBuffer, confidence: confidenceThreshold)
            let now = CACurrentMediaTime()
            if lastFrameTime > 0 {
                let instant = 1 / max(now - lastFrameTime, 1e-6)
                smoothedFPS = smoothedFPS == 0 ? instant : smoothedFPS * 0.9 + instant * 0.1
            }
            lastFrameTime = now

            let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
            let stats = FrameStats(
                fps: smoothedFPS,
                inferenceMs: result.inferenceMs,
                postprocessMs: result.postprocessMs,
                objectCount: result.detections.count
            )
            DispatchQueue.main.async {
                self.detections = result.detections
                self.imageSize = size
                self.stats = stats
            }
        } catch {
            DispatchQueue.main.async { self.errorMessage = "推理失败：\(error.localizedDescription)" }
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
            DispatchQueue.main.async { self.detections = [] }
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
