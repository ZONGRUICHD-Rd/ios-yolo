import CoreML
import CoreVideo
import Foundation
import QuartzCore
import Vision

/// Which Core ML compute units the model is allowed to use.
/// Core ML has no "GPU only" option; `.cpuAndGPU` is the closest (CPU only takes ops the GPU can't run).
enum ComputeMode: String, CaseIterable, Identifiable {
    case neuralEngine
    case gpu
    case cpu
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .neuralEngine: return "NPU"
        case .gpu: return "GPU"
        case .cpu: return "CPU"
        case .all: return "Auto"
        }
    }

    var detail: String {
        switch self {
        case .neuralEngine: return "Apple Neural Engine（.cpuAndNeuralEngine）"
        case .gpu: return "Metal GPU（.cpuAndGPU）"
        case .cpu: return "仅 CPU（.cpuOnly）"
        case .all: return "系统自动调度（.all）"
        }
    }

    var computeUnits: MLComputeUnits {
        switch self {
        case .neuralEngine: return .cpuAndNeuralEngine
        case .gpu: return .cpuAndGPU
        case .cpu: return .cpuOnly
        case .all: return .all
        }
    }
}

struct Detection {
    let classIndex: Int
    let label: String
    let confidence: Float
    /// Normalized to the input image, top-left origin.
    let rect: CGRect
}

struct ModelInfo: Identifiable, Hashable {
    let name: String
    let url: URL
    /// Compiled size on disk, bytes.
    let byteCount: Int

    var id: String { name }

    private static let scales: [Character] = ["n", "s", "m", "l", "x"]

    /// Size suffix of Ultralytics names ("yolov8l" → "l"); nil for other names such as "best".
    var scale: Character? {
        guard name.lowercased().hasPrefix("yolo"), let last = name.last, Self.scales.contains(last) else { return nil }
        return last
    }

    /// "yolov8l" → "yolov8"; names without a scale are their own family.
    var family: String { scale == nil ? name : String(name.dropLast()) }

    var familyTitle: String {
        family.lowercased().hasPrefix("yolo") ? "YOLO" + family.dropFirst(4) : family
    }

    var menuTitle: String { String(format: "%@ · %.0f MB", name, Double(byteCount) / 1_048_576) }

    /// Compiled models shipped in `YOLOApp.app/Models`, grouped by family and ordered n, s, m, l, x.
    static func bundled() -> [ModelInfo] {
        guard let dir = Bundle.main.resourceURL?.appendingPathComponent("Models"),
              let urls = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return [] }
        func rank(_ model: ModelInfo) -> Int {
            model.scale.flatMap { scales.firstIndex(of: $0) } ?? scales.count
        }
        return urls
            .filter { $0.pathExtension == "mlmodelc" }
            .map { ModelInfo(name: $0.deletingPathExtension().lastPathComponent, url: $0, byteCount: directorySize($0)) }
            .sorted { ($0.family, rank($0), $0.name) < ($1.family, rank($1), $1.name) }
    }

    private static func directorySize(_ url: URL) -> Int {
        guard let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let file as URL in files {
            total += (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }
}

struct ModelGroup: Identifiable {
    let title: String
    let models: [ModelInfo]

    var id: String { title }

    /// Consecutive models of the same family, as `bundled()` orders them.
    static func groups(of models: [ModelInfo]) -> [ModelGroup] {
        var groups: [ModelGroup] = []
        for model in models {
            if let last = groups.last, last.title == model.familyTitle {
                groups[groups.count - 1] = ModelGroup(title: last.title, models: last.models + [model])
            } else {
                groups.append(ModelGroup(title: model.familyTitle, models: [model]))
            }
        }
        return groups
    }
}

struct BenchmarkStats {
    let meanMs: Double
    let minMs: Double
    let medianMs: Double
}

enum DetectorError: LocalizedError {
    case noImageInput
    case pixelBufferCreationFailed

    var errorDescription: String? {
        switch self {
        case .noImageInput: return "模型没有图像输入"
        case .pixelBufferCreationFailed: return "无法创建输入缓冲区"
        }
    }
}

/// Wraps a YOLO Core ML model. Not thread-safe: use each instance from one queue at a time.
final class YOLODetector {
    let info: ModelInfo
    let computeMode: ComputeMode
    let model: MLModel
    let classNames: [String]
    let inputName: String
    let inputSize: CGSize
    let loadTime: TimeInterval

    private let inputPixelFormat: OSType
    private let request: VNCoreMLRequest

    init(info: ModelInfo, computeMode: ComputeMode) throws {
        self.info = info
        self.computeMode = computeMode

        let config = MLModelConfiguration()
        config.computeUnits = computeMode.computeUnits
        let start = CACurrentMediaTime()
        let model = try MLModel(contentsOf: info.url, configuration: config)
        loadTime = CACurrentMediaTime() - start
        self.model = model

        guard let input = model.modelDescription.inputDescriptionsByName.values.first(where: { $0.type == .image }),
              let constraint = input.imageConstraint
        else { throw DetectorError.noImageInput }
        inputName = input.name
        inputSize = CGSize(width: constraint.pixelsWide, height: constraint.pixelsHigh)
        inputPixelFormat = constraint.pixelFormatType

        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        classNames = YOLODetector.parseClassNames(metadata?["names"])

        let request = VNCoreMLRequest(model: try VNCoreMLModel(for: model))
        // Ultralytics' own iOS app also stretches the frame to the model input.
        request.imageCropAndScaleOption = .scaleFill
        self.request = request
    }

    // MARK: - Inference

    struct Result {
        let detections: [Detection]
        let inferenceMs: Double
        let postprocessMs: Double
    }

    func detect(pixelBuffer: CVPixelBuffer, confidence: Float, iouThreshold: Float = 0.45) throws -> Result {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        let t0 = CACurrentMediaTime()
        try handler.perform([request])
        let t1 = CACurrentMediaTime()

        let output = (request.results as? [VNCoreMLFeatureValueObservation])?
            .compactMap { $0.featureValue.multiArrayValue }
            .first
        let detections = output.map { decode($0, confidence: confidence, iouThreshold: iouThreshold) } ?? []
        let t2 = CACurrentMediaTime()
        return Result(detections: detections, inferenceMs: (t1 - t0) * 1000, postprocessMs: (t2 - t1) * 1000)
    }

    /// Times raw `MLModel.prediction` on a constant input, excluding Vision's resize.
    func benchmark(iterations: Int, warmup: Int) throws -> BenchmarkStats {
        let buffer = try makeInputBuffer()
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
        for _ in 0..<warmup {
            _ = try model.prediction(from: provider)
        }
        var times: [Double] = []
        times.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            _ = try model.prediction(from: provider)
            times.append((CACurrentMediaTime() - start) * 1000)
        }
        times.sort()
        return BenchmarkStats(
            meanMs: times.reduce(0, +) / Double(times.count),
            minMs: times[0],
            medianMs: times[times.count / 2]
        )
    }

    private func makeInputBuffer() throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, Int(inputSize.width), Int(inputSize.height), inputPixelFormat, attrs, &buffer
        )
        guard status == kCVReturnSuccess, let buffer else { throw DetectorError.pixelBufferCreationFailed }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, 114, CVPixelBufferGetDataSize(buffer))
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    // MARK: - Decoding

    private func decode(_ array: MLMultiArray, confidence: Float, iouThreshold: Float) -> [Detection] {
        var shape = array.shape.map(\.intValue)
        var strides = array.strides.map(\.intValue)
        while shape.count < 3 {
            shape.insert(1, at: 0)
            strides.insert(0, at: 0)
        }
        guard shape.count == 3 else { return [] }

        return array.withUnsafeBytes { raw -> [Detection] in
            let tensor = Tensor(raw: raw, type: array.dataType, s1: strides[1], s2: strides[2])
            if shape[2] == 6 || shape[2] == 7 {
                // NMS-free end-to-end head (YOLO26 / YOLOv10): [1, maxDet, x1 y1 x2 y2 score class]
                return decodeEndToEnd(tensor, count: shape[1], confidence: confidence)
            }
            // Raw head (YOLO11 / YOLOv8): [1, 4 + classes, anchors] with cx cy w h + class scores
            return decodeRaw(tensor, channels: shape[1], anchors: shape[2], confidence: confidence, iouThreshold: iouThreshold)
        }
    }

    private func decodeEndToEnd(_ t: Tensor, count: Int, confidence: Float) -> [Detection] {
        var boxes: [(rect: CGRect, score: Float, cls: Int)] = []
        for i in 0..<count {
            let score = t[i, 4]
            guard score >= confidence else { continue }
            let x1 = CGFloat(t[i, 0]), y1 = CGFloat(t[i, 1]), x2 = CGFloat(t[i, 2]), y2 = CGFloat(t[i, 3])
            boxes.append((CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1), score, Int(t[i, 5].rounded())))
        }
        return boxes.map { makeDetection(rect: $0.rect, score: $0.score, cls: $0.cls) }
    }

    private func decodeRaw(_ t: Tensor, channels: Int, anchors: Int, confidence: Float, iouThreshold: Float) -> [Detection] {
        let classCount = channels - 4
        guard classCount > 0 else { return [] }

        // Class-major scan keeps reads contiguous along the anchor axis.
        var bestScore = [Float](repeating: 0, count: anchors)
        var bestClass = [Int](repeating: 0, count: anchors)
        for c in 0..<classCount {
            for a in 0..<anchors {
                let s = t[4 + c, a]
                if s > bestScore[a] {
                    bestScore[a] = s
                    bestClass[a] = c
                }
            }
        }

        var candidates: [(rect: CGRect, score: Float, cls: Int)] = []
        for a in 0..<anchors where bestScore[a] >= confidence {
            let cx = CGFloat(t[0, a]), cy = CGFloat(t[1, a]), w = CGFloat(t[2, a]), h = CGFloat(t[3, a])
            candidates.append((CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h), bestScore[a], bestClass[a]))
        }
        candidates.sort { $0.score > $1.score }
        if candidates.count > 3000 { candidates.removeLast(candidates.count - 3000) }

        var kept: [(rect: CGRect, score: Float, cls: Int)] = []
        for candidate in candidates {
            let overlaps = kept.contains { $0.cls == candidate.cls && iou($0.rect, candidate.rect) > iouThreshold }
            if !overlaps {
                kept.append(candidate)
                if kept.count >= 300 { break }
            }
        }
        return kept.map { makeDetection(rect: $0.rect, score: $0.score, cls: $0.cls) }
    }

    private func makeDetection(rect: CGRect, score: Float, cls: Int) -> Detection {
        // Outputs are in model-input pixels; .scaleFill maps them linearly onto the full frame.
        let normalized = CGRect(
            x: rect.minX / inputSize.width,
            y: rect.minY / inputSize.height,
            width: rect.width / inputSize.width,
            height: rect.height / inputSize.height
        ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let label = classNames.indices.contains(cls) ? classNames[cls] : "class \(cls)"
        return Detection(classIndex: cls, label: label, confidence: score, rect: normalized.isNull ? .zero : normalized)
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let union = a.width * a.height + b.width * b.height - interArea
        return union > 0 ? Float(interArea / union) : 0
    }

    /// Parses Ultralytics' Python-dict metadata, e.g. `{0: 'person', 1: "kid's bike"}`.
    static func parseClassNames(_ text: String?) -> [String] {
        guard let text,
              let regex = try? NSRegularExpression(pattern: #"(\d+)\s*:\s*(?:'([^']*)'|"([^"]*)")"#)
        else { return cocoClassNames }
        var names: [Int: String] = [:]
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let idRange = Range(match.range(at: 1), in: text), let id = Int(text[idRange]) else { continue }
            let nameRange = Range(match.range(at: 2), in: text) ?? Range(match.range(at: 3), in: text)
            if let nameRange { names[id] = String(text[nameRange]) }
        }
        guard let maxId = names.keys.max() else { return cocoClassNames }
        return (0...maxId).map { names[$0] ?? "class \($0)" }
    }
}

/// Strided read access to the last two axes of a rank-3 MLMultiArray (batch 0).
private struct Tensor {
    let raw: UnsafeRawBufferPointer
    let type: MLMultiArrayDataType
    let s1: Int
    let s2: Int

    subscript(i: Int, j: Int) -> Float {
        let offset = i * s1 + j * s2
        switch type {
        case .float32: return raw.load(fromByteOffset: offset * 4, as: Float.self)
        case .float16: return Float(raw.load(fromByteOffset: offset * 2, as: Float16.self))
        case .double: return Float(raw.load(fromByteOffset: offset * 8, as: Double.self))
        case .int32: return Float(raw.load(fromByteOffset: offset * 4, as: Int32.self))
        default: return 0
        }
    }
}

let cocoClassNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
