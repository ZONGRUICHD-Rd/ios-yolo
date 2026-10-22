import CoreGraphics
import CoreVideo
import Foundation
import simd

/// How the printed pattern is oriented relative to AprilTag's reference images.
/// It decides which physical corner is the "top-left" origin.
enum TagPatternOrientation: String, CaseIterable, Identifiable {
    /// Rotated 180° from the reference images, like the bundled tag36h11 A4 sheet (ID label below the tag).
    case rotated180
    /// Upright AprilTag reference images (tag36_11_00000.png …).
    case official

    var id: String { rawValue }

    var title: String {
        switch self {
        case .rotated180: return "打印图"
        case .official: return "官方图"
        }
    }

    /// Sign of the printed top-left corner in AprilTag's tag frame (x right, y down, origin at center).
    fileprivate var cornerSign: Double {
        switch self {
        case .rotated180: return 1
        case .official: return -1
        }
    }
}

/// Pose of one tag with its origin at the tag's top-left corner (outer black square).
/// Right-handed axes in the camera frame: X along the top edge to the right, Y up (along the left
/// edge, away from the tag), Z perpendicular out of the tag toward the viewer — the ArUco/OpenCV
/// marker convention, moved to the corner.
struct TagPose {
    let id: Int
    /// Detected corners in frame pixels, in AprilTag's ring order.
    let corners: [SIMD2<Double>]
    /// Meters, camera frame (x right, y down, z forward).
    let origin: SIMD3<Double>
    let xAxis: SIMD3<Double>
    let yAxis: SIMD3<Double>
    let zAxis: SIMD3<Double>
    let size: Double
    /// Seconds since last detected; 0 when detected in this frame.
    let age: Double

    /// `track` holds AprilTag's tag-center pose (x right, y down, z into the tag of the reference image).
    init(track: yolo_tag_track, tagSize: Double, orientation: TagPatternOrientation) {
        let r = withUnsafeBytes(of: track.R) { Array($0.bindMemory(to: Double.self)) }
        let t = withUnsafeBytes(of: track.t) { Array($0.bindMemory(to: Double.self)) }
        let c = withUnsafeBytes(of: track.corners) { Array($0.bindMemory(to: Double.self)) }
        let column0 = SIMD3(r[0], r[3], r[6]), column1 = SIMD3(r[1], r[4], r[7]), column2 = SIMD3(r[2], r[5], r[8])
        // A 180°-rotated print swaps which corner is top-left and flips the in-plane axes.
        let sign = orientation.cornerSign
        id = Int(track.id)
        corners = (0..<4).map { SIMD2(c[2 * $0], c[2 * $0 + 1]) }
        origin = SIMD3(t[0], t[1], t[2]) + (column0 + column1) * (sign * tagSize / 2)
        xAxis = column0 * -sign
        yAxis = column1 * sign
        zAxis = -column2
        size = tagSize
        age = track.age
    }
}

/// tag36h11 detector over the luma plane of 420f frames. Not thread-safe: use from one queue.
final class AprilTagDetector {
    private let handle: OpaquePointer
    private var results = [yolo_tag_result](repeating: yolo_tag_result(), count: 64)

    init() {
        let threads = max(2, min(ProcessInfo.processInfo.activeProcessorCount, 6))
        handle = yolo_tag_detector_create(Int32(threads), 2.0)
    }

    deinit {
        yolo_tag_detector_destroy(handle)
    }

    /// Detections with both pose candidates each; see `AprilTagTracker` for choosing between them.
    func detect(pixelBuffer: CVPixelBuffer, intrinsics: CameraIntrinsics, tagSize: Double) -> [yolo_tag_result] {
        guard CVPixelBufferIsPlanar(pixelBuffer) else { return [] }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let luma = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return [] }

        let count = yolo_tag_detect(
            handle,
            luma.assumingMemoryBound(to: UInt8.self),
            Int32(CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)),
            Int32(CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)),
            Int32(CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)),
            tagSize, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            &results, Int32(results.count)
        )
        return Array(results.prefix(Int(count)))
    }
}

/// What the overlay draws for one tracked tag; points are normalized to the frame (top-left origin).
struct TagOverlay: Identifiable {
    let id: Int
    let quad: [CGPoint]
    let origin: CGPoint
    /// Projected ends of the X, Y and Z axes; empty when any end falls behind the camera.
    let axisEnds: [CGPoint]
    /// Origin in the camera frame, meters.
    let position: SIMD3<Double>
    let trail: [CGPoint]
    /// Not seen this frame; shown from its last pose until the hold time runs out.
    let isStale: Bool
}

/// Stabilized tracking (see AprilTagTracker.h): resolves the planar pose ambiguity using the board
/// plane and the track's history, smooths poses with a One Euro filter, and keeps a short trail.
final class AprilTagTracker {
    /// 0 = most responsive, 1 = steadiest.
    var smoothing = 0.5 {
        didSet { applySmoothing() }
    }
    var trailLength = 30

    private let handle: OpaquePointer
    private var tracks = [yolo_tag_track](repeating: yolo_tag_track(), count: 64)
    private var trails: [Int: [CGPoint]] = [:]

    init() {
        handle = yolo_tag_tracker_create()
        applySmoothing()
    }

    deinit {
        yolo_tag_tracker_destroy(handle)
    }

    func reset() {
        yolo_tag_tracker_reset(handle)
        trails.removeAll()
    }

    func update(with detections: [yolo_tag_result], timestamp: Double, intrinsics: CameraIntrinsics,
                imageSize: CGSize, tagSize: Double, orientation: TagPatternOrientation) -> [TagOverlay] {
        let count = yolo_tag_tracker_update(
            handle, detections, Int32(detections.count), timestamp, &tracks, Int32(tracks.count)
        )
        let poses = tracks.prefix(Int(count)).map { TagPose(track: $0, tagSize: tagSize, orientation: orientation) }

        trails = trails.filter { id, _ in poses.contains { $0.id == id } }
        return poses.map { pose in
            var trail = trails[pose.id, default: []]
            if pose.age == 0, let point = Self.project(pose.origin, intrinsics, imageSize) {
                trail.append(point)
                if trail.count > trailLength { trail.removeFirst(trail.count - trailLength) }
                trails[pose.id] = trail
            }
            return overlay(for: pose, trail: trail, intrinsics: intrinsics, imageSize: imageSize)
        }
    }

    private func applySmoothing() {
        // Minimum cutoff from 4 Hz (responsive) down to 0.16 Hz (steady); speed-dependent terms stay fixed.
        let minCutoff = 4.0 * pow(0.04, min(max(smoothing, 0), 1))
        yolo_tag_tracker_set_filter(handle, minCutoff, 8.0, 1.5)
    }

    private func overlay(for pose: TagPose, trail: [CGPoint], intrinsics: CameraIntrinsics, imageSize: CGSize) -> TagOverlay {
        let project = { (point: SIMD3<Double>) in Self.project(point, intrinsics, imageSize) }
        let ends = [pose.xAxis, pose.yAxis, pose.zAxis].map { project(pose.origin + $0 * pose.size) }
        let width = Double(imageSize.width), height = Double(imageSize.height)
        return TagOverlay(
            id: pose.id,
            quad: pose.corners.map { CGPoint(x: $0.x / width, y: $0.y / height) },
            origin: project(pose.origin) ?? .zero,
            axisEnds: ends.contains { $0 == nil } ? [] : ends.compactMap { $0 },
            position: pose.origin,
            trail: trail,
            isStale: pose.age > 0
        )
    }

    /// Pinhole projection into normalized frame coordinates.
    static func project(_ point: SIMD3<Double>, _ k: CameraIntrinsics, _ imageSize: CGSize) -> CGPoint? {
        guard point.z > 1e-4 else { return nil }
        let u = k.fx * point.x / point.z + k.cx
        let v = k.fy * point.y / point.z + k.cy
        return CGPoint(x: u / Double(imageSize.width), y: v / Double(imageSize.height))
    }
}
