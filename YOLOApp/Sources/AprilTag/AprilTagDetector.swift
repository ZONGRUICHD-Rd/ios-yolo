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
/// Right-handed axes in the camera frame: X along the top edge to the right,
/// Y along the left edge downward, Z perpendicular into the tag (away from the camera when facing it).
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

    func detect(pixelBuffer: CVPixelBuffer, intrinsics: CameraIntrinsics, tagSize: Double,
                orientation: TagPatternOrientation) -> [TagPose] {
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
        return results.prefix(Int(count)).compactMap { makePose($0, tagSize: tagSize, orientation: orientation) }
    }

    private func makePose(_ result: yolo_tag_result, tagSize: Double, orientation: TagPatternOrientation) -> TagPose? {
        let r = withUnsafeBytes(of: result.R) { Array($0.bindMemory(to: Double.self)) }
        let t = withUnsafeBytes(of: result.t) { Array($0.bindMemory(to: Double.self)) }
        let c = withUnsafeBytes(of: result.corners) { Array($0.bindMemory(to: Double.self)) }
        guard (r + t).allSatisfy(\.isFinite), t[2] > 0 else { return nil }

        // Columns of R are AprilTag's tag axes; flip x/y for a 180°-rotated print (z stays, so still right-handed).
        let sign = orientation.cornerSign
        let xAxis = SIMD3(r[0], r[3], r[6]) * -sign
        let yAxis = SIMD3(r[1], r[4], r[7]) * -sign
        let zAxis = SIMD3(r[2], r[5], r[8])
        let center = SIMD3(t[0], t[1], t[2])
        return TagPose(
            id: Int(result.id),
            corners: (0..<4).map { SIMD2(c[2 * $0], c[2 * $0 + 1]) },
            origin: center - (xAxis + yAxis) * (tagSize / 2),
            xAxis: xAxis,
            yAxis: yAxis,
            zAxis: zAxis,
            size: tagSize
        )
    }
}

/// What the overlay draws for one tracked tag; points are normalized to the frame (top-left origin).
struct TagOverlay: Identifiable {
    let id: Int
    let quad: [CGPoint]
    let origin: CGPoint
    /// Projected ends of the X, Y and Z axes; empty when any end falls behind the camera.
    let axisEnds: [CGPoint]
    /// Smoothed origin in the camera frame, meters.
    let position: SIMD3<Double>
    let trail: [CGPoint]
    /// Not seen this frame; shown from its last pose until `holdTime` runs out.
    let isStale: Bool
}

/// Tracks tags by ID across frames: smooths the reported position, keeps a short trail
/// and holds briefly-missed tags so the overlay doesn't flicker.
final class AprilTagTracker {
    private struct Track {
        var pose: TagPose
        var intrinsics: CameraIntrinsics
        var smoothedOrigin: SIMD3<Double>
        var lastSeen: CFTimeInterval
        var trail: [CGPoint]
    }

    var holdTime: CFTimeInterval = 0.3
    var trailLength = 30
    private var tracks: [Int: Track] = [:]

    func reset() {
        tracks.removeAll()
    }

    func update(with poses: [TagPose], intrinsics: CameraIntrinsics, imageSize: CGSize,
                at now: CFTimeInterval) -> [TagOverlay] {
        for pose in poses {
            var track = tracks[pose.id]
                ?? Track(pose: pose, intrinsics: intrinsics, smoothedOrigin: pose.origin, lastSeen: now, trail: [])
            if simd_distance(track.smoothedOrigin, pose.origin) > pose.size * 2 {
                // Large jump (e.g. a second tag with the same ID): restart instead of smearing.
                track.smoothedOrigin = pose.origin
                track.trail.removeAll()
            } else {
                track.smoothedOrigin += (pose.origin - track.smoothedOrigin) * 0.4
            }
            track.pose = pose
            track.intrinsics = intrinsics
            track.lastSeen = now
            if let point = Self.project(pose.origin, intrinsics, imageSize) {
                track.trail.append(point)
                if track.trail.count > trailLength { track.trail.removeFirst(track.trail.count - trailLength) }
            }
            tracks[pose.id] = track
        }
        tracks = tracks.filter { now - $0.value.lastSeen <= holdTime }

        return tracks.values
            .sorted { $0.pose.id < $1.pose.id }
            .map { overlay(for: $0, imageSize: imageSize, isStale: $0.lastSeen < now) }
    }

    private func overlay(for track: Track, imageSize: CGSize, isStale: Bool) -> TagOverlay {
        let pose = track.pose
        let project = { (point: SIMD3<Double>) in Self.project(point, track.intrinsics, imageSize) }
        let ends = [pose.xAxis, pose.yAxis, pose.zAxis].map { project(pose.origin + $0 * pose.size) }
        let width = Double(imageSize.width), height = Double(imageSize.height)
        return TagOverlay(
            id: pose.id,
            quad: pose.corners.map { CGPoint(x: $0.x / width, y: $0.y / height) },
            origin: project(pose.origin) ?? .zero,
            axisEnds: ends.contains { $0 == nil } ? [] : ends.compactMap { $0 },
            position: track.smoothedOrigin,
            trail: track.trail,
            isStale: isStale
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
