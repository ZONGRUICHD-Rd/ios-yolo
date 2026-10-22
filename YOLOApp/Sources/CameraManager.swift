import AVFoundation
import CoreVideo
import simd

/// Pinhole intrinsics in pixels of the delivered (portrait) frame.
struct CameraIntrinsics {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    /// True when taken from AVFoundation's per-frame matrix, false when estimated from the field of view.
    var isFromCamera: Bool
}

struct CameraFrame {
    let pixelBuffer: CVPixelBuffer
    let intrinsics: CameraIntrinsics
    /// Presentation time, seconds.
    let timestamp: Double
}

/// Back-camera capture delivering portrait 420f (full-range NV12) frames on `videoQueue`.
final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "yolo.camera.video", qos: .userInitiated)
    /// Called on `videoQueue` for every frame that isn't dropped.
    var onFrame: ((CameraFrame) -> Void)?

    private let sessionQueue = DispatchQueue(label: "yolo.camera.session")
    private let output = AVCaptureVideoDataOutput()
    private var isConfigured = false
    /// Horizontal field of view of the active format in degrees (along the sensor's long side).
    private var fieldOfView: Float = 70

    /// Requests permission, configures the session and starts it. `completion` runs on the main queue.
    func start(completion: @escaping (String?) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                DispatchQueue.main.async { completion("未获得相机权限，请在 设置 > 隐私 > 相机 中开启") }
                return
            }
            self.sessionQueue.async {
                if !self.isConfigured {
                    if let error = self.configure() {
                        DispatchQueue.main.async { completion(error) }
                        return
                    }
                    self.isConfigured = true
                }
                if !self.session.isRunning { self.session.startRunning() }
                DispatchQueue.main.async { completion(nil) }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configure() -> String? {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return "找不到后置摄像头"
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1920x1080
        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else { return "无法添加相机输入" }
            session.addInput(input)
        } catch {
            return "相机打开失败：\(error.localizedDescription)"
        }

        // Luma plane feeds AprilTag directly; Vision converts for YOLO.
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return "无法添加视频输出" }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90 // portrait buffers
            }
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        enableHighFrameRate(device, fps: 60)
        fieldOfView = device.activeFormat.videoFieldOfView
        return nil
    }

    /// Picks a 1080p format that supports `fps` so the camera doesn't cap the measured FPS at 30.
    private func enableHighFrameRate(_ device: AVCaptureDevice, fps: Double) {
        let format = device.formats.first { format in
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dims.width == 1920 && dims.height == 1080
                && format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= fps }
        }
        guard let format, (try? device.lockForConfiguration()) != nil else { return }
        device.activeFormat = format
        let duration = CMTime(value: 1, timescale: CMTimeScale(fps))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.unlockForConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let width = Double(CVPixelBufferGetWidth(pixelBuffer))
        let height = Double(CVPixelBufferGetHeight(pixelBuffer))
        let matrix = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data
        let intrinsics = matrix.flatMap { parseIntrinsics($0, width: width, height: height) }
            ?? estimatedIntrinsics(width: width, height: height)
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(CameraFrame(pixelBuffer: pixelBuffer, intrinsics: intrinsics, timestamp: timestamp))
    }

    private func parseIntrinsics(_ data: Data, width: Double, height: Double) -> CameraIntrinsics? {
        guard data.count >= MemoryLayout<matrix_float3x3>.size else { return nil }
        let m = data.withUnsafeBytes { $0.loadUnaligned(as: matrix_float3x3.self) }
        var fx = Double(m.columns.0.x), fy = Double(m.columns.1.y)
        var cx = Double(m.columns.2.x), cy = Double(m.columns.2.y)
        // The matrix may describe the unrotated landscape sensor; rotate it 90° clockwise like the frame.
        if (cx > cy) != (width > height) {
            (fx, fy, cx, cy) = (fy, fx, width - cy, cx)
        }
        let plausible = fx > 0 && fy > 0 && abs(cx - width / 2) < width * 0.2 && abs(cy - height / 2) < height * 0.2
        return plausible ? CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy, isFromCamera: true) : nil
    }

    private func estimatedIntrinsics(width: Double, height: Double) -> CameraIntrinsics {
        let focal = max(width, height) / 2 / tan(Double(fieldOfView) * .pi / 360)
        return CameraIntrinsics(fx: focal, fy: focal, cx: width / 2, cy: height / 2, isFromCamera: false)
    }
}
