import AVFoundation
import CoreVideo

/// Back-camera capture delivering portrait BGRA frames on `videoQueue`.
final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "yolo.camera.video", qos: .userInitiated)
    /// Called on `videoQueue` for every frame that isn't dropped.
    var onFrame: ((CVPixelBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "yolo.camera.session")
    private let output = AVCaptureVideoDataOutput()
    private var isConfigured = false

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

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return "无法添加视频输出" }
        session.addOutput(output)

        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90 // portrait buffers
        }
        enableHighFrameRate(device, fps: 60)
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
        onFrame?(pixelBuffer)
    }
}
