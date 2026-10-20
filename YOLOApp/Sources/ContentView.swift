import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = DetectorViewModel()
    @State private var showBenchmark = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: viewModel.camera.session)
                .ignoresSafeArea()
            DetectionOverlay(detections: viewModel.detections, imageSize: viewModel.imageSize)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                statsHUD
                Spacer()
                controlPanel
            }
            .padding(.horizontal, 12)

            if viewModel.isLoading {
                ProgressView("正在加载模型…\n首次在 NPU 上加载需要编译，可能需要数秒")
                    .multilineTextAlignment(.center)
                    .padding(20)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
        }
        .onAppear { viewModel.start() }
        .sheet(isPresented: $showBenchmark) {
            BenchmarkView(viewModel: viewModel)
        }
    }

    private var statsHUD: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(viewModel.selectedModel?.name ?? "无模型")
                    .font(.headline)
                Text(viewModel.computeMode.title)
                    .font(.caption.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor, in: Capsule())
                Spacer()
                Text(String(format: "%.1f FPS", viewModel.stats.fps))
                    .font(.headline.monospacedDigit())
            }
            HStack(spacing: 12) {
                metric("推理", String(format: "%.1f ms", viewModel.stats.inferenceMs))
                metric("后处理", String(format: "%.2f ms", viewModel.stats.postprocessMs))
                metric("目标", "\(viewModel.stats.objectCount)")
            }
            if let error = viewModel.errorMessage {
                Text(error).font(.caption).foregroundColor(.red)
            } else if let status = viewModel.statusMessage {
                Text(status).font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.top, 8)
    }

    private func metric(_ title: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(title).foregroundColor(.secondary)
            Text(value).monospacedDigit()
        }
        .font(.caption)
    }

    private var controlPanel: some View {
        VStack(spacing: 12) {
            Picker("计算单元", selection: $viewModel.computeMode) {
                ForEach(ComputeMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(viewModel.computeMode.detail)
                .font(.caption2)
                .foregroundColor(.secondary)

            HStack {
                Text("置信度").font(.caption)
                Slider(value: $viewModel.confidence, in: 0.05...0.95, step: 0.05)
                Text(String(format: "%.2f", viewModel.confidence))
                    .font(.caption.monospacedDigit())
                    .frame(width: 36)
            }

            HStack {
                Menu {
                    ForEach(viewModel.models) { model in
                        Button(model.name) { viewModel.selectedModel = model }
                    }
                } label: {
                    Label(viewModel.selectedModel?.name ?? "选择模型", systemImage: "cube.box")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button {
                    showBenchmark = true
                } label: {
                    Label("跑分对比", systemImage: "speedometer")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .disabled(viewModel.isLoading || viewModel.isBenchmarking)
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.bottom, 8)
    }
}
