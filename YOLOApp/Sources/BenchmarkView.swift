import SwiftUI

struct BenchmarkView: View {
    @ObservedObject var viewModel: DetectorViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("依次在 NPU / GPU / CPU 上加载 \(viewModel.selectedModel?.name ?? "-")，"
                        + "预热 5 次后测量 50 次 MLModel.prediction（不含相机与 Vision 缩放）。"
                        + "设备占比来自 MLComputePlan（iOS 17.4+），按算子估算开销加权。")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section("结果") {
                    if viewModel.benchmarkResults.isEmpty && !viewModel.isBenchmarking {
                        Text("尚未运行").foregroundColor(.secondary)
                    }
                    ForEach(viewModel.benchmarkResults) { result in
                        row(result)
                    }
                    if viewModel.isBenchmarking {
                        HStack {
                            ProgressView()
                            Text("测试中…（相机检测已暂停）").foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("计算单元跑分")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("开始") { viewModel.runBenchmark() }
                        .disabled(viewModel.isBenchmarking)
                }
            }
        }
    }

    private func row(_ result: BenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(result.mode.title).font(.headline)
                Text(result.mode.detail).font(.caption).foregroundColor(.secondary)
            }
            if let error = result.error {
                Text(error).font(.caption).foregroundColor(.red)
            }
            if let stats = result.stats {
                HStack(spacing: 14) {
                    value("平均", String(format: "%.2f ms", stats.meanMs))
                    value("中位", String(format: "%.2f ms", stats.medianMs))
                    value("最快", String(format: "%.2f ms", stats.minMs))
                }
                HStack(spacing: 14) {
                    value("吞吐", String(format: "%.0f FPS", 1000 / stats.meanMs))
                    if let load = result.loadMs {
                        value("加载", String(format: "%.0f ms", load))
                    }
                }
            }
            if let breakdown = result.breakdown {
                Text("调度：\(breakdown.summary)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func value(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundColor(.secondary)
            Text(text).font(.subheadline.monospacedDigit())
        }
    }
}
