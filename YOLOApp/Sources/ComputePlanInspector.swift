import CoreML
import Foundation

/// Share of a model's work that Core ML plans to run on each device for a given compute mode.
struct DeviceBreakdown {
    /// Fractions in 0...1, keyed by "NPU" / "GPU" / "CPU".
    let costShare: [String: Double]
    let opCounts: [String: Int]

    var summary: String {
        let order = ["NPU", "GPU", "CPU"]
        let parts = order.compactMap { key -> String? in
            guard let ops = opCounts[key], ops > 0 else { return nil }
            let share = costShare[key].map { String(format: "%.0f%%", $0 * 100) } ?? "-"
            return "\(key) \(share) (\(ops) ops)"
        }
        return parts.isEmpty ? "无数据" : parts.joined(separator: " · ")
    }
}

enum ComputePlanInspector {
    /// Uses MLComputePlan (iOS 17.4+) to see which device each ML Program op is scheduled on.
    static func breakdown(modelURL: URL, mode: ComputeMode) async -> DeviceBreakdown? {
        guard #available(iOS 17.4, *) else { return nil }
        let config = MLModelConfiguration()
        config.computeUnits = mode.computeUnits
        guard let plan = try? await MLComputePlan.load(contentsOf: modelURL, configuration: config),
              case let .program(program) = plan.modelStructure,
              let main = program.functions["main"]
        else { return nil }

        var cost: [String: Double] = [:]
        var counts: [String: Int] = [:]
        func visit(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                if let usage = plan.deviceUsage(for: op) {
                    let key = deviceName(usage.preferred)
                    counts[key, default: 0] += 1
                    cost[key, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
                }
                op.blocks.forEach(visit)
            }
        }
        visit(main.block)

        let total = cost.values.reduce(0, +)
        let share = total > 0 ? cost.mapValues { $0 / total } : [:]
        return DeviceBreakdown(costShare: share, opCounts: counts)
    }

    @available(iOS 17.0, *)
    private static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .neuralEngine: return "NPU"
        case .gpu: return "GPU"
        case .cpu: return "CPU"
        @unknown default: return "?"
        }
    }
}
