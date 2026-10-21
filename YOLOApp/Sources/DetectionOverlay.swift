import SwiftUI

/// Maps normalized frame coordinates onto a `.resizeAspectFill` preview of that frame.
struct AspectFillMapping {
    private let shown: CGSize
    private let origin: CGPoint

    init(imageSize: CGSize, viewSize: CGSize) {
        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        shown = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        origin = CGPoint(x: (viewSize.width - shown.width) / 2, y: (viewSize.height - shown.height) / 2)
    }

    func point(_ p: CGPoint) -> CGPoint {
        CGPoint(x: origin.x + p.x * shown.width, y: origin.y + p.y * shown.height)
    }

    func rect(_ r: CGRect) -> CGRect {
        CGRect(origin: point(r.origin), size: CGSize(width: r.width * shown.width, height: r.height * shown.height))
    }
}

/// Draws YOLO boxes and AprilTag poses over a `.resizeAspectFill` camera preview of the same frame.
struct DetectionOverlay: View {
    let detections: [Detection]
    let tags: [TagOverlay]
    let imageSize: CGSize

    var body: some View {
        Canvas { context, size in
            guard imageSize.width > 0, imageSize.height > 0 else { return }
            let mapping = AspectFillMapping(imageSize: imageSize, viewSize: size)

            for detection in detections {
                let rect = mapping.rect(detection.rect)
                let color = Color.forClass(detection.classIndex)
                context.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(color), lineWidth: 2.5)
                let label = "\(detection.label) \(Int(detection.confidence * 100))%"
                drawLabel(label, at: CGPoint(x: rect.minX, y: rect.minY), above: true, color: color, in: &context)
            }

            for tag in tags {
                drawTag(tag, mapping: mapping, in: &context)
            }
        }
        .allowsHitTesting(false)
    }

    private static let axisColors: [(name: String, color: Color)] = [("X", .red), ("Y", .green), ("Z", .blue)]

    private func drawTag(_ tag: TagOverlay, mapping: AspectFillMapping, in context: inout GraphicsContext) {
        var layer = context
        layer.opacity = tag.isStale ? 0.35 : 1

        if tag.trail.count > 1 {
            var trail = Path()
            trail.addLines(tag.trail.map(mapping.point))
            layer.stroke(trail, with: .color(.yellow.opacity(0.7)), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }

        let quad = tag.quad.map(mapping.point)
        var outline = Path()
        outline.addLines(quad)
        outline.closeSubpath()
        layer.fill(outline, with: .color(.cyan.opacity(0.15)))
        layer.stroke(outline, with: .color(.cyan), lineWidth: 3)

        let origin = mapping.point(tag.origin)
        for (end, axis) in zip(tag.axisEnds.map(mapping.point), Self.axisColors) {
            var line = Path()
            line.move(to: origin)
            line.addLine(to: end)
            layer.stroke(line, with: .color(axis.color), style: StrokeStyle(lineWidth: 4, lineCap: .round))
            let text = layer.resolve(Text(axis.name).font(.system(size: 14, weight: .heavy)).foregroundColor(axis.color))
            layer.draw(text, at: end, anchor: .center)
        }
        layer.fill(Path(ellipseIn: CGRect(x: origin.x - 5, y: origin.y - 5, width: 10, height: 10)), with: .color(.white))

        let p = tag.position * 100
        let distance = (tag.position * tag.position).sum().squareRoot() * 100
        let label = String(format: "ID %ld · %.1f cm\n(%.1f, %.1f, %.1f)", tag.id, distance, p.x, p.y, p.z)
        let center = CGPoint(
            x: quad.map(\.x).reduce(0, +) / CGFloat(max(quad.count, 1)),
            y: quad.map(\.y).reduce(0, +) / CGFloat(max(quad.count, 1))
        )
        drawLabel(label, at: center, above: false, color: .cyan, in: &layer)
    }

    /// Draws text on a colored tag; `above` places it above `point` (box captions), otherwise centered on it.
    private func drawLabel(_ string: String, at point: CGPoint, above: Bool, color: Color, in context: inout GraphicsContext) {
        let text = context.resolve(
            Text(string)
                .font(.system(size: 13, weight: .semibold).monospacedDigit())
                .foregroundColor(.black)
        )
        let textSize = text.measure(in: CGSize(width: 400, height: 80))
        let background = above
            ? CGRect(x: point.x, y: max(point.y - textSize.height - 4, 0), width: textSize.width + 8, height: textSize.height + 4)
            : CGRect(x: point.x - textSize.width / 2 - 4, y: point.y - textSize.height / 2 - 2,
                     width: textSize.width + 8, height: textSize.height + 4)
        context.fill(Path(roundedRect: background, cornerRadius: 4), with: .color(color))
        context.draw(text, at: CGPoint(x: background.minX + 4, y: background.minY + 2), anchor: .topLeading)
    }
}

extension Color {
    static func forClass(_ index: Int) -> Color {
        Color(hue: Double((index * 47) % 360) / 360, saturation: 0.8, brightness: 0.95)
    }
}
