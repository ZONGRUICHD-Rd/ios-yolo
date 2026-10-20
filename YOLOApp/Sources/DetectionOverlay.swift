import SwiftUI

/// Draws detections over a `.resizeAspectFill` camera preview of the same frame.
struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize

    var body: some View {
        Canvas { context, size in
            guard imageSize.width > 0, imageSize.height > 0 else { return }
            let scale = max(size.width / imageSize.width, size.height / imageSize.height)
            let shown = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            let origin = CGPoint(x: (size.width - shown.width) / 2, y: (size.height - shown.height) / 2)

            for detection in detections {
                let rect = CGRect(
                    x: origin.x + detection.rect.minX * shown.width,
                    y: origin.y + detection.rect.minY * shown.height,
                    width: detection.rect.width * shown.width,
                    height: detection.rect.height * shown.height
                )
                let color = Color.forClass(detection.classIndex)
                context.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(color), lineWidth: 2.5)

                let label = context.resolve(
                    Text("\(detection.label) \(Int(detection.confidence * 100))%")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                )
                let textSize = label.measure(in: CGSize(width: 400, height: 40))
                let tag = CGRect(
                    x: rect.minX,
                    y: max(rect.minY - textSize.height - 4, 0),
                    width: textSize.width + 8,
                    height: textSize.height + 4
                )
                context.fill(Path(roundedRect: tag, cornerRadius: 4), with: .color(color))
                context.draw(label, at: CGPoint(x: tag.minX + 4, y: tag.minY + 2), anchor: .topLeading)
            }
        }
        .allowsHitTesting(false)
    }
}

extension Color {
    static func forClass(_ index: Int) -> Color {
        Color(hue: Double((index * 47) % 360) / 360, saturation: 0.8, brightness: 0.95)
    }
}
