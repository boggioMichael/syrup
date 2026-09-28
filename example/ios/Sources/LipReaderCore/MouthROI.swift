// MouthROI.swift — where the mouth is inside a face, and the crop a lip
// reader wants (mouth.py), plus the few pixel-level measurements the
// pipeline makes itself on the BGRA frames.
//
// With Vision landmarks the geometry is exact: the outer lips give the
// corners, the inner lips the lip line, the eyes the ruler. Without
// landmarks the mouth is placed from the face box alone, good enough to
// track speaking activity but less good for reading.
import CoreGraphics
import CoreVideo
import Foundation

public enum MouthCrop {
    public static let width = 100
    public static let height = 50
    static let horizontalPad = 0.19
    /// LipNet's preprocessing pads the mouth by multiplying the corners'
    /// absolute x coordinates by (1 -/+ 0.19), so the crop scale it was
    /// trained with depends on where the mouth sat in the 360x288 GRID frame:
    /// about x = 180. The same scale is reproduced here for a mouth anywhere in
    /// any frame by padding as if it were at that position; otherwise a face on
    /// the right of a wide frame gets a crop the model has never seen.
    static let gridMouthX = 180.0
}

public enum FrameError: Error { case notBGRA, noBaseAddress }

// MARK: - Pixel access

/// Read access to a locked 32BGRA pixel buffer. Outside the frame is black,
/// which is also what mouth.py pads a crop with.
public struct BGRAFrame {
    public let width: Int
    public let height: Int
    let bytesPerRow: Int
    let base: UnsafePointer<UInt8>

    /// Locks `buffer` for the duration of `body`.
    public static func withLocked<T>(_ buffer: CVPixelBuffer, _ body: (BGRAFrame) throws -> T) throws -> T {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { throw FrameError.notBGRA }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw FrameError.noBaseAddress }
        let frame = BGRAFrame(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer),
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              base: UnsafePointer(address.assumingMemoryBound(to: UInt8.self)))
        return try body(frame)
    }

    @inline(__always) func rgb(_ x: Int, _ y: Int) -> (r: Float, g: Float, b: Float) {
        guard x >= 0, y >= 0, x < width, y < height else { return (0, 0, 0) }
        let p = base + y * bytesPerRow + x * 4
        return (Float(p[2]), Float(p[1]), Float(p[0]))
    }

    /// OpenCV's RGB2GRAY weights, 0...255.
    @inline(__always) func gray(_ x: Int, _ y: Int) -> Float {
        let p = rgb(x, y)
        return 0.299 * p.r + 0.587 * p.g + 0.114 * p.b
    }

    /// Box-filtered resample of `rect` (source pixels) to outW x outH:
    /// row-major RGB bytes and gray values 0...255. Up to 4x4 samples per
    /// output pixel, so a mouth or a face costs the same whatever the frame size.
    func resample(_ rect: CGRect, width outW: Int, height outH: Int) -> (rgb: [UInt8], gray: [Float]) {
        var rgb = [UInt8](repeating: 0, count: outW * outH * 3)
        var gray = [Float](repeating: 0, count: outW * outH)
        let sx = rect.width / Double(outW), sy = rect.height / Double(outH)
        let nx = max(1, min(4, Int(sx.rounded(.up)))), ny = max(1, min(4, Int(sy.rounded(.up))))
        let inv = 1 / Float(nx * ny)
        for oy in 0..<outH {
            for ox in 0..<outW {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for j in 0..<ny {
                    let y = Int((rect.minY + (Double(oy) + (Double(j) + 0.5) / Double(ny)) * sy).rounded(.down))
                    for i in 0..<nx {
                        let x = Int((rect.minX + (Double(ox) + (Double(i) + 0.5) / Double(nx)) * sx).rounded(.down))
                        let p = self.rgb(x, y)
                        r += p.r; g += p.g; b += p.b
                    }
                }
                r *= inv; g *= inv; b *= inv
                let o = oy * outW + ox
                rgb[o * 3] = UInt8(min(255, r.rounded())); rgb[o * 3 + 1] = UInt8(min(255, g.rounded())); rgb[o * 3 + 2] = UInt8(min(255, b.rounded()))
                gray[o] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }
        return (rgb, gray)
    }

    /// 8x8x8 colour histogram over a 160x90 grid of samples, normalised to
    /// sum 1 (cuts.py builds the same from a 160x90 resize).
    func coarseHistogram() -> [Float] {
        var hist = [Float](repeating: 0, count: 512)
        for gy in 0..<90 {
            let y = (gy * height) / 90
            for gx in 0..<160 {
                let x = (gx * width) / 160
                let p = base + y * bytesPerRow + x * 4
                hist[(Int(p[2]) >> 5) * 64 + (Int(p[1]) >> 5) * 8 + (Int(p[0]) >> 5)] += 1
            }
        }
        let n = Float(160 * 90)
        return hist.map { $0 / n }
    }

    /// Share of clearly-dark pixels in `rect` relative to `normaliser`
    /// (mouth.py measure_openness): skin is the 80th percentile of the region,
    /// dark is below darkest + factor * (skin - darkest). Sampled on a grid of
    /// at most 64x64 points; each sample stands for its stride's pixels.
    func darkShare(in rect: CGRect, factor: Float, normaliser: Float) -> Float {
        let x0 = max(Int(rect.minX), 0), y0 = max(Int(rect.minY), 0)
        let x1 = min(Int(rect.maxX), width), y1 = min(Int(rect.maxY), height)
        guard y1 - y0 >= 2, x1 - x0 >= 2 else { return 0 }
        let stepX = max(1, (x1 - x0) / 64), stepY = max(1, (y1 - y0) / 64)
        var values: [Float] = []
        values.reserveCapacity(4096)
        var y = y0
        while y < y1 {
            var x = x0
            while x < x1 { values.append(gray(x, y)); x += stepX }
            y += stepY
        }
        let sorted = values.sorted()
        let darkest = sorted[0]
        let skin = sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * 0.8).rounded()))]
        let threshold = darkest + factor * (skin - darkest)
        let dark = values.reduce(0) { $0 + ($1 <= threshold ? 1 : 0) }
        return min(1, Float(dark * stepX * stepY) / max(1, normaliser))
    }
}

extension Box {
    public var cgRect: CGRect { CGRect(x: x, y: y, width: w, height: h) }
    public init(_ rect: CGRect) { self.init(x: rect.minX, y: rect.minY, w: rect.width, h: rect.height) }
}

// MARK: - Landmarks and estimates

/// Vision's lip and eye landmarks, in frame pixels with y down (FaceTracker
/// does the flip).
public struct MouthLandmarks: Sendable {
    public var outerLips: [CGPoint]
    public var innerLips: [CGPoint]
    public var leftEye: CGPoint?
    public var rightEye: CGPoint?

    public init(outerLips: [CGPoint], innerLips: [CGPoint], leftEye: CGPoint? = nil, rightEye: CGPoint? = nil) {
        self.outerLips = outerLips; self.innerLips = innerLips; self.leftEye = leftEye; self.rightEye = rightEye
    }

    public var mouthWidth: Double {
        let xs = outerLips.map(\.x)
        return (xs.max() ?? 0) - (xs.min() ?? 0)
    }

    /// The eye distance, mouth.py's ruler; the corners sit at ±0.47 of it,
    /// which is what says the ruler when the eyes are missing.
    public var scale: Double {
        if let l = leftEye, let r = rightEye { return hypot(r.x - l.x, r.y - l.y) }
        return mouthWidth / 0.94
    }
}

public struct MouthEstimate: Sendable {
    public var centre: CGPoint
    public var left: Double
    public var right: Double
    public var fromLandmarks: Bool
    /// 0...1: dark pixels in the band around the lip line, a proxy for how open the mouth is.
    public var openness: Double
}

public enum MouthLocator {
    /// The lip line and corners from landmarks; nil when they are degenerate.
    static func fromLandmarks(_ frame: BGRAFrame, _ lm: MouthLandmarks) -> MouthEstimate? {
        guard lm.outerLips.count >= 2 else { return nil }
        let xs = lm.outerLips.map(\.x)
        let left = xs.min()!, right = xs.max()!
        let d = lm.scale
        guard d >= 8, right - left >= 4 else { return nil }
        let line = lm.innerLips.isEmpty ? lm.outerLips : lm.innerLips
        let cy = line.map(\.y).reduce(0, +) / Double(line.count)
        let centre = CGPoint(x: (left + right) / 2, y: cy)
        return MouthEstimate(centre: centre, left: left, right: right, fromLandmarks: true,
                             openness: measureOpenness(frame, centre: centre, left: left, right: right, scale: d))
    }

    /// The share of clearly-dark pixels in a band of ±0.25 d around the lip
    /// line between the corners, relative to the mouth width: a closed mouth
    /// is a thin line, an open one a cavity (mouth.py measure_openness).
    static func measureOpenness(_ frame: BGRAFrame, centre: CGPoint, left: Double, right: Double, scale d: Double) -> Double {
        let band = CGRect(x: left, y: centre.y - 0.25 * d, width: right - left, height: 0.5 * d)
        return Double(frame.darkShare(in: band, factor: 0.35, normaliser: Float((right - left) * 0.5 * d)))
    }

    /// A mouth placed from the face box alone: roughly where a frontal
    /// detector's box puts it (mouth.py locate_from_box).
    static func fromBox(_ frame: BGRAFrame, _ face: Box) -> MouthEstimate {
        let cx = face.x + face.w / 2, cy = face.y + face.h * 0.78, half = face.w * 0.22
        let region = CGRect(x: cx - half, y: cy - face.h * 0.1, width: 2 * half, height: 0.2 * face.h)
        let clipped = region.intersection(CGRect(x: 0, y: 0, width: frame.width, height: frame.height))
        let openness = clipped.isNull ? 0 : frame.darkShare(in: clipped, factor: 0.35, normaliser: Float(clipped.width * clipped.height * 0.5))
        return MouthEstimate(centre: CGPoint(x: cx, y: cy), left: cx - half, right: cx + half, fromLandmarks: false, openness: Double(openness))
    }
}

// MARK: - Per-person tracker and crop

/// The LipNet crop of one frame: 100x50, width-major ([x][y][rgb]) as the
/// original code stores it, bytes 0...255 (divide by 255 for the model), and
/// its gray version in 0...1 for the motion signal.
public struct MouthCropFrame: Sendable {
    public let rgb: [UInt8]
    public let gray: [Float]
}

/// Per person: smoothed mouth position and a crop scale fixed for the track.
public final class MouthTracker {
    let smoothing: Double
    public private(set) var centre: CGPoint?
    public private(set) var left: Double?
    public private(set) var right: Double?
    public private(set) var ratio: Double?
    private var fromLandmarks = false
    private var ratioFromLandmarks = false
    private var scale: Double?
    private var lastFace: Box?

    public init(smoothing: Double = 0.4) { self.smoothing = smoothing }

    public func update(_ frame: BGRAFrame, face: Box, landmarks: MouthLandmarks?) -> MouthEstimate {
        var estimate = landmarks.flatMap { MouthLocator.fromLandmarks(frame, $0) }
        if estimate != nil, let lm = landmarks {
            scale = lm.scale
        } else if fromLandmarks, let c = centre, let l = left, let r = right, let d = scale {
            // No landmarks this frame: keep the lip-line geometry, moved with
            // the face box, and measure openness the same way as before so the
            // signal stays comparable from frame to frame.
            var dx = 0.0, dy = 0.0
            if let last = lastFace {
                dx = face.centre.x - last.centre.x
                dy = face.centre.y - last.centre.y
            }
            let moved = CGPoint(x: c.x + dx, y: c.y + dy)
            estimate = MouthEstimate(centre: moved, left: l + dx, right: r + dx, fromLandmarks: false,
                                     openness: MouthLocator.measureOpenness(frame, centre: moved, left: l + dx, right: r + dx, scale: d))
        }
        let e = estimate ?? MouthLocator.fromBox(frame, face)
        lastFace = face
        let a = smoothing
        if let c = centre, let l = left, let r = right {
            centre = CGPoint(x: c.x + (e.centre.x - c.x) * a, y: c.y + (e.centre.y - c.y) * a)
            left = l + (e.left - l) * a
            right = r + (e.right - r) * a
        } else {
            centre = e.centre; left = e.left; right = e.right
        }
        fromLandmarks = fromLandmarks || e.fromLandmarks
        if ratio == nil || (e.fromLandmarks && !ratioFromLandmarks) {
            let width = right! - left!
            let padded = width + 2 * MouthCrop.horizontalPad * MouthCrop.gridMouthX
            ratio = Double(MouthCrop.width) / max(padded, 1)
            ratioFromLandmarks = e.fromLandmarks
        }
        return MouthEstimate(centre: centre!, left: left!, right: right!, fromLandmarks: e.fromLandmarks, openness: e.openness)
    }

    /// The crop's footprint in frame coordinates.
    public func box() -> Box? {
        guard let c = centre, let ratio else { return nil }
        let w = Double(MouthCrop.width) / ratio, h = Double(MouthCrop.height) / ratio
        return Box(x: c.x - w / 2, y: c.y - h / 2, w: w, h: h)
    }

    /// LipNet's 100x50 crop. Sampling the footprint directly at the track's
    /// ratio is the same as mouth.py's resize-then-cut; a footprint beyond
    /// the frame edge reads black there, with the mouth kept centred.
    public func crop(_ frame: BGRAFrame) -> MouthCropFrame {
        guard let box = box() else {
            return MouthCropFrame(rgb: [UInt8](repeating: 0, count: MouthCrop.width * MouthCrop.height * 3),
                                  gray: [Float](repeating: 0, count: MouthCrop.width * MouthCrop.height))
        }
        let (rowMajor, gray) = frame.resample(box.cgRect, width: MouthCrop.width, height: MouthCrop.height)
        var widthMajor = [UInt8](repeating: 0, count: rowMajor.count)
        for y in 0..<MouthCrop.height {
            for x in 0..<MouthCrop.width {
                let src = (y * MouthCrop.width + x) * 3, dst = (x * MouthCrop.height + y) * 3
                widthMajor[dst] = rowMajor[src]; widthMajor[dst + 1] = rowMajor[src + 1]; widthMajor[dst + 2] = rowMajor[src + 2]
            }
        }
        return MouthCropFrame(rgb: widthMajor, gray: gray.map { $0 / 255 })
    }
}

public enum FaceMotion {
    /// The face box, clipped to the frame, as 32x32 gray in 0...1 (pipeline.py _observe).
    static func sample(_ frame: BGRAFrame, face: Box) -> [Float] {
        let clipped = face.cgRect.intersection(CGRect(x: 0, y: 0, width: frame.width, height: frame.height))
        guard !clipped.isNull, clipped.width >= 1, clipped.height >= 1 else { return [Float](repeating: 0, count: 32 * 32) }
        return frame.resample(clipped, width: 32, height: 32).gray.map { $0 / 255 }
    }

    /// Mean absolute difference, the motion measure of pipeline.py.
    static func meanAbsDiff(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var sum: Float = 0
        for i in 0..<a.count { sum += abs(a[i] - b[i]) }
        return Double(sum / Float(a.count))
    }
}
