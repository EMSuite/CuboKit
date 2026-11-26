import CuboCore
import CuboVirtual
import Foundation
import Testing
@testable import CuboBragg

/// The disk matrix from tools/bragg-matrix/pins.py: nanobeam patterns of twelve zones (fcc Al and Au, diamond Si, MgO,
/// bcc Fe, SrTiO3, hcp Mg, WS2, rutile TiO2) at 2 and 4 mrad, noiseless and at 2e4 electrons, simulated with abTEM
/// 1.0.10, and py4DSTEM 0.14.19's answer on each. End to end as a user runs it: the vacuum probe's radius
/// (`get_probe_size`), its sigmoid kernel (`Probe.get_kernel`), then `find_Bragg_disks` with poly and multicorr subpixel.
/// The fast target runs the noiseless poly cases; `matrixMulticorrAndDose` the rest.
struct BraggMatrix {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    let meta: [String: Any]
    let blob: [Double]

    init() throws {
        meta = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.directory.appendingPathComponent("bragg-matrix.json"))) as! [String: Any]
        let data = try Data(contentsOf: Self.directory.appendingPathComponent("bragg-matrix.f64"))
        blob = data.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 8).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt64.self))) }
        }
    }

    func array(_ name: String) -> [Double] {
        let entry = (meta["arrays"] as! [String: Any])[name] as! [String: Any]
        let offset = entry["offset"] as! Int, shape = entry["shape"] as! [Int]
        return Array(blob[offset..<(offset + shape.reduce(1, *))])
    }

    func scene(_ id: String) -> [String: Any] { (meta["scenes"] as! [String: Any])[id] as! [String: Any] }
    var cases: [[String: Any]] { meta["cases"] as! [[String: Any]] }

    func image(_ values: [Double], _ s: [String: Any]) -> DetectorImage {
        DetectorImage(detectorRows: s["rows"] as! Int, detectorColumns: s["columns"] as! Int, values: values, path: .cpu)
    }

    /// Runs one case and returns what differs from py4DSTEM: the radius, the kernel, the peaks, the recall.
    func check(_ c: [String: Any]) throws -> [String] {
        let id = c["id"] as! String
        let s = scene(c["scene"] as! String)
        let n = (s["rows"] as! Int) * (s["columns"] as! Int)
        var failures: [String] = []
        // The vacuum probe's radius and the sigmoid kernel, as the user builds them.
        let vacuum = image(array(s["vacuum"] as! String), s)
        let r = try ProbeSize.estimate(vacuum).radius
        let pinnedRadius = s["probe_radius"] as! Double
        if abs(r - pinnedRadius) > 1e-12 * pinnedRadius { failures.append("\(id): probe radius \(r) against \(pinnedRadius)") }
        let rows = Double(s["rows"] as! Int), columns = Double(s["columns"] as! Int)
        let kernel = try ProbeKernel.sigmoid(vacuum, innerRadius: pinnedRadius, outerRadius: 2 * pinnedRadius,
                                             profile: .sineSquared, origin: (row: rows / 2, column: columns / 2))
        let pinnedKernel = array(s["kernel"] as! String)
        let scale = pinnedKernel.map(abs).max() ?? 1
        let kernelGap = zip(kernel.values, pinnedKernel).map { abs($0 - $1) }.max() ?? 0
        if kernelGap > 1e-12 * scale { failures.append("\(id): kernel differs by \(kernelGap / scale) of its maximum") }
        // The disks, against py4DSTEM's, in order.
        let all = array(s["patterns"] as! String), p = c["pattern"] as! Int
        let pattern = image(Array(all[(p * n)..<((p + 1) * n)]), s)
        let found = try BraggDisks.detect(pattern, kernel: image(pinnedKernel, s), options: options(c["options"] as! [String: Any]))
        let expected = peaks(array(c["expected"] as! String))
        if found.count != expected.count {
            failures.append("\(id): \(found.count) peaks against py4DSTEM's \(expected.count)")
        } else {
            // As sets: symmetric discs of a noiseless zone axis tie in intensity, and a tie's order is unspecified.
            var position = 0.0, intensity = 0.0
            for q in expected {
                let nearest = found.min { a, b in
                    hypot(a.row - q.row, a.column - q.column) < hypot(b.row - q.row, b.column - q.column)
                }!
                position = max(position, hypot(nearest.row - q.row, nearest.column - q.column))
                intensity = max(intensity, abs(nearest.intensity - q.intensity) / max(abs(q.intensity), 1e-300))
            }
            if position > 1e-9 || intensity > 1e-10 {
                failures.append("\(id): position gap \(position) px, relative intensity gap \(intensity)")
            }
        }
        // The strong discs found inside their disc: at least py4DSTEM's share.
        let reflections = s["reflections"] as! [[Double]], strong = s["strong"] as! [Int]
        let reach = max(1.5, pinnedRadius)
        let hits = strong.filter { i in
            let q = reflections[i]
            return found.contains { ((($0.row - q[2]) * ($0.row - q[2]) + ($0.column - q[3]) * ($0.column - q[3])).squareRoot()) <= reach }
        }.count
        let recall = strong.isEmpty ? 1 : Double(hits) / Double(strong.count)
        let pinnedRecall = c["recall"] as! Double
        if recall < pinnedRecall - 1e-12 { failures.append("\(id): recall \(recall) below py4DSTEM's \(pinnedRecall)") }
        return failures
    }
}

@Suite("Bragg disks: the NBED matrix against py4DSTEM")
struct BraggMatrixTests {
    /// The noiseless poly case of every scene (24): runs on every pull request.
    @Test func matrixNoiselessPoly() throws {
        let m = try BraggMatrix()
        let fast = m.cases.filter { ($0["fast"] as? Bool) == true }
        #expect(fast.count == 24)
        for c in fast {
            let failures = try m.check(c)
            #expect(failures.isEmpty, "\(failures.joined(separator: "; "))")
        }
    }

    /// The multicorr and the dosed cases (72).
    @Test func matrixMulticorrAndDose() throws {
        let m = try BraggMatrix()
        for c in m.cases where (c["fast"] as? Bool) != true {
            let failures = try m.check(c)
            #expect(failures.isEmpty, "\(failures.joined(separator: "; "))")
        }
    }
}
