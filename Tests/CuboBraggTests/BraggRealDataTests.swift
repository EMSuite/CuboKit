import CuboCore
import CuboVirtual
import Foundation
import Testing
@testable import CuboBragg

/// The disk finder on recorded data, from tools/bragg-real/pins.py: 14 patterns of two public 4D-STEM datasets (Zenodo
/// 14113591, a precession scan of a twinned GaAs nanowire, 144 x 144 uint8; Zenodo 8233585, NiO and layered oxide
/// grains on a four-chip detector with its gaps, 384 x 384 uint8; both CC-BY-4.0), each with py4DSTEM 0.14.19's answer.
/// End to end as py4DSTEM's tutorial (orient_strain_01_WS2) runs a dataset without a vacuum probe: the probe radius of
/// the scan's mean pattern (`get_probe_size`), a synthetic probe of that radius (`generate_synthetic_probe`, width 0.7),
/// its sigmoid kernel (radii 1.2 r and 4 r), then `find_Bragg_disks` with the tutorial's keywords and the two it tunes
/// per dataset (the absolute floor just above the off-sample noise peaks, the spacing under the spot spacing). One
/// pattern per dataset is off the sample: noise alone, nothing but the direct beam found.
/// Tolerances are the simulated matrix's (BraggMatrixTests): the same arithmetic on the same Double inputs.
struct BraggRealData {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    let meta: [String: Any]
    let doubles: [Double]
    let bytes: [UInt8]

    init() throws {
        meta = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.directory.appendingPathComponent("bragg-real.json"))) as! [String: Any]
        let data = try Data(contentsOf: Self.directory.appendingPathComponent("bragg-real.f64"))
        doubles = data.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 8).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt64.self))) }
        }
        bytes = try [UInt8](Data(contentsOf: Self.directory.appendingPathComponent("bragg-real.u8")))
    }

    func array(_ name: String) -> [Double] {
        let entry = (meta["arrays"] as! [String: Any])[name] as! [String: Any]
        let offset = entry["offset"] as! Int, count = (entry["shape"] as! [Int]).reduce(1, *)
        if entry["file"] as! String == "u8" { return bytes[offset..<(offset + count)].map(Double.init) }
        return Array(doubles[offset..<(offset + count)])
    }

    func scene(_ id: String) -> [String: Any] { (meta["scenes"] as! [String: Any])[id] as! [String: Any] }
    var cases: [[String: Any]] { meta["cases"] as! [[String: Any]] }

    func image(_ values: [Double], _ s: [String: Any]) -> DetectorImage {
        DetectorImage(detectorRows: s["rows"] as! Int, detectorColumns: s["columns"] as! Int, values: values, path: .cpu)
    }

    /// The scene's probe radius and kernel built as the user builds them; what differs from py4DSTEM.
    func checkKernel(_ id: String) throws -> [String] {
        let s = scene(id)
        var failures: [String] = []
        let r = try ProbeSize.estimate(image(array(s["mean"] as! String), s)).radius
        let pinnedRadius = s["probe_radius"] as! Double
        if abs(r - pinnedRadius) > 1e-12 * pinnedRadius { failures.append("\(id): probe radius \(r) against \(pinnedRadius)") }
        let rows = s["rows"] as! Int, columns = s["columns"] as! Int
        let probe = try SyntheticProbe.make(radius: pinnedRadius, width: s["synthetic_width"] as! Double,
                                            detectorRows: rows, detectorColumns: columns)
        let radii = s["kernel_radii"] as! [Double]
        // py4DSTEM's get_kernel(origin=(Q[0]/2, Q[0]/2)); both detectors are square.
        let kernel = try ProbeKernel.sigmoid(probe, innerRadius: radii[0], outerRadius: radii[1], profile: .sineSquared,
                                             origin: (row: Double(rows) / 2, column: Double(rows) / 2))
        let pinnedKernel = array(s["kernel"] as! String)
        let scale = pinnedKernel.map(abs).max() ?? 1
        let gap = zip(kernel.values, pinnedKernel).map { abs($0 - $1) }.max() ?? 0
        if gap > 1e-12 * scale { failures.append("\(id): kernel differs by \(gap / scale) of its maximum") }
        return failures
    }

    /// One recorded pattern through the finder with py4DSTEM's kernel; what differs from py4DSTEM's peaks (as sets).
    func checkCase(_ c: [String: Any]) throws -> [String] {
        let id = c["id"] as! String
        let s = scene(c["scene"] as! String)
        let n = (s["rows"] as! Int) * (s["columns"] as! Int)
        let all = array(s["patterns"] as! String), p = c["pattern"] as! Int
        let pattern = image(Array(all[(p * n)..<((p + 1) * n)]), s)
        let kernel = image(array(s["kernel"] as! String), s)
        let found = try BraggDisks.detect(pattern, kernel: kernel, options: options(c["options"] as! [String: Any]))
        let expected = peaks(array(c["expected"] as! String))
        guard found.count == expected.count else { return ["\(id): \(found.count) peaks against py4DSTEM's \(expected.count)"] }
        var position = 0.0, intensity = 0.0
        for q in expected {
            let nearest = found.min { a, b in hypot(a.row - q.row, a.column - q.column) < hypot(b.row - q.row, b.column - q.column) }!
            position = max(position, hypot(nearest.row - q.row, nearest.column - q.column))
            intensity = max(intensity, abs(nearest.intensity - q.intensity) / max(abs(q.intensity), 1e-300))
        }
        if position > 1e-9 || intensity > 1e-10 {
            return ["\(id): position gap \(position) px, relative intensity gap \(intensity)"]
        }
        return []
    }
}

@Suite("Bragg disks: recorded public data against py4DSTEM")
struct BraggRealDataTests {
    @Test func fixtureCoversBothDatasets() throws {
        let m = try BraggRealData()
        #expect((m.meta["versions"] as! [String: String])["py4DSTEM"] == "0.14.19")
        #expect(Set(m.cases.map { $0["scene"] as! String }) == ["nanowire", "nio"])
        #expect(m.cases.count == 14)
        // The off-sample patterns: noise alone gives the direct beam at most.
        for id in ["nanowire_0_27", "nio_0_0"] {
            let c = try #require(m.cases.first { $0["id"] as! String == id })
            #expect(peaks(m.array(c["expected"] as! String)).count <= 1, "\(id)")
        }
    }

    @Test(arguments: ["nanowire", "nio"])
    func probeRadiusAndKernel(_ scene: String) throws {
        let failures = try BraggRealData().checkKernel(scene)
        #expect(failures.isEmpty, "\(failures.joined(separator: "; "))")
    }

    @Test func disksOnRecordedPatterns() throws {
        let m = try BraggRealData()
        for c in m.cases {
            let failures = try m.checkCase(c)
            #expect(failures.isEmpty, "\(failures.joined(separator: "; "))")
        }
    }
}
