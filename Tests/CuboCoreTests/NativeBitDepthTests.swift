//
//  NativeBitDepthTests.swift
//  CuboKit#90: a cube kept in memory holds its stored element type, and every read equals the Float32 path bit for bit.
//

import CuboCore
import Foundation
import Testing

/// A uint16 and a uint8 cube with the same shape, their bytes and the Float32 values the old path held.
private func cubes() throws -> (shape: CubeShape, u16: [UInt16], u8: [UInt8]) {
    let shape = try CubeShape(scanRows: 7, scanColumns: 5, detectorRows: 12, detectorColumns: 9)
    var state: UInt64 = 11
    var u16 = [UInt16](), u8 = [UInt8]()
    for _ in 0..<shape.elementCount {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        u16.append(UInt16(truncatingIfNeeded: state >> 37))
        u8.append(UInt8(truncatingIfNeeded: state >> 51))
    }
    return (shape, u16, u8)
}

private func bytes<T>(_ v: [T]) -> [UInt8] { v.withUnsafeBytes { Array($0) } }
private func bits(_ v: [Float]) -> [UInt32] { v.map(\.bitPattern) }

/// A source that hands out stored tiles, as a file reader does.
private actor StoredSource: StoredCubeSource {
    nonisolated let shape: CubeShape
    nonisolated let elementType: ElementType
    private let inner: MemoryCube
    init(_ inner: MemoryCube) { self.inner = inner; shape = inner.shape; elementType = inner.elementType }
    func readPattern(row: Int, column: Int) async throws -> DiffractionPattern { try await inner.readPattern(row: row, column: column) }
    func readScanTile(rows: Range<Int>) async throws -> ScanTile { try await inner.readScanTile(rows: rows) }
    func readStoredScanTile(rows: Range<Int>) async throws -> StoredScanTile { try await inner.readStoredScanTile(rows: rows) }
}

struct NativeBitDepthTests {
    @Test func aUInt16AndAUInt8CubeHoldTheirStoredBytes() async throws {
        let (shape, u16, u8) = try cubes()
        for (type, raw, size) in [(ElementType.uint16, bytes(u16), 2), (.uint8, bytes(u8), 1)] {
            let cube = try MemoryCube(shape: shape, elementType: type, bytes: raw)
            #expect(cube.elementType == type)
            #expect(cube.byteCount == shape.elementCount * size)
            // loading from a stored source keeps the type: one byte count, not four
            let loaded = try await MemoryCube(loading: StoredSource(cube), tiling: .rows(2))
            #expect(loaded.elementType == type && loaded.byteCount == shape.elementCount * size)
            #expect(try await loaded.readStoredScanTile(rows: 0..<shape.scanRows).bytes == raw)
        }
    }

    @Test func readsEqualTheFloat32PathBitForBit() async throws {
        let (shape, u16, u8) = try cubes()
        for (type, raw, floats) in [(ElementType.uint16, bytes(u16), u16.map(Float.init)), (.uint8, bytes(u8), u8.map(Float.init))] {
            let stored = try MemoryCube(shape: shape, elementType: type, bytes: raw)
            let float = try MemoryCube(shape: shape, values: floats)
            #expect(bits(stored.values) == bits(float.values))
            for rows in [0..<1, 2..<5, 6..<7] {
                #expect(bits(try await stored.readScanTile(rows: rows).values) == bits(try await float.readScanTile(rows: rows).values))
            }
            for (r, c) in [(0, 0), (3, 4), (6, 2)] {
                #expect(bits(try await stored.readPattern(row: r, column: c).values)
                        == bits(try await float.readPattern(row: r, column: c).values))
            }
        }
    }

    @Test func aMappedUInt16FileLoadsAtTwoBytesAndReadsAsTheMapping() async throws {
        let (shape, u16, _) = try cubes()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("native-\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        let dict = "{'descr': '<u2', 'fortran_order': False, 'shape': (7, 5, 12, 9), }"
        var header = dict
        while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        data.append(contentsOf: [UInt8(header.utf8.count & 0xFF), UInt8(header.utf8.count >> 8)])
        data.append(contentsOf: Array(header.utf8))
        data.append(contentsOf: bytes(u16))
        try data.write(to: url)
        let mapped = try MappedCube(npy: url)
        let memory = try await MemoryCube(loading: mapped)
        #expect(memory.elementType == .uint16 && memory.byteCount == shape.elementCount * 2)
        #expect(bits(memory.values) == bits(try await mapped.readScanTile(rows: 0..<shape.scanRows).values))
    }

    @Test func editsGiveTheFloat32PathsResult() async throws {
        let (shape, u16, _) = try cubes()
        let stored = try MemoryCube(shape: shape, elementType: .uint16, bytes: bytes(u16))
        let float = try MemoryCube(shape: shape, values: u16.map(Float.init))
        // set_scan_shape shares the stored memory under the new shape
        let reshaped = try await CubeEditing.setScanShape(of: stored, rows: 5, columns: 7).cube
        #expect(reshaped.elementType == .uint16 && reshaped.shape.scanRows == 5)
        #expect(bits(reshaped.values) == bits(float.values))
        let viaSource = try await CubeEditing.setScanShape(of: StoredSource(stored), rows: 35, columns: 1).cube
        #expect(viaSource.elementType == .uint16 && bits(viaSource.values) == bits(float.values))
        let sa = try await CubeEditing.swapScanAxes(of: stored).cube, sb = try await CubeEditing.swapScanAxes(of: float).cube
        #expect(sa.shape == sb.shape && bits(sa.values) == bits(sb.values))
        let ra = try await CubeEditing.swapScanAndDetector(of: stored).cube
        let rb = try await CubeEditing.swapScanAndDetector(of: float).cube
        #expect(ra.shape == rb.shape && bits(ra.values) == bits(rb.values))
        let a = try await CubeEditing.binScan(of: stored, by: 2).cube, b = try await CubeEditing.binScan(of: float, by: 2).cube
        #expect(bits(a.values) == bits(b.values))
        let ha = try HotPixelFilter.filter(stored, threshold: 0.2), hb = try HotPixelFilter.filter(float, threshold: 0.2)
        #expect(bits(ha.cube.values) == bits(hb.cube.values) && ha.mask == hb.mask && bits(ha.mean) == bits(hb.mean))
    }

    @Test func theEightByteTypesAreHeldAsFloat32() throws {
        let shape = try CubeShape(scanRows: 1, scanColumns: 2, detectorRows: 1, detectorColumns: 2)
        let cube = try MemoryCube(shape: shape, elementType: .float64, bytes: bytes([1.5, -2.0, 3.25, 1e40] as [Double]))
        #expect(cube.elementType == .float32 && cube.byteCount == 16)
        #expect(cube.values == [1.5, -2, 3.25, .infinity])
    }

    @Test func aStackAboveItsCapIsRefusedWithTheNumbers() {
        #expect(throws: MemoryCapExceeded(what: "x", bytes: 800, cap: 799)) {
            try MemoryCapExceeded.check("x", count: 100, elementBytes: 8, cap: 799)
        }
        #expect(throws: Never.self) { try MemoryCapExceeded.check("x", count: 100, elementBytes: 8, cap: 800) }
    }
}
