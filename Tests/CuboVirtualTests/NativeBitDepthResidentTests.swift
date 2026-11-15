//
//  NativeBitDepthResidentTests.swift
//  CuboKit#90: a resident cube holds its stored element type; every reduction over it equals the Float32 resident cube's, bit
//  for bit, on the CPU and on Metal.
//

import CuboCore
@testable import CuboVirtual
import Foundation
import Metal
import Testing

/// 9 x 8 x 40 x 36: a uint16 and a uint8 cube, their stored-type MemoryCubes and the Float32 cubes of the same values.
private func pairs() throws -> [(type: ElementType, stored: MemoryCube, float: MemoryCube, size: Int)] {
    let shape = try CubeShape(scanRows: 9, scanColumns: 8, detectorRows: 40, detectorColumns: 36)
    var state: UInt64 = 5
    var u16 = [UInt16](), u8 = [UInt8]()
    for _ in 0..<shape.elementCount {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        u16.append(UInt16(truncatingIfNeeded: state >> 37))
        u8.append(UInt8(truncatingIfNeeded: state >> 51))
    }
    return [
        (.uint16, try MemoryCube(shape: shape, elementType: .uint16, bytes: u16.withUnsafeBytes { Array($0) }),
         try MemoryCube(shape: shape, values: u16.map(Float.init)), 2),
        (.uint8, try MemoryCube(shape: shape, elementType: .uint8, bytes: u8.withUnsafeBytes { Array($0) }),
         try MemoryCube(shape: shape, values: u8.map(Float.init)), 1),
    ]
}

private func bits(_ v: [Double]) -> [UInt64] { v.map(\.bitPattern) }

@Suite(.serialized, .enabled(if: VirtualDiffraction.metalIsAvailable, "no Metal device on this machine"))
struct NativeBitDepthResidentTests {
    @Test func theResidentBufferHoldsTheStoredBytes() async throws {
        for p in try pairs() {
            let resident = try await ResidentCube(loading: p.stored, tiling: .rows(2))
            #expect(resident.elementType == p.type)
            #expect(resident.byteCount == p.stored.shape.elementCount * p.size)
            #expect(resident.storage.buffer.length == p.stored.shape.elementCount * p.size)
            #expect(ResidentCube.availability(of: p.stored.shape, elementType: p.type, physicalMemory: UInt64(4 * resident.byteCount))
                    == .available || !VirtualDiffraction.metalIsAvailable)
            let floatResident = try await ResidentCube(loading: p.float)
            #expect(floatResident.elementType == .float32 && floatResident.byteCount == p.float.shape.elementCount * 4)
            #expect(try TileFeed(MetalReducers.context(), resident, tiling: .automatic).kind == .converting)
            #expect(try TileFeed(MetalReducers.context(), p.stored, tiling: .automatic).kind == .converting)
        }
    }

    @Test func reductionsEqualTheFloat32ResidentCubeBitForBit() async throws {
        let geometry = DetectorGeometry.annulus(centerRow: 19.5, centerColumn: 17, inner: 4, outer: 13)
        for p in try pairs() {
            let stored = try await ResidentCube(loading: p.stored)
            let float = try await ResidentCube(loading: p.float)
            for rows in [0..<1, 3..<9] {
                #expect(try await stored.readScanTile(rows: rows) == float.readScanTile(rows: rows))
            }
            #expect(try await stored.readPattern(row: 4, column: 5) == float.readPattern(row: 4, column: 5))
            for path in ComputePath.allCases {
                for (a, b) in [(stored as any CubeSource, float as any CubeSource), (p.stored, p.float)] {
                    let ia = try await VirtualImaging.image(of: a, geometry: geometry, path: path, tiling: .rows(4))
                    let ib = try await VirtualImaging.image(of: b, geometry: geometry, path: path, tiling: .rows(4))
                    #expect(bits(ia.values) == bits(ib.values))
                    let ma = try await VirtualDiffraction.meanAndMax(of: a, path: path)
                    let mb = try await VirtualDiffraction.meanAndMax(of: b, path: path)
                    #expect(bits(ma.mean.values) == bits(mb.mean.values) && bits(ma.max.values) == bits(mb.max.values))
                    let ca = try await CenterOfMass.measure(of: a, path: path), cb = try await CenterOfMass.measure(of: b, path: path)
                    #expect(bits(ca.row.values) == bits(cb.row.values) && bits(ca.column.values) == bits(cb.column.values))
                }
            }
        }
    }

    @Test func theCapIsCheckedAtTheStoredSize() {
        guard let shape = try? CubeShape(scanRows: 100, scanColumns: 100, detectorRows: 100, detectorColumns: 100) else { return }
        // 1e8 elements: 400 MB in Float32, 200 MB in uint16; a 600 MB machine may hold only the uint16 cube (cap 300 MB)
        #expect(ResidentCube.availability(of: shape, elementType: .uint16, physicalMemory: 600_000_000) == .available)
        #expect(ResidentCube.availability(of: shape, physicalMemory: 600_000_000)
                == .unavailable(.exceedsMemoryCap(bytes: 400_000_000, cap: 300_000_000, physicalMemory: 600_000_000)))
    }
}
