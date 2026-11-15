//
//  ResidentCube.swift
//  A cube held in one Metal buffer (storage order, in the element type it was stored with when the source hands that out, else
//  Float32; CuboKit#90) so that repeated `.metal` reductions read it in place instead of
//  copying every tile to the GPU on each call. On Apple silicon the buffer is shared memory, so the CPU reads it too: a
//  `ResidentCube` is an ordinary ``CubeSource`` and every CPU function accepts it.
//

import CuboCore
import Foundation
import Metal

/// Thrown by ``ResidentCube/init(loading:tiling:)`` for a cube that cannot be held resident (but a machine without a Metal device
/// throws `CuboVirtualError.noMetalDevice`, as before).
public enum ResidentCubeError: Error, Equatable, Sendable, LocalizedError {
    case unavailable(ResidentCube.UnavailableReason)

    public var errorDescription: String? {
        if case .unavailable(let reason) = self { return reason.message }
        return nil
    }
}

/// The Metal buffer behind a ``ResidentCube`` and the element type it holds; immutable after construction.
final class ResidentStorage: @unchecked Sendable {
    let buffer: MTLBuffer
    let elementType: ElementType
    init(buffer: MTLBuffer, elementType: ElementType) {
        self.buffer = buffer
        self.elementType = elementType
    }

    /// Converts elements `first ..< first + count` to Float32 into `destination`, as ``MappedCube`` converts.
    func convert(_ first: Int, _ count: Int, into destination: UnsafeMutablePointer<Float>) {
        elementType.convert(buffer.contents() + first * elementType.byteSize, count: count, into: destination)
    }
}

/// A cube resident in GPU-accessible memory, in its stored element type (see ``ElementType/isKeptAsStored``). For a Float32
/// cube `.metal` virtual imaging, mean and max, centre of mass and the selected-area pattern run over its buffer without a
/// copy; for any other type they convert each tile into one Float32 staging buffer. Reads (`readPattern`, `readScanTile`)
/// convert out of it, with the conversion ``MappedCube`` uses, so every value equals the Float32 path's bit for bit.
public actor ResidentCube: StoredCubeSource {
    public nonisolated let shape: CubeShape
    nonisolated let storage: ResidentStorage
    /// The element type held in the buffer.
    public nonisolated var elementType: ElementType { storage.elementType }

    /// Loads every pattern of `source`, tile by tile, into one buffer. Throws `CuboVirtualError.noMetalDevice` without a
    /// device and ``ResidentCubeError/unavailable(_:)`` when the cube is larger than the memory cap
    /// (``memoryCapFraction`` of the physical memory) or than the device's largest buffer (`maxBufferLength`); ask
    /// ``availability(of:)`` first to get the reason as a value and its words as ``UnavailableReason/message``.
    ///
    /// Order of the checks: ``availability(of:)`` reports the memory cap before the Metal device, whereas this initialiser asks for
    /// the device first. A Mac without Metal therefore always gets `noMetalDevice` here, even for a cube above the cap, where
    /// `availability` would answer `exceedsMemoryCap`.
    public init(loading source: some CubeSource, tiling: ScanTiling = .automatic) async throws {
        let metal = try MetalReducers.context()
        let shape = source.shape
        let stored = source as? any StoredCubeSource
        let type = stored?.elementType.heldType ?? .float32
        if case .unavailable(let reason) = Self.availability(of: shape, elementType: type) {
            throw ResidentCubeError.unavailable(reason)
        }
        let bytes = shape.elementCount * type.byteSize
        guard let buffer = metal.device.makeBuffer(length: max(bytes, 1), options: .storageModeShared) else {
            throw CuboVirtualError.metal("allocating CuboKit resident cube \(shape) (\(bytes) bytes)")
        }
        buffer.label = "CuboKit resident cube \(shape) \(type.rawValue)"
        let base = buffer.contents()
        for range in try tiling.ranges(for: shape) {
            try checkInterrupted()
            let target = base + range.lowerBound * shape.elementsPerScanRow * type.byteSize
            if let stored, type != .float32 {
                let tile = try await stored.readStoredScanTile(rows: range)
                guard tile.elementType == type, tile.rows == range, tile.shape == shape,
                      tile.bytes.count == range.count * shape.elementsPerScanRow * type.byteSize else {
                    throw CuboVirtualError.invalidInput("a stored tile of \(tile.elementType.rawValue) rows \(tile.rows) for \(type.rawValue) rows \(range)")
                }
                tile.bytes.withUnsafeBytes { target.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            } else {
                let tile = try await source.readScanTile(rows: range)
                tile.values.withUnsafeBytes { target.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
        }
        self.shape = shape
        self.storage = ResidentStorage(buffer: buffer, elementType: type)
    }

    /// The share of the machine's physical memory one resident cube may take (half: the Metal buffer is shared memory on Apple
    /// silicon, and the rest is the app's, the system's and the other cubes').
    public static let memoryCapFraction = 0.5

    /// Why a cube cannot be held resident here.
    public enum UnavailableReason: Equatable, Sendable {
        /// This machine has no Metal device.
        case noMetalDevice
        /// The cube (`bytes`, in the type it is held in) is larger than the memory cap (`cap` bytes, ``memoryCapFraction`` of `physicalMemory`).
        case exceedsMemoryCap(bytes: Int, cap: Int, physicalMemory: Int)
        /// The cube is larger than the device's largest single buffer.
        case exceedsMetalBufferLimit(bytes: Int, limit: Int)

        /// The reason in a sentence, for a status line or a refusal.
        public var message: String {
            func gb(_ n: Int) -> String { String(format: "%.1f GB", Double(n) / 1e9) }
            switch self {
            case .noMetalDevice: return "This Mac has no Metal device, so the cube cannot be held resident."
            case .exceedsMemoryCap(let bytes, let cap, let physical):
                return "The cube is \(gb(bytes)) in memory; a resident cube may use at most half of this Mac's memory (\(gb(cap)) of \(gb(physical)))."
            case .exceedsMetalBufferLimit(let bytes, let limit):
                return "The cube is \(gb(bytes)); the largest single Metal buffer on this Mac is \(gb(limit))."
            }
        }
    }

    public enum Availability: Equatable, Sendable {
        case available
        case unavailable(UnavailableReason)
    }

    /// Whether a cube of this shape, held as `elementType` (Float32 by default; pass the source's
    /// ``ElementType/heldType``), can be held resident on this machine, or why not. Checks, in order, the memory cap
    /// (``memoryCapFraction`` of `physicalMemory`, this machine's by default) and then the Metal device and its largest buffer.
    public static func availability(of shape: CubeShape, elementType: ElementType = .float32,
                                    physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Availability {
        let bytes = shape.elementCount * elementType.heldType.byteSize   // `CubeShape` guarantees elementCount * 8 fits an Int
        let physical = Int(clamping: physicalMemory)
        let cap = Int(Double(physical) * memoryCapFraction)
        guard bytes <= cap else { return .unavailable(.exceedsMemoryCap(bytes: bytes, cap: cap, physicalMemory: physical)) }
        guard case .success(let metal) = MetalContext.shared else { return .unavailable(.noMetalDevice) }
        guard bytes <= metal.device.maxBufferLength else {
            return .unavailable(.exceedsMetalBufferLimit(bytes: bytes, limit: metal.device.maxBufferLength))
        }
        return .available
    }

    /// Whether ``availability(of:)`` is `.available` (`false` without a Metal device, above the memory cap or the buffer limit).
    public static func fits(_ shape: CubeShape, elementType: ElementType = .float32) -> Bool {
        availability(of: shape, elementType: elementType) == .available
    }

    /// Bytes held on the GPU side: `elementCount · elementType.byteSize`.
    public nonisolated var byteCount: Int { shape.elementCount * elementType.byteSize }

    public func readPattern(row: Int, column: Int) throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let start = shape.scanIndex(row: row, column: column) * shape.detectorCount
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: values(start, shape.detectorCount))
    }

    public func readScanTile(rows: Range<Int>) throws -> ScanTile {
        try shape.requireRows(rows)
        return try ScanTile(shape: shape, rows: rows,
                            values: values(rows.lowerBound * shape.elementsPerScanRow, rows.count * shape.elementsPerScanRow))
    }

    public func readStoredScanTile(rows: Range<Int>) throws -> StoredScanTile {
        try shape.requireRows(rows)
        let size = elementType.byteSize, stride = shape.elementsPerScanRow
        let bytes = Array(UnsafeRawBufferPointer(start: storage.buffer.contents() + rows.lowerBound * stride * size,
                                                 count: rows.count * stride * size))
        return try StoredScanTile(shape: shape, rows: rows, elementType: elementType, bytes: bytes)
    }

    private func values(_ start: Int, _ count: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            if let target = buffer.baseAddress { storage.convert(start, count, into: target) }
            initialized = count
        }
    }
}
