//
//  StoredCube.swift
//  Tiles in the stored element type, and the memory that holds a cube in it (CuboKit#90). A cube kept in memory holds the
//  bytes it was stored with (uint8, uint16, ...) and converts each read to Float32 with ``ElementType``'s conversion, the
//  same one ``MappedCube`` uses, so a value read from memory equals the value read from the file, bit for bit. Float32
//  exists only while a computation runs.
//

import Foundation

/// Contiguous scan rows of a cube in its stored element type: `[rows.count][scanColumns][detectorRows][detectorColumns]`
/// elements, native little-endian.
public struct StoredScanTile: Sendable, Equatable {
    /// The cube's shape (not the tile's).
    public let shape: CubeShape
    /// Which scan rows these are.
    public let rows: Range<Int>
    public let elementType: ElementType
    /// `rows.count · elementsPerScanRow · elementType.byteSize` bytes.
    public let bytes: [UInt8]

    public init(shape: CubeShape, rows: Range<Int>, elementType: ElementType, bytes: [UInt8]) throws {
        try shape.requireRows(rows)
        let expected = rows.count * shape.elementsPerScanRow * elementType.byteSize
        guard bytes.count == expected else { throw CuboError.valueCountMismatch(expected: expected, received: bytes.count) }
        self.shape = shape
        self.rows = rows
        self.elementType = elementType
        self.bytes = bytes
    }

    /// Elements in the tile.
    public var elementCount: Int { rows.count * shape.elementsPerScanRow }

    /// The tile in Float32, converted as ``MappedCube`` converts.
    public func floatValues() -> [Float] {
        let type = elementType, count = elementCount
        return bytes.withUnsafeBytes { raw in
            [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                if let base = buffer.baseAddress { type.convertAligning(raw.baseAddress!, count: count, into: base) }
                initialized = count
            }
        }
    }
}

/// A source that can hand out its tiles in the element type it stores, without converting them. A consumer that keeps the
/// cube in memory (``MemoryCube``, CuboVirtual's `ResidentCube`, an app's own resident cube) asks for these so it holds the
/// stored bytes and not Float32.
public protocol StoredCubeSource: CubeSource {
    /// The element type the tiles come in.
    nonisolated var elementType: ElementType { get }
    /// Contiguous scan rows in ``elementType``; `rows` must lie inside the cube and must not be empty.
    func readStoredScanTile(rows: Range<Int>) async throws -> StoredScanTile
}

extension ElementType {
    /// Whether a cube of this type is kept in memory as stored: every type of at most four bytes. The 8-byte types are held as
    /// Float32, which is half their size (and is what every read hands out anyway).
    public var isKeptAsStored: Bool { byteSize <= MemoryLayout<Float>.stride }

    /// The element type a cube stored as `self` is held in memory with: itself when ``isKeptAsStored``, else Float32.
    public var heldType: ElementType { isKeptAsStored ? self : .float32 }

    /// ``convert(_:count:into:)`` for a source that may not be aligned to ``byteSize`` (an `[UInt8]`'s storage always is in
    /// practice; this copies through an aligned buffer when it is not).
    package func convertAligning(_ source: UnsafeRawPointer, count: Int, into destination: UnsafeMutablePointer<Float>) {
        if Int(bitPattern: source) % byteSize == 0 {
            convert(source, count: count, into: destination)
            return
        }
        let bytes = count * byteSize
        let aligned = UnsafeMutableRawPointer.allocate(byteCount: max(bytes, 1), alignment: 16)
        defer { aligned.deallocate() }
        aligned.copyMemory(from: source, byteCount: bytes)
        convert(aligned, count: count, into: destination)
    }
}

/// One cube's elements in their stored type, in one 16-byte-aligned allocation; written only while the owner is built,
/// read-only after (which is what the unchecked conformance states).
package final class StoredBytes: @unchecked Sendable {
    package let elementType: ElementType
    package let count: Int
    package let base: UnsafeMutableRawPointer
    package var byteCount: Int { count * elementType.byteSize }

    /// Uninitialised room for `count` elements; the owner fills it before the first read.
    package init(elementType: ElementType, count: Int) {
        self.elementType = elementType
        self.count = count
        self.base = UnsafeMutableRawPointer.allocate(byteCount: max(count * elementType.byteSize, 1), alignment: 16)
    }

    deinit { base.deallocate() }

    /// Copies the stored bytes of `tile` to element `first`.
    package func write(_ tile: StoredScanTile, atElement first: Int) {
        tile.bytes.withUnsafeBytes { raw in
            (base + first * elementType.byteSize).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
    }

    /// Elements `first ..< first + count` in Float32.
    package func floats(_ first: Int, _ count: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            if let target = buffer.baseAddress { convert(first, count, into: target) }
            initialized = count
        }
    }

    /// Converts elements `first ..< first + count` to Float32 into `destination`.
    package func convert(_ first: Int, _ count: Int, into destination: UnsafeMutablePointer<Float>) {
        elementType.convert(base + first * elementType.byteSize, count: count, into: destination)
    }

    /// Elements `first ..< first + count` as stored bytes.
    package func bytes(_ first: Int, _ count: Int) -> [UInt8] {
        let size = elementType.byteSize
        return Array(UnsafeRawBufferPointer(start: base + first * size, count: count * size))
    }
}

/// Thrown where a computation must hold a whole stack in memory and the stack is larger than its cap (CuboKit#90): a refusal
/// with the numbers, instead of an allocation that pushes the machine into swap.
public struct MemoryCapExceeded: Error, Equatable, Sendable, LocalizedError {
    /// What would have been held.
    public let what: String
    /// Bytes it would take, and the cap.
    public let bytes: Int
    public let cap: Int

    public init(what: String, bytes: Int, cap: Int) {
        self.what = what
        self.bytes = bytes
        self.cap = cap
    }

    /// The default cap: half the physical memory, the share a resident cube may take.
    public static func defaultCap(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        Int(clamping: physicalMemory / 2)
    }

    /// Throws unless `count` elements of `elementBytes` each fit `cap` (an overflow does not fit).
    public static func check(_ what: String, count: Int, elementBytes: Int, cap: Int) throws {
        let (bytes, overflow) = count.multipliedReportingOverflow(by: elementBytes)
        guard !overflow, bytes <= cap else { throw MemoryCapExceeded(what: what, bytes: overflow ? .max : bytes, cap: cap) }
    }

    public var errorDescription: String? {
        String(format: "%@ would take %.1f GB in memory; the limit is %.1f GB.", what, Double(bytes) / 1e9, Double(cap) / 1e9)
    }
}
