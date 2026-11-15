//
//  MemoryCube.swift
//  The eager cube: the whole array in memory in storage order, in the element type it was stored with (Float32 when made from
//  Float32 values or from a source that hands out only Float32). Reads convert per pattern or tile, as MappedCube does, so
//  a MemoryCube equals the lazy MappedCube of the same bytes, bit for bit (tested in CuboCoreTests).
//

/// A cube held in memory, `[ry][rx][qy][qx]`: Float32, or the stored element type when loaded from a ``StoredCubeSource``
/// (see ``ElementType/isKeptAsStored``). Every read is Float32.
public actor MemoryCube: StoredCubeSource {
    public nonisolated let shape: CubeShape
    /// The element type held in memory.
    public nonisolated let elementType: ElementType
    nonisolated let storage: Storage

    enum Storage: Sendable {
        case float([Float])
        case stored(StoredBytes)
    }

    /// Every value in Float32, in storage order. For a cube held in its stored type this converts the whole cube into a new
    /// array on each call: read tiles instead where memory matters.
    public nonisolated var values: [Float] {
        switch storage {
        case .float(let v): v
        case .stored(let s): s.floats(0, s.count)
        }
    }

    /// The Float32 values when the cube is held as Float32 (read in place without a copy), else nil.
    package nonisolated var floatValues: [Float]? {
        if case .float(let v) = storage { return v }
        return nil
    }

    /// Bytes held: `elementCount · elementType.byteSize`.
    public nonisolated var byteCount: Int { shape.elementCount * elementType.byteSize }

    /// A cube from its Float32 values in storage order.
    public init(shape: CubeShape, values: [Float]) throws {
        guard values.count == shape.elementCount else {
            throw CuboError.valueCountMismatch(expected: shape.elementCount, received: values.count)
        }
        self.init(shape: shape, elementType: .float32, storage: .float(values))
    }

    private init(shape: CubeShape, elementType: ElementType, storage: Storage) {
        self.shape = shape
        self.elementType = elementType
        self.storage = storage
    }

    /// A cube from its elements in `elementType`, native little-endian, in storage order. A type that is not
    /// ``ElementType/isKeptAsStored`` is converted to Float32.
    public init(shape: CubeShape, elementType: ElementType, bytes: [UInt8]) throws {
        let tile = try StoredScanTile(shape: shape, rows: 0..<shape.scanRows, elementType: elementType, bytes: bytes)
        if elementType == .float32 || !elementType.isKeptAsStored {
            try self.init(shape: shape, values: tile.floatValues())
        } else {
            let store = StoredBytes(elementType: elementType, count: shape.elementCount)
            store.write(tile, atElement: 0)
            self.init(shape: shape, elementType: elementType, storage: .stored(store))
        }
    }

    /// Shares `other`'s memory under a new shape with the same element count.
    init(reshaping other: MemoryCube, to shape: CubeShape) throws {
        guard shape.elementCount == other.shape.elementCount else {
            throw CuboError.valueCountMismatch(expected: shape.elementCount, received: other.shape.elementCount)
        }
        self.init(shape: shape, elementType: other.elementType, storage: other.storage)
    }

    /// Every pattern of `source`, read tile by tile (the eager path of a lazy source). A ``StoredCubeSource`` whose type is
    /// ``ElementType/isKeptAsStored`` is held in that type; any other source is held as Float32.
    public init(loading source: some CubeSource, tiling: ScanTiling = .automatic) async throws {
        let shape = source.shape
        if let stored = source as? any StoredCubeSource, stored.elementType.isKeptAsStored, stored.elementType != .float32 {
            let type = stored.elementType
            let store = StoredBytes(elementType: type, count: shape.elementCount)
            for range in try tiling.ranges(for: shape) {
                try Task.checkCancellation()
                let tile = try await stored.readStoredScanTile(rows: range)
                let expected = range.count * shape.elementsPerScanRow * type.byteSize
                guard tile.elementType == type, tile.rows == range, tile.shape == shape, tile.bytes.count == expected else {
                    throw CuboError.valueCountMismatch(expected: expected, received: tile.bytes.count)
                }
                store.write(tile, atElement: range.lowerBound * shape.elementsPerScanRow)
            }
            self.init(shape: shape, elementType: type, storage: .stored(store))
            return
        }
        var all = [Float]()
        all.reserveCapacity(shape.elementCount)
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            all.append(contentsOf: tile.values)
        }
        try self.init(shape: shape, values: all)
    }

    public func readPattern(row: Int, column: Int) throws -> DiffractionPattern {
        try shape.requirePosition(row: row, column: column)
        let start = shape.scanIndex(row: row, column: column) * shape.detectorCount
        return try DiffractionPattern(detectorRows: shape.detectorRows, detectorColumns: shape.detectorColumns,
                                      values: floats(start, shape.detectorCount))
    }

    public func readScanTile(rows: Range<Int>) throws -> ScanTile {
        try shape.requireRows(rows)
        let stride = shape.elementsPerScanRow
        return try ScanTile(shape: shape, rows: rows, values: floats(rows.lowerBound * stride, rows.count * stride))
    }

    public func readStoredScanTile(rows: Range<Int>) throws -> StoredScanTile {
        try shape.requireRows(rows)
        let stride = shape.elementsPerScanRow, first = rows.lowerBound * stride, count = rows.count * stride
        let bytes: [UInt8]
        switch storage {
        case .float(let v): bytes = v[first..<(first + count)].withUnsafeBytes { Array($0) }
        case .stored(let s): bytes = s.bytes(first, count)
        }
        return try StoredScanTile(shape: shape, rows: rows, elementType: elementType, bytes: bytes)
    }

    /// Converts elements `first ..< first + count` to Float32 into `destination` (no intermediate array).
    package nonisolated func convert(_ first: Int, _ count: Int, into destination: UnsafeMutablePointer<Float>) {
        switch storage {
        case .float(let v): v.withUnsafeBufferPointer { destination.update(from: $0.baseAddress! + first, count: count) }
        case .stored(let s): s.convert(first, count, into: destination)
        }
    }

    private func floats(_ first: Int, _ count: Int) -> [Float] {
        switch storage {
        case .float(let v): Array(v[first..<(first + count)])
        case .stored(let s): s.floats(first, count)
        }
    }
}
