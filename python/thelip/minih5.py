"""A minimal HDF5 reader for Keras weight files: superblock v0/v1, v1 object
headers, old-style groups (symbol table + B-tree + local heap), contiguous
datasets of little-endian floats/ints. Enough to pull weights out of an
.h5 written by h5py without h5py.

Reference: the HDF5 File Format Specification (v2/v3).
"""
import struct
import numpy as np


class H5:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        b = self.buf
        assert b[:8] == b"\x89HDF\r\n\x1a\n", "not an HDF5 file"
        version = b[8]
        if version in (0, 1):
            self.offset_size = b[13]
            self.length_size = b[14]
            # group leaf K (2), group internal K (2), consistency flags (4),
            # and in version 1 the indexed-storage K (2) + reserved (2).
            pos = 24 + (4 if version == 1 else 0)
            base_addr = self._addr(pos)
            pos += self.offset_size
            pos += self.offset_size  # free-space info address
            self.eof = self._addr(pos)
            pos += self.offset_size
            pos += self.offset_size  # driver information block
            # Root group symbol table entry.
            self.root = self._symbol_table_entry(pos)
        elif version in (2, 3):
            self.offset_size = b[9]
            self.length_size = b[10]
            pos = 12
            base = self._addr(pos); pos += self.offset_size
            pos += self.offset_size  # superblock extension
            self.eof = self._addr(pos); pos += self.offset_size
            root_header = self._addr(pos)
            self.root = {"name_off": 0, "header": root_header, "btree": None, "heap": None}
        else:
            raise ValueError(f"unsupported superblock version {version}")

    # --- primitives -------------------------------------------------------
    def _addr(self, pos):
        return int.from_bytes(self.buf[pos:pos + self.offset_size], "little")

    def _len(self, pos):
        return int.from_bytes(self.buf[pos:pos + self.length_size], "little")

    def _symbol_table_entry(self, pos):
        name_off = self._addr(pos); pos += self.offset_size
        header = self._addr(pos); pos += self.offset_size
        cache_type = struct.unpack_from("<I", self.buf, pos)[0]; pos += 4
        pos += 4  # reserved
        entry = {"name_off": name_off, "header": header, "btree": None, "heap": None}
        if cache_type == 1:
            entry["btree"] = self._addr(pos)
            entry["heap"] = self._addr(pos + self.offset_size)
        return entry

    # --- object headers -----------------------------------------------------
    def messages(self, header_addr):
        """All (type, data bytes) messages of an object header, v1 or v2."""
        b = self.buf
        out = []
        if b[header_addr] == 1:
            pos = header_addr
            num = struct.unpack_from("<H", b, pos + 2)[0]
            size = struct.unpack_from("<I", b, pos + 8)[0]
            pos += 16  # version(1) reserved(1) nmsgs(2) refcount(4) size(4) + 4 pad
            end = pos + size
            self._read_v1_messages(pos, end, num, out)
        elif b[header_addr:header_addr + 4] == b"OHDR":
            self._read_v2_header(header_addr, out)
        else:
            raise ValueError(f"unknown object header at {header_addr}: {b[header_addr:header_addr+4]!r}")
        return out

    def _read_v1_messages(self, pos, end, num, out):
        b = self.buf
        count = 0
        while pos + 8 <= end and count < num:
            mtype, msize, mflags = struct.unpack_from("<HHB", b, pos)
            pos += 8
            data = b[pos:pos + msize]
            if mtype == 0x10:  # continuation
                cont_addr = self._addr(pos)
                cont_len = self._len(pos + self.offset_size)
                self._read_v1_messages(cont_addr, cont_addr + cont_len, num - count - 1, out)
            else:
                out.append((mtype, data))
            pos += msize
            count += 1

    def _read_v2_header(self, addr, out):
        b = self.buf
        pos = addr + 4
        version = b[pos]; pos += 1
        flags = b[pos]; pos += 1
        if flags & 0x20:
            pos += 16
        if flags & 0x10:
            pos += 4
        size_bytes = 1 << (flags & 0x3)
        chunk_size = int.from_bytes(b[pos:pos + size_bytes], "little"); pos += size_bytes
        self._read_v2_messages(pos, pos + chunk_size, flags, out)

    def _read_v2_messages(self, pos, end, flags, out):
        b = self.buf
        while pos + 4 <= end - 4:  # trailing checksum
            mtype = b[pos]; msize = struct.unpack_from("<H", b, pos + 1)[0]; mflags = b[pos + 3]
            pos += 4
            if flags & 0x4:
                pos += 2
            data = b[pos:pos + msize]
            if mtype == 0x10:
                cont_addr = self._addr(pos)
                cont_len = self._len(pos + self.offset_size)
                # continuation block: "OCHK" signature then messages
                self._read_v2_messages(cont_addr + 4, cont_addr + cont_len, flags, out)
            else:
                out.append((mtype, data))
            pos += msize

    # --- groups ---------------------------------------------------------------
    def children(self, entry):
        """{name: symbol-table-entry-like dict} of a group."""
        msgs = self.messages(entry["header"])
        result = {}
        for mtype, data in msgs:
            if mtype == 0x11:  # symbol table message
                btree = int.from_bytes(data[:self.offset_size], "little")
                heap = int.from_bytes(data[self.offset_size:2 * self.offset_size], "little")
                result.update(self._walk_btree(btree, heap))
            elif mtype == 0x06:  # link message (new-style groups)
                result.update(self._link_message(data))
            elif mtype == 0x02:  # link info: may point at a fractal heap (dense storage)
                pass
        return result

    def _link_message(self, data):
        pos = 0
        version = data[pos]; pos += 1
        flags = data[pos]; pos += 1
        link_type = 0
        if flags & 0x8:
            link_type = data[pos]; pos += 1
        if flags & 0x4:
            pos += 8
        if flags & 0x10:
            pos += 1  # charset
        size_bytes = 1 << (flags & 0x3)
        name_len = int.from_bytes(data[pos:pos + size_bytes], "little"); pos += size_bytes
        name = data[pos:pos + name_len].decode(); pos += name_len
        if link_type != 0:
            return {}
        header = int.from_bytes(data[pos:pos + self.offset_size], "little")
        return {name: {"name_off": 0, "header": header, "btree": None, "heap": None}}

    def _heap_string(self, heap_addr, offset):
        b = self.buf
        assert b[heap_addr:heap_addr + 4] == b"HEAP", "bad local heap"
        data_addr = self._addr(heap_addr + 8 + self.length_size * 2)
        start = data_addr + offset
        end = b.index(b"\x00", start)
        return b[start:end].decode()

    def _walk_btree(self, addr, heap):
        b = self.buf
        assert b[addr:addr + 4] == b"TREE", f"bad B-tree at {addr}"
        node_type = b[addr + 4]
        level = b[addr + 5]
        entries = struct.unpack_from("<H", b, addr + 6)[0]
        pos = addr + 8 + 2 * self.offset_size  # skip left/right siblings
        result = {}
        for i in range(entries):
            pos += self.length_size  # key (heap offset of the largest name), ignored
            child = self._addr(pos); pos += self.offset_size
            if level == 0:
                result.update(self._snod(child, heap))
            else:
                result.update(self._walk_btree(child, heap))
        return result

    def _snod(self, addr, heap):
        b = self.buf
        assert b[addr:addr + 4] == b"SNOD", f"bad SNOD at {addr}"
        count = struct.unpack_from("<H", b, addr + 6)[0]
        pos = addr + 8
        result = {}
        for _ in range(count):
            entry = self._symbol_table_entry(pos)
            pos += 2 * self.offset_size + 8 + 16
            result[self._heap_string(heap, entry["name_off"])] = entry
        return result

    # --- datasets ---------------------------------------------------------------
    def dataset(self, entry):
        msgs = self.messages(entry["header"])
        dims = None
        dtype = None
        layout = None
        for mtype, data in msgs:
            if mtype == 0x01:  # dataspace
                version = data[0]
                rank = data[1]
                if version == 1:
                    pos = 8
                else:
                    pos = 4
                dims = [int.from_bytes(data[pos + i * self.length_size:pos + (i + 1) * self.length_size], "little") for i in range(rank)]
            elif mtype == 0x03:  # datatype
                cls = data[0] & 0x0F
                size = struct.unpack_from("<I", data, 4)[0]
                bit0 = data[1] & 1  # byte order: 0 little
                if cls == 1:
                    dtype = np.dtype(("<" if bit0 == 0 else ">") + f"f{size}")
                elif cls == 0:
                    signed = (data[1] >> 3) & 1
                    dtype = np.dtype(("<" if bit0 == 0 else ">") + ("i" if signed else "u") + str(size))
                else:
                    dtype = ("class", cls, size)
            elif mtype == 0x08:  # layout
                version = data[0]
                if version == 3:
                    cls = data[1]
                    if cls == 1:  # contiguous
                        addr = int.from_bytes(data[2:2 + self.offset_size], "little")
                        size = int.from_bytes(data[2 + self.offset_size:2 + self.offset_size + self.length_size], "little")
                        layout = ("contiguous", addr, size)
                    elif cls == 0:  # compact
                        size = struct.unpack_from("<H", data, 2)[0]
                        layout = ("compact", data[4:4 + size])
                    elif cls == 2:
                        layout = ("chunked", data)
                elif version in (1, 2):
                    rank = data[1]
                    cls = data[2]
                    pos = 8
                    if cls == 1:
                        addr = int.from_bytes(data[pos:pos + self.offset_size], "little")
                        pos += self.offset_size
                        sizes = [struct.unpack_from("<I", data, pos + 4 * i)[0] for i in range(rank)]
                        layout = ("contiguous", addr, int(np.prod(sizes)))
                    elif cls == 0:
                        pos += 4 * rank
                        size = struct.unpack_from("<I", data, pos)[0]
                        layout = ("compact", data[pos + 4:pos + 4 + size])
        if dims is None or dtype is None or layout is None:
            raise ValueError(f"not a readable dataset: dims={dims} dtype={dtype} layout={layout}")
        if not isinstance(dtype, np.dtype):
            raise ValueError(f"unsupported datatype {dtype}")
        count = int(np.prod(dims)) if dims else 1
        if layout[0] == "contiguous":
            raw = self.buf[layout[1]:layout[1] + count * dtype.itemsize]
        elif layout[0] == "compact":
            raw = layout[1]
        else:
            raise ValueError("chunked datasets are not supported")
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(dims)

    def is_group(self, entry):
        return any(t in (0x11, 0x02, 0x06) for t, _ in self.messages(entry["header"]))

    def tree(self, entry=None, prefix=""):
        """Yield (path, entry, is_group) for everything under a group."""
        entry = entry or self.root
        for name, child in self.children(entry).items():
            path = f"{prefix}/{name}"
            if self.is_group(child):
                yield path, child, True
                yield from self.tree(child, path)
            else:
                yield path, child, False

    def arrays(self):
        """{path: ndarray} for every dataset in the file."""
        out = {}
        for path, entry, is_group in self.tree():
            if not is_group:
                out[path] = self.dataset(entry)
        return out


if __name__ == "__main__":
    import sys
    h = H5(sys.argv[1])
    for path, entry, is_group in h.tree():
        if is_group:
            print(path + "/")
        else:
            a = h.dataset(entry)
            print(f"{path}  {a.shape} {a.dtype}")
