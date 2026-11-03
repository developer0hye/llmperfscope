"""Reads what a GGUF file says about itself: its metadata and the list of its tensors, without touching the weights."""

from __future__ import annotations

import os
import re
import struct

_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING = 8
_ARRAY = 9
# An array longer than this is kept as its length: a vocabulary has hundreds of thousands of strings nobody here needs.
_KEEP_ARRAY = 256
_SPLIT_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")

# ggml's type numbers, for labels only. A type missing here shows as `type N` and everything still works.
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}


def type_name(number):
    return TYPE_NAMES.get(number, f"type {number}")


class _Reader:
    def __init__(self, f):
        self.f = f

    def fixed(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) < size:
            raise ValueError("the file ends inside its header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        length = self.fixed("<Q")
        data = self.f.read(length)
        if len(data) < length:
            raise ValueError("the file ends inside its header")
        return data.decode("utf-8", "replace")

    def value(self, kind):
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            inner = self.fixed("<I")
            count = self.fixed("<Q")
            if count > _KEEP_ARRAY:
                self._skip(inner, count)
                return {"array_length": count}
            return [self.value(inner) for _ in range(count)]
        fmt = _FIXED.get(kind)
        if fmt is None:
            raise ValueError(f"unknown metadata type {kind}")
        return self.fixed(fmt)

    def _skip(self, inner, count):
        if inner == _STRING:
            for _ in range(count):
                self.f.seek(self.fixed("<Q"), os.SEEK_CUR)
        elif inner == _ARRAY:
            for _ in range(count):
                self.value(_ARRAY)
        else:
            self.f.seek(struct.calcsize(_FIXED[inner]) * count, os.SEEK_CUR)


def read_file(path):
    """One file. A tensor's size comes from where the next one starts, so a new tensor type needs no table of sizes."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        reader = _Reader(f)
        version = reader.fixed("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} is too old")
        n_tensors = reader.fixed("<Q")
        n_metadata = reader.fixed("<Q")
        metadata = {}
        for _ in range(n_metadata):
            key = reader.string()
            metadata[key] = reader.value(reader.fixed("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = reader.string()
            dims = [reader.fixed("<Q") for _ in range(reader.fixed("<I"))]
            kind = reader.fixed("<I")
            tensors.append({"name": name, "dims": dims, "type": type_name(kind), "offset": reader.fixed("<Q")})
        alignment = metadata.get("general.alignment", 32)
        start = (f.tell() + alignment - 1) // alignment * alignment
        size = os.fstat(f.fileno()).st_size
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for i, tensor in enumerate(ordered):
        end = ordered[i + 1]["offset"] if i + 1 < len(ordered) else size - start
        tensor["bytes"] = max(end - tensor["offset"], 0)
    return {"version": version, "metadata": metadata, "tensors": tensors, "file_bytes": size}


def parts(path):
    """The files a model is in: the one given, then the other files of a split one that exist."""
    match = _SPLIT_NAME.search(str(path))
    if not match or int(match.group(2)) <= 1:
        return [str(path)]
    others = (_SPLIT_NAME.sub(f"-{i:05d}-of-{match.group(2)}.gguf", str(path)) for i in range(1, int(match.group(2)) + 1))
    return [str(path)] + [part for part in others if part != str(path) and os.path.exists(part)]


def read(path):
    """A model, with all the files of a split one: its first file has the metadata, the others tensors."""
    first = read_file(path)
    for part in parts(path)[1:]:
        other = read_file(part)
        first["tensors"] += other["tensors"]
        first["file_bytes"] += other["file_bytes"]
    return first
