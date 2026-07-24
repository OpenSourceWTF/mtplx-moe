"""Range-fetch + parse GGUF (v3) split headers from HF, locate tensor bytes.

We never download whole files. We fetch the header of each split (small,
grows only until fully parsed), read tensor infos, then range-fetch just the
byte span of the specific expert-slice we sample.

GGUF v3 binary layout (little-endian):
  magic u32 'GGUF' | version u32 | tensor_count u64 | kv_count u64
  kv_count x {  key:gstr | vtype:u32 | value(vtype)  }
  tensor_count x { name:gstr | n_dims:u32 | dims[n_dims]:u64 | type:u32 | offset:u64 }
  pad to alignment (general.alignment or 32)
  tensor data (offset is relative to this data section start)
gstr = { len:u64 | bytes[len] }
"""
from __future__ import annotations

import struct
import hashlib
from dataclasses import dataclass

import requests
from huggingface_hub import hf_hub_url

GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian

# gguf value types
(GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR,
 GT_ARR, GT_U64, GT_I64, GT_F64) = range(13)

_SCALAR = {
    GT_U8: ("<B", 1), GT_I8: ("<b", 1), GT_U16: ("<H", 2), GT_I16: ("<h", 2),
    GT_U32: ("<I", 4), GT_I32: ("<i", 4), GT_F32: ("<f", 4), GT_BOOL: ("<B", 1),
    GT_U64: ("<Q", 8), GT_I64: ("<q", 8), GT_F64: ("<d", 8),
}


class NeedMore(Exception):
    def __init__(self, need: int):
        self.need = need


class _Cur:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.buf):
            raise NeedMore(end)
        b = self.buf[self.pos:end]
        self.pos = end
        return b

    def scalar(self, t: int):
        fmt, n = _SCALAR[t]
        v = struct.unpack(fmt, self.take(n))[0]
        if t == GT_BOOL:
            return bool(v)
        return v

    def gstr(self) -> str:
        (ln,) = struct.unpack("<Q", self.take(8))
        return self.take(ln).decode("utf-8", "replace")


def _read_value(c: _Cur, vtype: int):
    if vtype in _SCALAR:
        return c.scalar(vtype)
    if vtype == GT_STR:
        return c.gstr()
    if vtype == GT_ARR:
        (elt,) = struct.unpack("<I", c.take(4))
        (count,) = struct.unpack("<Q", c.take(8))
        # Skip bulky arrays cheaply (tokenizer vocab) but still advance cursor.
        if elt == GT_STR:
            out = []
            for _ in range(count):
                out.append(c.gstr())
            return out
        fmt, n = _SCALAR[elt]
        raw = c.take(n * count)
        return list(struct.unpack("<" + fmt[1] * count, raw))
    raise ValueError(f"bad gguf value type {vtype}")


@dataclass
class TensorInfo:
    name: str
    dims: tuple  # ne[], ggml order (ne0 contiguous)
    ggml_type: int
    rel_offset: int  # relative to data section


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    kv: dict
    tensors: dict  # name -> TensorInfo
    data_offset: int  # absolute file offset of tensor data section
    alignment: int


def _parse_header(buf: bytes) -> GGUFHeader:
    c = _Cur(buf)
    magic, version, tcount, kvcount = struct.unpack("<IIQQ", c.take(24))
    if magic != GGUF_MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    kv = {}
    for _ in range(kvcount):
        key = c.gstr()
        (vtype,) = struct.unpack("<I", c.take(4))
        kv[key] = _read_value(c, vtype)
    tensors = {}
    for _ in range(tcount):
        name = c.gstr()
        (nd,) = struct.unpack("<I", c.take(4))
        dims = struct.unpack("<" + "Q" * nd, c.take(8 * nd))
        (gtype,) = struct.unpack("<I", c.take(4))
        (off,) = struct.unpack("<Q", c.take(8))
        tensors[name] = TensorInfo(name, dims, gtype, off)
    align = kv.get("general.alignment", 32)
    pad = (-c.pos) % align
    data_offset = c.pos + pad
    return GGUFHeader(version, tcount, kv, tensors, data_offset, align)


def _hf_url(repo: str, path: str) -> str:
    return hf_hub_url(repo_id=repo, filename=path)


def _range_get(url: str, start: int, end_inclusive: int, session: requests.Session) -> bytes:
    h = {"Range": f"bytes={start}-{end_inclusive}", "User-Agent": "iq-transcode-pilot"}
    r = session.get(url, headers=h, timeout=120)
    r.raise_for_status()
    return r.content


class SplitFile:
    """One split GGUF file, header parsed lazily via range fetch."""

    def __init__(self, repo: str, path: str, session: requests.Session):
        self.repo = repo
        self.path = path
        self.url = _hf_url(repo, path)
        self.session = session
        self.header: GGUFHeader | None = None
        self.header_bytes = 0

    def load_header(self, initial: int = 1 << 20, cap: int = 48 << 20) -> GGUFHeader:
        want = initial
        buf = b""
        while True:
            if len(buf) < want:
                buf += _range_get(self.url, len(buf), want - 1, self.session)
            try:
                self.header = _parse_header(buf)
                self.header_bytes = len(buf)
                return self.header
            except NeedMore as nm:
                want = max(nm.need, want * 2)
                if want > cap:
                    raise RuntimeError(f"header exceeds cap {cap} for {self.path}")

    def fetch_tensor(self, name: str, session: requests.Session | None = None):
        """Fetch full tensor bytes; returns (bytes, TensorInfo, sha256)."""
        assert self.header is not None
        ti = self.header.tensors[name]
        nbytes = _tensor_nbytes(ti)
        start = self.header.data_offset + ti.rel_offset
        raw = _range_get(self.url, start, start + nbytes - 1, session or self.session)
        return raw, ti, hashlib.sha256(raw).hexdigest()

    def fetch_expert_slice(self, name: str, expert: int, n_expert: int,
                           session: requests.Session | None = None):
        """Fetch just one expert's contiguous slice of a merged 3D exps tensor.

        Merged expert tensor dims (ggml order) = (ne0, ne1, n_expert). Expert
        E occupies a contiguous byte span: E * bytes_per_expert.
        Returns (bytes, TensorInfo, byte_start, byte_len, sha256).
        """
        assert self.header is not None
        ti = self.header.tensors[name]
        assert ti.dims[-1] == n_expert, (ti.dims, n_expert)
        total = _tensor_nbytes(ti)
        assert total % n_expert == 0, (total, n_expert)
        per = total // n_expert
        start = self.header.data_offset + ti.rel_offset + expert * per
        raw = _range_get(self.url, start, start + per - 1, session or self.session)
        return raw, ti, start, per, hashlib.sha256(raw).hexdigest()


# ggml block sizes (elems, bytes) for the types we touch
from gguf import GGML_QUANT_SIZES  # noqa: E402
from gguf.constants import GGMLQuantizationType as GQ  # noqa: E402


def _tensor_nbytes(ti: TensorInfo) -> int:
    nelem = 1
    for d in ti.dims:
        nelem *= d
    blk_elems, blk_bytes = GGML_QUANT_SIZES[GQ(ti.ggml_type)]
    assert nelem % blk_elems == 0, (nelem, blk_elems)
    return (nelem // blk_elems) * blk_bytes


def new_session() -> requests.Session:
    return requests.Session()
