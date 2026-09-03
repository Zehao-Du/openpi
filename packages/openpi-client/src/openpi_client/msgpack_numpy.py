"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools
import io

import msgpack
import numpy as np
from PIL import Image


_JPEG_ARRAY_KEY = b"__openpi_jpeg_array__"


def encode_jpeg_images(obj, quality: int):
    """Replace RGB uint8 arrays with JPEG payloads while preserving their dimensions."""
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")
    if isinstance(obj, np.ndarray) and obj.dtype == np.uint8 and obj.ndim == 3 and obj.shape[-1] == 3:
        buffer = io.BytesIO()
        Image.fromarray(obj, mode="RGB").save(buffer, format="JPEG", quality=quality, subsampling=0)
        return {
            _JPEG_ARRAY_KEY: True,
            b"data": buffer.getvalue(),
        }
    if isinstance(obj, dict):
        return {key: encode_jpeg_images(value, quality) for key, value in obj.items()}
    if isinstance(obj, list):
        return [encode_jpeg_images(value, quality) for value in obj]
    if isinstance(obj, tuple):
        return tuple(encode_jpeg_images(value, quality) for value in obj)
    return obj


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if _JPEG_ARRAY_KEY in obj:
        with Image.open(io.BytesIO(obj[b"data"])) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
