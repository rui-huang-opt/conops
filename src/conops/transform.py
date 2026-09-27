from typing import Protocol

import msgspec
import numpy as np
import numpy.random as npr
from numpy.typing import NDArray


class BasicMeta(msgspec.Struct, frozen=True):
    dtype: str
    shape: tuple[int, ...]


class QuantizeMeta(msgspec.Struct, frozen=True):
    source_dtype: str
    quantized_dtype: str
    shape: tuple[int, ...]
    scale: float


class Transform(Protocol):
    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        """
        Encode the state into a payload with its data type.

        Args:
            state (NDArray): The state to encode.

        Returns:
            tuple[bytes, NDArray]: A tuple containing the meta data and the encoded payload.
        """
        ...

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        """
        Decode the payload back into the original state using the meta data.

        Args:
            meta (bytes): The meta data.

            payload (bytes): The encoded payload.

        Returns:
            NDArray: The decoded state.
        """
        ...


class Identity:
    """
    Identity transform that performs no transformation.
    """

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        meta = BasicMeta(dtype=state.dtype.str, shape=state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)


class Quantize:
    """
    Quantize transform that scales and converts the state to a specified integer type.

    Parameters
    ----------
    quantized_dtype : str
        The target integer data type for quantization. Default is 'int8'.
    """

    def __init__(self, quantized_dtype: str = "int8"):
        self.quantized_dtype = quantized_dtype

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        source_dtype = state.dtype.str
        quantized_dtype = np.dtype(self.quantized_dtype)
        min_val = np.iinfo(quantized_dtype).min
        max_val = np.iinfo(quantized_dtype).max

        max_abs = float(np.max(np.abs(state)))
        scale = max_abs / max_val if max_abs > 0 else 1.0

        scaled_state = state / scale
        rounded_state = np.round(scaled_state)
        clipped_state = np.clip(rounded_state, min_val, max_val)
        quantized_state = clipped_state.astype(quantized_dtype, copy=False)

        meta = QuantizeMeta(
            source_dtype=source_dtype,
            quantized_dtype=quantized_dtype.str,
            shape=quantized_state.shape,
            scale=scale,
        )
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=QuantizeMeta)
        quantized_dtype = np.dtype(meta.quantized_dtype)
        source_dtype = np.dtype(meta.source_dtype)
        quantized_state = np.frombuffer(payload, quantized_dtype).reshape(meta.shape)
        return (quantized_state * meta.scale).astype(source_dtype, copy=False)


class DPMechanism:
    """
    Differential Privacy mechanism that adds Laplace noise to the state.

    Parameters
    ----------
    epsilon : float
        The privacy budget parameter.

    sensitivity : float
        The sensitivity of the function to which noise is added.
    """

    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._scale = sensitivity / epsilon

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        noise = npr.laplace(0, self._scale, size=state.shape)
        noisy_state = state + noise.astype(state.dtype, copy=False)
        dtype = noisy_state.dtype.str
        meta = BasicMeta(dtype=dtype, shape=noisy_state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, noisy_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)


class GaussianNoise:
    """
    Gaussian noise mechanism that adds Gaussian noise to the state.

    Parameters
    ----------
    loc : float
        The mean of the Gaussian noise.

    scale : float
        The standard deviation of the Gaussian noise.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = loc
        self.scale = scale

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        noise = npr.normal(self.loc, self.scale, state.shape)
        noisy_state = state + noise.astype(state.dtype, copy=False)
        dtype = noisy_state.dtype.str
        meta = BasicMeta(dtype=dtype, shape=noisy_state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, noisy_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)
