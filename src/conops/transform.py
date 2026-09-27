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


class FixedQuantizeMeta(msgspec.Struct, frozen=True):
    source_dtype: str
    quantized_dtype: str
    shape: tuple[int, ...]
    step: float


class DitheredQuantizeMeta(msgspec.Struct, frozen=True):
    source_dtype: str
    quantized_dtype: str
    shape: tuple[int, ...]
    step: float
    seed: int


class SignMeta(msgspec.Struct, frozen=True):
    source_dtype: str
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


class UniformQuantize:
    """
    Dynamic symmetric uniform quantization transform.

    The input state is quantized element-wise to a signed integer type using
    a scale determined from the maximum absolute value of the current state.
    The quantization levels are uniformly spaced and symmetric around zero.

    For a quantization range [-q_max, q_max], the scale is defined as

        scale = max(abs(state)) / q_max,

    and the state is quantized as

        q = clip(round(state / scale), -q_max, q_max).

    During decoding, the quantized state is dequantized by multiplying it by
    the transmitted scale and converted back to the original data type.

    Parameters
    ----------
    quantized_dtype : str, optional
        Signed integer data type used for quantization.
        Defaults to ``"int8"``.

    Raises
    ------
    TypeError
        If ``quantized_dtype`` is not a signed integer data type.
    """

    def __init__(self, quantized_dtype: str = "int8") -> None:
        dtype = np.dtype(quantized_dtype)

        if not np.issubdtype(dtype, np.signedinteger):
            raise TypeError("quantized_dtype must be a signed integer dtype")

        info = np.iinfo(dtype)

        self.quantized_dtype = dtype
        self._qmax = min(abs(info.min), info.max)

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        source_dtype = state.dtype.str

        if state.size == 0:
            scale = 1.0
        else:
            max_abs = float(np.max(np.abs(state)))
            scale = max_abs / self._qmax if max_abs > 0 else 1.0

        scaled_state = state / scale
        rounded_state = np.round(scaled_state)
        clipped_state = np.clip(rounded_state, -self._qmax, self._qmax)

        quantized_state = clipped_state.astype(self.quantized_dtype, copy=False)

        meta = QuantizeMeta(
            source_dtype=source_dtype,
            quantized_dtype=self.quantized_dtype.str,
            shape=quantized_state.shape,
            scale=scale,
        )

        meta_bytes = msgspec.msgpack.encode(meta)

        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=QuantizeMeta)

        dtype = np.dtype(meta.quantized_dtype)
        source_dtype = np.dtype(meta.source_dtype)

        quantized_state = np.frombuffer(payload, dtype=dtype).reshape(meta.shape)

        return (quantized_state * meta.scale).astype(source_dtype, copy=False)


class FixedUniformQuantize:
    """
    Fixed-step symmetric uniform quantization transform.

    Each state element is quantized using a fixed quantization step.
    In the absence of saturation, the element-wise quantization error
    is bounded by half of the quantization step.

    Parameters
    ----------
    step : float
        Quantization step size.

    quantized_dtype : str, optional
        Signed integer data type used for quantization.
        Defaults to ``"int16"``.
    """

    def __init__(self, step: float, quantized_dtype: str = "int16") -> None:
        if step <= 0:
            raise ValueError("step must be positive")

        dtype = np.dtype(quantized_dtype)

        if not np.issubdtype(dtype, np.signedinteger):
            raise TypeError("quantized_dtype must be a signed integer dtype")

        info = np.iinfo(dtype)

        self.step = step
        self.quantized_dtype = dtype
        self._qmax = min(abs(info.min), info.max)

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        scaled_state = state / self.step
        rounded_state = np.round(scaled_state)

        if np.any(np.abs(rounded_state) > self._qmax):
            raise OverflowError("state exceeds the representable quantization range")

        quantized_state = rounded_state.astype(self.quantized_dtype, copy=False)

        meta = FixedQuantizeMeta(
            source_dtype=state.dtype.str,
            quantized_dtype=self.quantized_dtype.str,
            shape=state.shape,
            step=self.step,
        )

        meta_bytes = msgspec.msgpack.encode(meta)

        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=FixedQuantizeMeta)

        dtype = np.dtype(meta.quantized_dtype)
        source_dtype = np.dtype(meta.source_dtype)

        quantized_state = np.frombuffer(payload, dtype=dtype).reshape(meta.shape)

        return (quantized_state * meta.step).astype(source_dtype, copy=False)


class StochasticQuantize:
    """
    Dynamic symmetric uniform quantization with stochastic rounding.

    The quantization scale is determined from the maximum absolute value
    of the current state. Each scaled value is randomly rounded to one
    of its two neighboring integer levels such that the reconstructed
    value is unbiased in expectation.

    Parameters
    ----------
    quantized_dtype : str, optional
        Signed integer data type used for quantization.
        Defaults to ``"int8"``.
    """

    def __init__(self, quantized_dtype: str = "int8") -> None:
        dtype = np.dtype(quantized_dtype)

        if not np.issubdtype(dtype, np.signedinteger):
            raise TypeError("quantized_dtype must be a signed integer dtype")

        info = np.iinfo(dtype)

        self.quantized_dtype = dtype
        self._qmax = min(abs(info.min), info.max)

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        if state.size == 0:
            scale = 1.0
        else:
            max_abs = float(np.max(np.abs(state)))
            scale = max_abs / self._qmax if max_abs > 0 else 1.0

        scaled_state = state / scale

        lower = np.floor(scaled_state)
        probability = scaled_state - lower

        random_values = npr.random(state.shape)

        quantized_state = lower + (random_values < probability)
        quantized_state = np.clip(quantized_state, -self._qmax, self._qmax)
        quantized_state = quantized_state.astype(self.quantized_dtype, copy=False)

        meta = QuantizeMeta(
            source_dtype=state.dtype.str,
            quantized_dtype=self.quantized_dtype.str,
            shape=state.shape,
            scale=scale,
        )

        meta_bytes = msgspec.msgpack.encode(meta)

        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=QuantizeMeta)

        dtype = np.dtype(meta.quantized_dtype)
        source_dtype = np.dtype(meta.source_dtype)

        quantized_state = np.frombuffer(payload, dtype=dtype).reshape(meta.shape)

        return (quantized_state * meta.scale).astype(source_dtype, copy=False)


class SignQuantize:
    """
    Sign-based quantization transform.

    Each state element is represented by its sign, while a shared scale
    is computed as the mean absolute value of the state. The decoded
    state is reconstructed as

        scale * sign(state).

    The encoded payload uses ``int8`` values in ``{-1, 1}``.

    Notes
    -----
    Although this is a sign quantizer, the current implementation stores
    each sign as an ``int8`` value. It therefore uses one byte per element
    rather than packing signs into individual bits.
    """

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        if state.size == 0:
            scale = 0.0
        else:
            scale = float(np.mean(np.abs(state)))

        quantized_state = np.where(state >= 0, 1, -1).astype(np.int8, copy=False)

        meta = SignMeta(source_dtype=state.dtype.str, shape=state.shape, scale=scale)

        meta_bytes = msgspec.msgpack.encode(meta)

        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=SignMeta)

        source_dtype = np.dtype(meta.source_dtype)

        quantized_state = np.frombuffer(payload, dtype=np.int8).reshape(meta.shape)

        return (quantized_state * meta.scale).astype(source_dtype, copy=False)


class DitheredQuantize:
    """
    Fixed-step uniform quantization with subtractive dithering.

    Before quantization, independent uniform dither is added to each
    state element. The same dither is regenerated during decoding from
    a transmitted random seed and subtracted from the reconstructed
    state.

    In the absence of saturation, the element-wise reconstruction error
    is bounded by half of the quantization step.

    Parameters
    ----------
    step : float
        Quantization step size.

    quantized_dtype : str, optional
        Signed integer data type used for quantization.
        Defaults to ``"int16"``.

    Raises
    ------
    ValueError
        If ``step`` is not positive.

    TypeError
        If ``quantized_dtype`` is not a signed integer data type.
    """

    def __init__(self, step: float, quantized_dtype: str = "int16") -> None:
        if step <= 0:
            raise ValueError("step must be positive")

        dtype = np.dtype(quantized_dtype)

        if not np.issubdtype(dtype, np.signedinteger):
            raise TypeError("quantized_dtype must be a signed integer dtype")

        info = np.iinfo(dtype)

        self.step = step
        self.quantized_dtype = dtype
        self._qmax = min(abs(info.min), info.max)

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        seed = int(npr.randint(0, np.iinfo(np.int32).max))

        rng = np.random.default_rng(seed)

        dither = rng.uniform(-0.5 * self.step, 0.5 * self.step, size=state.shape)

        scaled_state = (state + dither) / self.step
        rounded_state = np.round(scaled_state)

        if np.any(np.abs(rounded_state) > self._qmax):
            raise OverflowError("state exceeds the representable quantization range")

        quantized_state = rounded_state.astype(self.quantized_dtype, copy=False)

        meta = DitheredQuantizeMeta(
            source_dtype=state.dtype.str,
            quantized_dtype=self.quantized_dtype.str,
            shape=state.shape,
            step=self.step,
            seed=seed,
        )

        meta_bytes = msgspec.msgpack.encode(meta)

        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=DitheredQuantizeMeta)

        dtype = np.dtype(meta.quantized_dtype)
        source_dtype = np.dtype(meta.source_dtype)

        quantized_state = np.frombuffer(payload, dtype=dtype).reshape(meta.shape)

        rng = np.random.default_rng(meta.seed)

        dither = rng.uniform(-0.5 * meta.step, 0.5 * meta.step, size=meta.shape)

        reconstructed_state = quantized_state * meta.step - dither

        return reconstructed_state.astype(source_dtype, copy=False)


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
