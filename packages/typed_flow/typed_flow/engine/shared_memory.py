"""Shared memory transport for process-pool numpy arrays.

Replaces pickle serialization of large numpy arrays with shared memory
descriptors when returning results from ProcessPoolExecutor child processes.
A ~4.9 MB float32 tensor (640x640x3) serializes as a ~100 byte descriptor
instead of being pickled, reducing cross-process transfer time by ~90%.

Lifecycle:
1. Child process: allocate SharedMemory, copy array data, return descriptor
2. Coordinator: receive descriptor (tiny pickle), copy array from SHM, unlink
3. On error: coordinator cleans up any leaked SHM blocks
"""

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class SharedArrayDescriptor:
    """Lightweight picklable handle replacing a numpy array in cross-process transport."""

    shm_name: str
    shape: tuple[int, ...]
    dtype_str: str


def _array_to_shm(arr: npt.NDArray[np.floating]) -> tuple[SharedArrayDescriptor, SharedMemory]:
    shm = SharedMemory(create=True, size=arr.nbytes)
    shared_view: npt.NDArray[np.floating] = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shared_view, arr)
    return SharedArrayDescriptor(shm_name=shm.name, shape=arr.shape, dtype_str=str(arr.dtype)), shm


def _shm_to_array(desc: SharedArrayDescriptor) -> npt.NDArray[np.floating]:
    shm = SharedMemory(name=desc.shm_name, create=False)
    arr: npt.NDArray[np.floating] = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype_str), buffer=shm.buf).copy()
    shm.close()
    shm.unlink()
    return arr


def pack_arrays_to_shm(obj: object) -> tuple[object, list[SharedMemory]]:
    """Replace numpy array fields in a Pydantic BaseModel with SharedMemory descriptors.

    Returns the packed object and a list of SharedMemory handles that must
    remain open until the coordinator reads the data.
    Non-BaseModel objects and models without arrays pass through unchanged.
    """
    if not isinstance(obj, BaseModel):
        return obj, []

    handles: list[SharedMemory] = []
    updates: dict[str, object] = {}

    for field_name in type(obj).model_fields:
        value = getattr(obj, field_name)
        if isinstance(value, np.ndarray):
            desc, shm = _array_to_shm(value)
            updates[field_name] = desc
            handles.append(shm)
        elif isinstance(value, dict):
            new_dict: dict[str, object] = {}
            changed = False
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    desc, shm = _array_to_shm(v)
                    new_dict[k] = desc
                    handles.append(shm)
                    changed = True
                else:
                    new_dict[k] = v
            if changed:
                updates[field_name] = new_dict

    if not updates:
        return obj, []

    all_fields = {name: getattr(obj, name) for name in type(obj).model_fields}
    all_fields.update(updates)
    packed = type(obj).model_construct(**all_fields)
    return packed, handles


def unpack_arrays_from_shm(obj: object) -> object:
    """Reconstruct numpy arrays from SharedArrayDescriptor fields.

    Opens each shared memory block, copies data to a new array,
    then closes and unlinks the shared memory.
    Non-BaseModel objects and models without descriptors pass through unchanged.
    """
    if not isinstance(obj, BaseModel):
        return obj

    updates: dict[str, object] = {}

    for field_name in type(obj).model_fields:
        value = getattr(obj, field_name)
        if isinstance(value, SharedArrayDescriptor):
            updates[field_name] = _shm_to_array(value)
        elif isinstance(value, dict):
            new_dict: dict[str, object] = {}
            changed = False
            for k, v in value.items():
                if isinstance(v, SharedArrayDescriptor):
                    new_dict[k] = _shm_to_array(v)
                    changed = True
                else:
                    new_dict[k] = v
            if changed:
                updates[field_name] = new_dict

    if not updates:
        return obj

    all_fields = {name: getattr(obj, name) for name in type(obj).model_fields}
    all_fields.update(updates)
    return type(obj).model_construct(**all_fields)


def call_and_pack_shm(processor: object, event: object) -> object | None:
    """Call processor and pack result arrays into shared memory.

    Must be a module-level function (picklable for ProcessPoolExecutor).
    """
    result = processor(event)  # type: ignore[operator]
    if result is None:
        return None
    packed, _handles = pack_arrays_to_shm(result)
    return packed


def cleanup_leaked_shm(shm_names: set[str]) -> None:
    """Best-effort cleanup of shared memory blocks that were not unpacked."""
    for name in shm_names:
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to cleanup shared memory block: {}", name)
