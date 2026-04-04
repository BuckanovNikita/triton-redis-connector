"""Tests for shared memory transport of numpy arrays across process boundaries."""

from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict
from typed_flow.engine.shared_memory import (
    SharedArrayDescriptor,
    call_and_pack_shm,
    pack_arrays_to_shm,
    unpack_arrays_from_shm,
)

_RNG = np.random.default_rng(42)


class SimpleArrayModel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    data: "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"


class DictArrayModel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    outputs: "dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]"


class NoArrayModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: int


class MultiArrayModel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    arr1: "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
    arr2: "np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None" = None


SimpleArrayModel.model_rebuild()
DictArrayModel.model_rebuild()
MultiArrayModel.model_rebuild()


class TestPackUnpackRoundtrip:
    def test_simple_array_roundtrip(self) -> None:
        arr = _RNG.random((3, 640, 640), dtype=np.float32)
        model = SimpleArrayModel(name="test", data=arr)

        packed, handles = pack_arrays_to_shm(model)

        assert len(handles) == 1
        assert isinstance(packed, SimpleArrayModel)
        assert packed.name == "test"
        assert isinstance(packed.data, SharedArrayDescriptor)

        result = unpack_arrays_from_shm(packed)

        assert isinstance(result, SimpleArrayModel)
        assert result.name == "test"
        np.testing.assert_array_equal(result.data, arr)

        for h in handles:
            h.close()

    def test_dict_field_roundtrip(self) -> None:
        arr1 = _RNG.random((1, 84, 8400), dtype=np.float32)
        arr2 = _RNG.random((1, 32, 8400), dtype=np.float32)
        model = DictArrayModel(outputs={"output0": arr1, "output1": arr2})

        packed, handles = pack_arrays_to_shm(model)

        assert len(handles) == 2
        assert isinstance(packed.outputs["output0"], SharedArrayDescriptor)
        assert isinstance(packed.outputs["output1"], SharedArrayDescriptor)

        result = unpack_arrays_from_shm(packed)

        assert isinstance(result, DictArrayModel)
        np.testing.assert_array_equal(result.outputs["output0"], arr1)
        np.testing.assert_array_equal(result.outputs["output1"], arr2)

        for h in handles:
            h.close()

    def test_no_array_passthrough(self) -> None:
        model = NoArrayModel(name="test", value=42)

        packed, handles = pack_arrays_to_shm(model)

        assert packed is model
        assert len(handles) == 0

    def test_none_array_field(self) -> None:
        arr = _RNG.random((3, 640, 640), dtype=np.float32)
        model = MultiArrayModel(arr1=arr, arr2=None)

        packed, handles = pack_arrays_to_shm(model)

        assert len(handles) == 1
        assert isinstance(packed.arr1, SharedArrayDescriptor)
        assert packed.arr2 is None

        result = unpack_arrays_from_shm(packed)

        np.testing.assert_array_equal(result.arr1, arr)
        assert result.arr2 is None

        for h in handles:
            h.close()

    def test_multiple_arrays(self) -> None:
        arr1 = _RNG.random((3, 640, 640), dtype=np.float32)
        arr2 = _RNG.random((1, 84, 8400), dtype=np.float32)
        model = MultiArrayModel(arr1=arr1, arr2=arr2)

        packed, handles = pack_arrays_to_shm(model)

        assert len(handles) == 2

        result = unpack_arrays_from_shm(packed)

        np.testing.assert_array_equal(result.arr1, arr1)
        np.testing.assert_array_equal(result.arr2, arr2)

        for h in handles:
            h.close()

    def test_non_basemodel_passthrough(self) -> None:
        packed, handles = pack_arrays_to_shm("just a string")
        assert packed == "just a string"
        assert len(handles) == 0

        result = unpack_arrays_from_shm(42)
        assert result == 42


class TestShmCleanup:
    def test_unpack_cleans_up_shm(self) -> None:
        arr = _RNG.random((10, 10), dtype=np.float32)
        model = SimpleArrayModel(name="cleanup", data=arr)

        packed, handles = pack_arrays_to_shm(model)

        shm_name = packed.data.shm_name
        unpack_arrays_from_shm(packed)

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=shm_name, create=False)

        for h in handles:
            h.close()


class TestCallAndPackShm:
    def test_call_and_pack(self) -> None:
        arr = _RNG.random((3, 640, 640), dtype=np.float32)

        def processor(event: str) -> SimpleArrayModel:
            return SimpleArrayModel(name=event, data=arr)

        result = call_and_pack_shm(processor, "input_event")

        assert isinstance(result, SimpleArrayModel)
        assert isinstance(result.data, SharedArrayDescriptor)

        unpacked = unpack_arrays_from_shm(result)
        np.testing.assert_array_equal(unpacked.data, arr)

    def test_call_and_pack_none_result(self) -> None:
        def processor(_event: str) -> None:
            return None

        result = call_and_pack_shm(processor, "input")
        assert result is None
