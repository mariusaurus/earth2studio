# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from collections import OrderedDict
from importlib.metadata import version

import numpy as np
import pytest
import torch
import zarr

try:
    zarr_version = version("zarr")
    zarr_major_version = int(zarr_version.split(".")[0])
except Exception:
    zarr_major_version = 2

from earth2studio.io import ZarrBackend
from earth2studio.utils.coords import convert_multidim_to_singledim, split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zarr_field(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    chunks = OrderedDict({"time": 1, "variable": 1, "lat": 180, "lon": 180})

    # Test Memory Store
    z = ZarrBackend(chunks=chunks)
    assert isinstance(z.store, zarr.storage.MemoryStore)
    assert isinstance(z.root, zarr.Group)

    # Instantiate
    array_name = "fields"
    z.add_array(total_coords, array_name)

    # Check instantiation
    for dim in total_coords:
        assert dim in z
        assert dim in z.coords
        assert z[dim].shape == total_coords[dim].shape

    # Test __contains__
    assert array_name in z

    # Test __getitem__
    shape = tuple([len(dim) for dim in total_coords.values()])
    assert z[array_name].shape == shape

    # Test __len__
    assert len(z) == 5

    # Test __iter__
    for array in z:
        assert array in ["fields", "time", "variable", "lat", "lon"]

    # Test add_array with torch.Tensor
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test add_array with kwarg (overwrite)
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
        overwrite=True,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test add_array with list and kwarg (overwrite)
    z.add_array(
        total_coords,
        "dummy_1",
        data=[torch.randn(shape, device=device, dtype=torch.float32)],
        overwrite=True,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    z.add_array(
        total_coords,
        ["dummy_1"],
        data=[torch.randn(shape, device=device, dtype=torch.float32)],
        overwrite=True,
        fill_value=None,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z.write(x, total_coords, "fields_1")
    assert "fields_1" in z
    assert z["fields_1"].shape == x.shape

    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)
    z.write(partial_data, partial_coords, array_name)
    assert np.allclose(z[array_name][0, 0, :, :180], partial_data.to("cpu").numpy())

    xx, _ = z.read(partial_coords, array_name, device=device)
    assert torch.allclose(partial_data, xx)

    # Test Directory Store
    with tempfile.TemporaryDirectory() as td:
        file_name = os.path.join(td, "temp_zarr.zarr")

        z = ZarrBackend(file_name=file_name)
        assert os.path.exists(file_name)
        if zarr_major_version >= 3:
            assert isinstance(z.store, zarr.storage.LocalStore)
        else:
            assert isinstance(z.store, zarr.storage.DirectoryStore)
        assert isinstance(z.root, zarr.Group)

        # Check instantiation
        z.add_array(total_coords, array_name)

        # Check instatiation
        for dim in total_coords:
            assert dim in z
            assert dim in z.coords
            assert z[dim].shape == total_coords[dim].shape

        assert array_name in z
        assert z[array_name].shape == tuple([len(dim) for dim in total_coords.values()])

        # Test writing
        partial_coords = OrderedDict(
            {
                "time": np.asarray(time)[:1],
                "variable": np.asarray(variable)[:1],
                "lat": total_coords["lat"],
                "lon": total_coords["lon"][:180],
            }
        )
        partial_data = torch.randn((1, 1, 180, 180), device=device)
        z.write(partial_data, partial_coords, array_name)
        assert np.allclose(z[array_name][0, 0, :, :180], partial_data.to("cpu").numpy())


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zarr_variable(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Remove var names
    coords = total_coords.copy()
    var_names = coords.pop("variable")

    chunks = OrderedDict({"time": 1, "lat": 180, "lon": 180})

    # Test Memory Store
    z = ZarrBackend(chunks=chunks)
    assert isinstance(z.store, zarr.storage.MemoryStore)
    assert isinstance(z.root, zarr.Group)

    z.add_array(coords, var_names)
    # Check instantiation
    for dim in coords:
        assert z.coords[dim].shape == coords[dim].shape
        assert z[dim].shape == coords[dim].shape

    for var_name in var_names:
        assert var_name in z
        assert z[var_name].shape == tuple([len(values) for values in coords.values()])

    # Test writing
    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)
    z.write(*split_coords(partial_data, partial_coords, "variable"))
    assert np.allclose(z[variable[0]][0, :, :180], partial_data.to("cpu").numpy())

    # Test Directory Store
    with tempfile.TemporaryDirectory() as td:
        file_name = os.path.join(td, "temp_zarr.zarr")
        z = ZarrBackend(file_name=file_name, chunks=chunks)
        assert os.path.exists(file_name)
        if zarr_major_version >= 3:
            assert isinstance(z.store, zarr.storage.LocalStore)
        else:
            assert isinstance(z.store, zarr.storage.DirectoryStore)
        assert isinstance(z.root, zarr.Group)

        z.add_array(coords, var_names)
        # Check instantiation
        for dim in coords:
            assert z.coords[dim].shape == coords[dim].shape
            assert z[dim].shape == coords[dim].shape

        for var_name in var_names:
            assert var_name in z
            assert z[var_name].shape == tuple(
                [len(values) for values in coords.values()]
            )

        # Test writing
        partial_coords = OrderedDict(
            {
                "time": np.asarray(time)[:1],
                "variable": np.asarray(variable)[:1],
                "lat": total_coords["lat"],
                "lon": total_coords["lon"][:180],
            }
        )
        partial_data = torch.randn((1, 1, 180, 180), device=device)
        z.write(*split_coords(partial_data, partial_coords, "variable"))
        assert np.allclose(z[variable[0]][0, :, :180], partial_data.to("cpu").numpy())


@pytest.mark.parametrize(
    "overwrite",
    [True, False],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])  #
def test_zarr_file(overwrite: bool, device: str, tmp_path: str) -> None:
    time = [np.datetime64("1958-01-31T00:00:00")]
    variable = ["t2m", "tcwv"]
    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test File Store
    z = ZarrBackend(tmp_path / "test.zarr", backend_kwargs={"overwrite": overwrite})

    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    z.add_array(total_coords, array_name, data=dummy)

    # Check to see if write overwrite in add array works
    if overwrite:
        z.add_array(total_coords, array_name, data=dummy, overwrite=True)
    else:
        with pytest.raises(RuntimeError):
            z.add_array(total_coords, array_name, data=dummy)

    z = ZarrBackend(tmp_path / "test.zarr", backend_kwargs={"overwrite": overwrite})
    # Check to see if write overwrite in constructor allows redefintion Zarr
    if overwrite:
        z.add_array(total_coords, array_name, data=dummy)
    else:
        with pytest.raises(RuntimeError):
            z.add_array(total_coords, array_name, data=dummy)


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zarr_exceptions(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    chunks = OrderedDict({"time": 1, "variable": 1, "lat": 180, "lon": 180})

    # Test Memory Store
    z = ZarrBackend(chunks=chunks)
    assert isinstance(z.store, zarr.storage.MemoryStore)
    assert isinstance(z.root, zarr.Group)

    # Test mismatch between len(array_names) and len(data)
    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        z.add_array(total_coords, array_name, data=[dummy] * 2)

    # Test trying to add the same array twice.
    z.add_array(
        total_coords,
        ["dummy_1"],
        data=[dummy],
        overwrite=False,
    )
    with pytest.raises(RuntimeError):
        z.add_array(
            total_coords,
            ["dummy_1"],
            data=[dummy],
            overwrite=False,
        )

    # Try to write with bad coords
    bad_coords = {"ensemble": np.arange(0)} | total_coords
    bad_shape = (1,) + shape
    dummy = torch.randn(bad_shape, device=device, dtype=torch.float32)
    with pytest.raises(AssertionError):
        z.write(dummy, bad_coords, "dummy_1")

    # Try to write with too many array names
    with pytest.raises(ValueError):
        z.write([dummy, dummy], bad_coords, "dummy_1")


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zarr_field_multidim(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(0, 360, 360, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": LAT,
            "lon": LON,
        }
    )

    adjusted_coords, _ = convert_multidim_to_singledim(total_coords)
    chunks = OrderedDict({"time": 1, "variable": 1})

    # Test Memory Store
    z = ZarrBackend(chunks=chunks)
    assert isinstance(z.store, zarr.storage.MemoryStore)
    assert isinstance(z.root, zarr.Group)

    # Instantiate
    array_name = "fields"
    z.add_array(total_coords, array_name)

    # Check instantiation
    for dim in adjusted_coords:
        assert dim in z
        assert dim in z.coords
        assert z[dim].shape == adjusted_coords[dim].shape

    # Test __contains__
    assert array_name in z

    # Test __getitem__
    shape = tuple([len(dim) for dim in adjusted_coords.values()])
    assert z[array_name].shape == shape

    # Test __len__
    assert len(z) == 7

    # Test __iter__
    for array in z:
        assert array in ["fields", "time", "variable", "lat", "lon", "ilat", "ilon"]

    # Test add_array with torch.Tensor
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test add_array with kwarg (overwrite)
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
        overwrite=True,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test add_array with list and kwarg (overwrite)
    z.add_array(
        total_coords,
        "dummy_1",
        data=[torch.randn(shape, device=device, dtype=torch.float32)],
        overwrite=True,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    z.add_array(
        total_coords,
        ["dummy_1"],
        data=[torch.randn(shape, device=device, dtype=torch.float32)],
        overwrite=True,
        fill_value=None,
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z.write(x, adjusted_coords, array_name)

    xx, _ = z.read(adjusted_coords, array_name, device=device)
    assert torch.allclose(x, xx)

    # Test separate write
    z.write(x, total_coords, "fields_1")
    assert "fields_1" in z
    assert z["fields_1"].shape == x.shape

    xx, _ = z.read(total_coords, "fields_1", device=device)
    assert torch.allclose(x, xx)
