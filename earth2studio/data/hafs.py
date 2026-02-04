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

import asyncio
import concurrent.futures
import functools
import hashlib
import os
import pathlib
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_forecast_inputs,
)
from earth2studio.lexicon.hafs import HAFSLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

try:
    import pyproj
except ImportError:
    OptionalDependencyFailure("data")
    pyproj = None

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# Silence FutureWarning from cfgrib
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class HAFSAsyncTask:
    """Small helper struct for Async tasks"""

    data_array_indices: tuple[int, int, int]
    hafs_file_uri: str
    hafs_byte_offset: int
    hafs_byte_length: int
    hafs_modifier: Callable


@check_optional_dependencies()
class HAFS:
    """Hurricane Analysis and Forecast System (HAFS) data source provides hourly
    tropical cyclone analysis and forecast data developed by NOAA. This data source
    is provided on a grid similar to HRRR at 1-hour intervals. The spatial dimensionality
    of HAFS data is [1059, 1799].

    The `hafs_x` and `hafs_y` coordinates of the resulting `DataArray` are the native
    coordinates of the HAFS model. The corresponding CRS can be set up with cartopy:

    .. code-block:: python

        import cartopy.crs as ccrs

        proj_hafs = ccrs.LambertConformal(
            central_longitude=262.5,
            central_latitude=38.5,
            standard_parallels=(38.5, 38.5),
            globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
        )

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws'), by default 'aws'
    domain : str, optional
        Domain to use ('parent' or 'storm'), by default 'parent'
        Note: 'storm' domain is not yet implemented
    storm_id : str, optional
        Storm ID for file naming, by default '10l'
    model_id : str, optional
        Model ID for file naming, by default 'hfsa'
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://noaa-nws-hafs-pds.s3.amazonaws.com/index.html#hfsa/
    """

    HAFS_BUCKET_NAME = "noaa-nws-hafs-pds"
    MAX_BYTE_SIZE = 5000000

    @staticmethod
    def _product_to_hafs_type(product: str) -> str:
        """Map HRRR-style product names to HAFS file type identifiers

        Parameters
        ----------
        product : str
            Product name (wrfsfc, wrfprs, wrfnat)

        Returns
        -------
        str
            HAFS type identifier - always "atm" since HAFS only has atm files
        """
        # HAFS only provides atm files, regardless of product type
        return "atm"

    def __init__(
        self,
        domain: str | None = None,
        storm_id: str | None = None,
        model_id: str | None = None,
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers

        # Validate and set domain
        if domain not in ["parent", "storm"]:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'parent' or 'storm'")

        self._domain = domain
        self._storm_id = storm_id
        self._model_id = model_id

        self.lexicon = HAFSLexicon
        self.async_timeout = async_timeout

        self._source = "aws"
        if self._source == "aws":
            self.uri_prefix = "noaa-nws-hafs-pds"

            # HAFS data availability - adjust date range as needed
            def _range(time: datetime) -> None:
                # HAFS data availability may differ from HRRR
                # TODO: Update with actual HAFS data availability dates
                if time < datetime(year=2023, month=6, day=19, hour=18):
                    raise ValueError(
                        f"Requested date {time} needs to be on or after 6th June, 2020, 6pm for HAFS"
                    )

            self._history_range = _range
        else:
            raise ValueError(f"Invalid HAFS source { self._source}")

        try:
            nest_asyncio.apply()  # Monkey patch asyncio to work in notebooks
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            # Else we assume that async calls will be used which in that case
            # we will init the group in the call function when we have the loop
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of fsspec file stores

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        if self._source == "aws":
            self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)
        elif self._source == "nomads":
            # HTTP file system, tried FTP but didnt work
            self.fs = HTTPFileSystem(asynchronous=True)

    def set_storm_model_domain(
        self,
        storm_id: str,
        model_id: str,
        domain: str,
    ) -> None:
        """Set the storm, model, and domain for the HAFS object"""
        self._storm_id = storm_id
        self._model_id = model_id
        self._domain = domain

        if self._domain not in ["parent", "storm"]:
            raise ValueError(
                f"Invalid domain '{self._domain}'. Must be 'parent' or 'storm'"
            )

        if self._model_id not in ["hfsa", "hfsb"]:
            raise ValueError(
                f"Invalid model ID '{self._model_id}'. Must be 'hfsa' or 'hfsb'"
            )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HAFS analysis data (lead time 0)

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HAFS lexicon.

        Returns
        -------
        xr.DataArray
            HAFS weather data array
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Modify the worker amount
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, lead_time, variable), timeout=self.async_timeout
            )
        )

        return xr_array

    async def close(self) -> None:
        """Close the file system session and release resources.

        Call this method when you are completely done using the HAFS data source
        to properly clean up async resources. Alternatively, use the async context
        manager pattern with `async with HAFS(...) as source:`.
        """
        if self.fs is not None:
            if hasattr(self.fs, "_session") and self.fs._session is not None:
                await self.fs._session.close()
            self.fs = None

    async def __aenter__(self) -> "HAFS":
        """Async context manager entry."""
        if self.fs is None:
            await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HAFS lexicon.

        Returns
        -------
        xr.DataArray
            HAFS weather data array
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        # time, variable = prep_data_inputs(time, variable)
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session()
        else:
            session = None

        # Determine grid dimensions by fetching one variable first
        # HAFS grid dimensions need to be read from the actual data
        lat, lon = await self._determine_grid_size(time[0], lead_time[0])

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with
        xr_array = xr.DataArray(
            data=np.zeros(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(lat),
                    len(lon),
                )
            ),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "lat": lat,
                "lon": lon,
            },
        )
        if self._domain == 'storm':
            xr_array.coords["lat_south"] = ("lead_time", np.zeros(len(lead_time)))
            xr_array.coords["lon_west"] = ("lead_time", np.zeros(len(lead_time)))


        async_tasks = []
        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching HAFS data", disable=(not self._verbose)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array

    async def _create_tasks(
        self, time: list[datetime], lead_time: list[timedelta], variable: list[str]
    ) -> list[HAFSAsyncTask]:
        """Create download tasks, each corresponding to one grib byte range

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        variables : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[dict]
            List of download tasks
        """
        tasks: list[HAFSAsyncTask] = []  # group pressure-level variables

        # Start with fetching all index files for each time / lead time
        # TODO: Update so only needed products (can tell from parsing variables), for
        # now fetch all index files because its cheap and easier
        products = ["wrfsfc", "wrfprs", "wrfnat"]
        args = [
            self._grib_index_uri(t, lt, p)
            for t in time
            for lt in lead_time
            for p in products
        ]
        func_map = map(self._fetch_index, args)
        results = await tqdm.gather(
            *func_map, desc="Fetching HAFS index files", disable=True
        )
        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                # Get index file dictionaries for each possible product
                index_files = {p: results.pop(0) for p in products}
                for k, v in enumerate(variable):
                    try:
                        hafs_name_str, modifier = self.lexicon[v]  # type: ignore
                        hafs_name = hafs_name_str.split("::") + ["", ""]
                        product = hafs_name[0]
                        variable_name = hafs_name[1]
                        level = hafs_name[2]
                        forecastvld = hafs_name[3]
                        index = hafs_name[4]

                        # Create index key to find byte range
                        hafs_key = f"{variable_name}::{level}"

                        if variable_name in ["WIND"]:
                            # Convert numpy.timedelta64 to hours if needed
                            if isinstance(lt, np.timedelta64):
                                hours = int(lt / np.timedelta64(1, "h"))
                            else:
                                hours = int(lt.total_seconds() // 3600)
                            if hours == 0:
                                hafs_key = f"{hafs_key}::0-0 day max fcst"
                            else:
                                hafs_key = (
                                    f"{hafs_key}::{hours-1}-{hours} hour max fcst"
                                )
                        elif forecastvld:
                            hafs_key = f"{hafs_key}::{forecastvld}"
                        else:
                            # Convert numpy.timedelta64 to seconds if needed
                            if isinstance(lt, np.timedelta64):
                                lt_seconds = lt / np.timedelta64(1, "s")
                            else:
                                lt_seconds = lt.total_seconds()
                            if lt_seconds == 0:
                                hafs_key = f"{hafs_key}::anl"
                            else:
                                if isinstance(lt, np.timedelta64):
                                    hours = int(lt / np.timedelta64(1, "h"))
                                else:
                                    hours = int(lt.total_seconds() // 3600)
                                hafs_key = f"{hafs_key}::{hours:d} hour fcst"
                        if index and index.isnumeric():
                            hafs_key = index

                        # Special cases
                        # could do this better with templates, but this is single instance
                        if variable_name == "APCP":
                            # Convert numpy.timedelta64 to hours if needed
                            if isinstance(lt, np.timedelta64):
                                hours = int(lt / np.timedelta64(1, "h"))
                            else:
                                hours = int(lt.total_seconds() // 3600)

                            if hours == 0:
                                hafs_key = f"{variable_name}::{level}::0-0 day acc fcst"
                            else:
                                hafs_key = f"{variable_name}::{level}::{hours-3:d}-{hours:d} hour acc fcst"

                    except KeyError as e:
                        logger.error(
                            f"variable id {variable} not found in HAFS lexicon"
                        )
                        raise e

                    # Get byte range from index
                    byte_offset = None
                    byte_length = None
                    for key, value in index_files[product].items():
                        if hafs_key in key:
                            byte_offset = value[0]
                            byte_length = value[1]
                            break

                    if byte_length is None or byte_offset is None:
                        logger.warning(
                            f"Variable {v} not found in index file for time {t} at {lt}, values will be unset"
                        )
                        continue

                    tasks.append(
                        HAFSAsyncTask(
                            data_array_indices=(i, j, k),
                            hafs_file_uri=self._grib_uri(t, lt, product),
                            hafs_byte_offset=byte_offset,
                            hafs_byte_length=byte_length,
                            hafs_modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: HAFSAsyncTask,
        xr_array: xr.DataArray,
    ) -> xr.DataArray:
        """Small wrapper to pack arrays into the DataArray"""
        out, _lat_south, _lon_west = await self.fetch_array(
            task.hafs_file_uri,
            task.hafs_byte_offset,
            task.hafs_byte_length,
            task.hafs_modifier,
        )
        i, j, k = task.data_array_indices
        xr_array[i, j, k] = out

        if self._domain == 'storm':
            xr_array.coords["lat_south"].values[j] = _lat_south
            xr_array.coords["lon_west"].values[j] = _lon_west

    async def fetch_array(
        self,
        grib_uri: str,
        byte_offset: int,
        byte_length: int,
        modifier: Callable,
    ) -> np.ndarray:
        """Fetch HAFS data array. This will first fetch the index file to get byte range
        of the needed data, fetch the respective grib files and lastly combining grib
        files into single data array.

        Parameters
        ----------
        grib_uri : str
            URI to grib file
        byte_offset : int
            Byte offset in file
        byte_length : int
            Byte length to read
        modifier : Callable
            Function to modify data values

        Returns
        -------
        np.ndarray
            Data array for given time and lead time
        """
        logger.debug(f"Fetching HAFS grib file: {grib_uri} {byte_offset}-{byte_length}")
        # Download the grib file to cache
        grib_file = await self._fetch_remote_file(
            grib_uri,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )
        # Open into xarray data-array
        da = xr.open_dataarray(
            grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        # Extract 2D array, handling possible extra dimensions (like level)
        values = da.values
        # Squeeze out singleton dimensions and get 2D spatial data
        while values.ndim > 2:
            values = values.squeeze()
        # Ensure we have a 2D array
        if values.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {values.ndim}D with shape {values.shape}"
            )

        lat_west, lon_south  = None, None
        if self._domain == 'storm':
            lat_west = da['latitude'][0]
            lon_south = da['longitude'][0]

        return modifier(values), lat_west, lon_south

    async def _determine_grid_size(
        self,
        time: datetime,
        lead_time: timedelta,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine HAFS grid dimensions by reading one variable from a GRIB file

        Parameters
        ----------
        time : datetime
            Time to fetch sample data from

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lat/lon coordinate arrays
        """
        # Fetch index file for wrfprs (pressure level) product
        index_uri = self._grib_index_uri(time, lead_time, "wrfprs")
        index_dict = await self._fetch_index(index_uri)

        # Find the first available variable in the index to get grid dimensions
        if not index_dict:
            raise ValueError("No variables found in HAFS index file")

        # Get the first available variable's byte range
        first_key = list(index_dict.keys())[0]
        byte_offset, byte_length = index_dict[first_key]

        # Fetch a small sample to determine grid dimensions
        grib_uri = self._grib_uri(time, lead_time, "wrfprs")
        grib_file = await self._fetch_remote_file(
            grib_uri,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )

        # Open into xarray to get dimensions
        da = xr.open_dataarray(
            grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )

        # Get grid dimensions from the data array
        # Handle possible extra dimensions (like level) by squeezing
        values = da.values
        while values.ndim > 2:
            values = values.squeeze()

        if values.ndim == 2:
            height, width = values.shape
        elif values.ndim == 1:
            # 1D data, try to infer from GRIB metadata
            if "y" in da.dims and "x" in da.dims:
                height = da.sizes["y"]
                width = da.sizes["x"]
            elif "latitude" in da.dims and "longitude" in da.dims:
                height = da.sizes["latitude"]
                width = da.sizes["longitude"]
            else:
                raise ValueError(
                    f"Could not determine grid dimensions from GRIB file. Shape: {values.shape}, dims: {da.dims}"
                )
        else:
            raise ValueError(f"Unexpected data shape: {values.shape}")

        # Extract lat/lon coordinates if available
        lat = None
        lon = None

        # Try to get coordinates from the data array
        if "latitude" in da.coords and "longitude" in da.coords:
            lat = da.coords["latitude"].values
            lon = da.coords["longitude"].values
        elif "lat" in da.coords and "lon" in da.coords:
            print("1" * 85)
            lat = da.coords["lat"].values
            lon = da.coords["lon"].values
            if lat.ndim == 1 and lon.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                lat = lat_2d
                lon = lon_2d
        else:
            print("2" * 85)
            # No coordinates found, create placeholder arrays
            lat = np.zeros((height, width))
            lon = np.zeros((height, width))

        return lat, lon

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for HAFS based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for HAFS"
                )
            # Check history range for given source
            self._history_range(time)

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int]]:
        """Fetch HAFS atmospheric index file

        Parameters
        ----------
        index_uri : str
            URI to grib index file to download

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of HAFS variables (byte offset, byte length)
        """
        # Grab index file
        try:
            index_file = await self._fetch_remote_file(index_uri)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The specified data index, {index_uri}, does not exist. Data seems to be missing."
            )
        with open(index_file) as file:
            index_lines = [line.rstrip() for line in file]

        index_table = {}
        # Note we actually drop the last variable here because its easier (SBT114)
        # GEFS has a solution for this if needed that involves appending a dummy line
        # Example of row: "1:0:d=2021111823:REFC:entire atmosphere:795 min fcst:"
        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[0]}::{lsplit[3]}::{lsplit[4]}::{lsplit[5]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        return index_table

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetches remote file into cache"""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            if self.fs.async_impl:
                if byte_length:
                    byte_length = int(byte_offset + byte_length)
                data = await self.fs._cat_file(path, start=byte_offset, end=byte_length)
            else:
                data = await asyncio.to_thread(
                    self.fs.read_block, path, offset=byte_offset, length=byte_length
                )
            with open(cache_path, "wb") as file:
                await asyncio.to_thread(file.write, data)

        return cache_path

    def _grib_uri(
        self, time: datetime, lead_time: timedelta, product: str = "wrfsfc"
    ) -> str:
        """Generates the URI for HAFS grib files

        HAFS file naming: {storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2
        Example: 10l.2024093000.hfsa.parent.atm.f000.grb2
        """
        # Convert numpy.datetime64 to Python datetime if needed
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        # Convert numpy.timedelta64 to hours if needed
        if isinstance(lead_time, np.timedelta64):
            lead_hour = int(lead_time / np.timedelta64(1, "h"))
        else:
            lead_hour = int(lead_time.total_seconds() // 3600)
        hafs_type = self._product_to_hafs_type(product)
        # HAFS structure: hfsa/YYYYMMDD/HH/{storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2
        file_name = f"hfsa/{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        time_str = f"{time.year}{time.month:0>2}{time.day:0>2}{time.hour:0>2}"
        filename = f"{self._storm_id}.{time_str}.{self._model_id}.{self._domain}.{hafs_type}.f{lead_hour:03d}.grb2"
        file_name = os.path.join(file_name, filename)
        return os.path.join(self.uri_prefix, file_name)

    def _grib_index_uri(
        self, time: datetime, lead_time: timedelta, product: str
    ) -> str:
        """Generates the URI for HAFS index grib files

        HAFS file naming: {storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2.idx
        Example: 10l.2024093000.hfsa.parent.atm.f000.grb2.idx
        """
        # Convert numpy.datetime64 to Python datetime if needed
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        # Convert numpy.timedelta64 to hours if needed
        if isinstance(lead_time, np.timedelta64):
            lead_hour = int(lead_time / np.timedelta64(1, "h"))
        else:
            lead_hour = int(lead_time.total_seconds() // 3600)
        hafs_type = self._product_to_hafs_type(product)
        # HAFS structure: hfsa/YYYYMMDD/HH/{storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2.idx
        file_name = f"hfsa/{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        time_str = f"{time.year}{time.month:0>2}{time.day:0>2}{time.hour:0>2}"
        filename = f"{self._storm_id}.{time_str}.{self._model_id}.{self._domain}.{hafs_type}.f{lead_hour:03d}.grb2.idx"
        file_name = os.path.join(file_name, filename)
        return os.path.join(self.uri_prefix, file_name)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "hafs")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_hafs")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        domain: str = "parent",
        storm_id: str = "10l",
        model_id: str = "hfsa",
    ) -> bool:
        """Checks if given date time is avaliable in the HAFS object store. Uses S3
        store

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access
        domain : str, optional
            Domain to check ('parent' or 'storm'), by default 'parent'
        storm_id : str, optional
            Storm ID for file naming, by default '10l'
        model_id : str, optional
            Model ID for file naming, by default 'hfsa'

        Returns
        -------
        bool
            If date time is avaiable
        """
        if domain not in ["parent", "storm"]:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'parent' or 'storm'")

        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        fs = s3fs.S3FileSystem(anon=True)
        # Object store directory for given time
        # Just picking the first variable to look for (using atm type from wrfnat)
        file_name = f"hfsa/{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        time_str = f"{time.year}{time.month:0>2}{time.day:0>2}{time.hour:0>2}"
        filename = f"{storm_id}.{time_str}.{model_id}.{domain}.atm.f000.grb2.idx"
        file_name = f"{file_name}{filename}"
        s3_uri = f"s3://{cls.HAFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists

    @classmethod
    def list_storm_ids(
        cls,
        time: datetime | np.datetime64,
        domain: str = "parent",
        model_id: str = "hfsa",
        return_file_names: bool = False,
    ) -> list[str] | tuple[list[str], list[str]]:
        """List available storm IDs for a given time from the HAFS object store.

        This function lists all files in the directory for the given time and extracts
        unique storm IDs from the filenames. File naming pattern:
        {storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check for available storm IDs
        domain : str, optional
            Domain to check ('parent' or 'storm'), by default 'parent'
        model_id : str, optional
            Model ID for file naming, by default 'hfsa'
        return_file_names : bool, optional
            If True, also return the list of all file names found, by default False

        Returns
        -------
        list[str] | tuple[list[str], list[str]]
            If return_files is False: list of unique storm IDs (sorted)
            If return_files is True: tuple of (list of unique storm IDs, list of all file names)
        """
        if domain not in ["parent", "storm"]:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'parent' or 'storm'")

        if model_id not in ["hfsa", "hfsb"]:
            raise ValueError(f"Invalid model ID '{model_id}'. Must be 'hfsa' or 'hfsb'")

        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        fs = s3fs.S3FileSystem(anon=True)
        # Object store directory for given time
        directory = f"hfsa/{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        s3_uri = f"s3://{cls.HAFS_BUCKET_NAME}/{directory}"

        # List all files in the directory
        try:
            files = fs.ls(s3_uri, detail=False)
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Error listing files in directory {s3_uri}: {e}")
            if return_file_names:
                return [], []
            return []

        # Extract storm IDs from filenames
        # Pattern: {storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2
        # Using a set to automatically remove duplicate storm IDs
        storm_ids = set()
        file_names = []

        for file_path in files:
            # Extract just the filename from the full path
            filename = os.path.basename(file_path)
            file_names.append(filename)

            # Extract storm_id (part before first dot)
            # Only process files that match the expected pattern
            if "." in filename:
                parts = filename.split(".")
                if (
                    len(parts) >= 5
                ):  # Should have at least: storm_id, timestamp, model_id, domain, type, ...
                    storm_id = parts[0]
                    # Verify it matches expected pattern by checking structure
                    # Expected: {storm_id}.YYYYMMDDHH.{model_id}.{domain}.{type}.f{lead_hour}.grb2
                    if len(parts) >= 4 and parts[2] == model_id and parts[3] == domain:
                        storm_ids.add(storm_id)

        storm_ids_list = sorted(list(storm_ids))

        if return_file_names:
            return storm_ids_list, file_names
        return storm_ids_list
