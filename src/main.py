# Copyright 2017 National Computational Infrastructure(NCI).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import os
from pathlib import Path
import datetime
import cftime
import tempfile
import subprocess
from subprocess import check_call

import osr
import click
import netCDF4
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling

import numpy as np

from fast_medoid import medoid

__fc__ = {"phot_veg": 0, "nphot_veg": 1, "bare_soil": 2}
__spring__ = [8, 9, 10]  # September, October and November
__summer__ = [11, 0, 1]  # December, January and February
__autumn__ = [2, 3, 4]  # March, April and May
__winter__ = [5, 6, 7]  # June, July and August

__composite_map__ = {
    "season": {
        "spring": __spring__,
        "summer": __summer__,
        "autumn": __autumn__,
        "winter": __winter__,
    },
    "month": {
        "January": [0],
        "February": [1],
        "March": [2],
        "April": [3],
        "May": [4],
        "June": [5],
        "July": [6],
        "August": [7],
        "September": [8],
        "October": [9],
        "November": [10],
        "December": [11],
    },
}
__outfile_fmt__ = (
    "FC_{c}_Medoid.v{ver}.MCD43A4.h{h:02d}v{v:02d}.{year}.006.tif"
)
__outfile_mosiac__ = "FC_{c}_Medoid.v{ver}.MCD43A4.{r}.006.tif"
__west__ = [
    (27, 11),
    (27, 12),
    (28, 11),
    (28, 10),
    (28, 12),
    (29, 10),
    (29, 11),
    (29, 12),
]
__WA_BOUNDS__ = ("112.5", "-35.5", "129.5", "-13.45")

__base_url__ = (
    "http://dapds00.nci.org.au/thredds/fileServer/"
    "tc43/modis-fc/v310/tiles/8-day/cover/"
)


def read_gospatial(filename):
    """Read the geo-spatial information from netCDF file."""
    geospatial = {}
    with rasterio.open(filename) as ds:
        for sds_name in ds.subdatasets:
            with rasterio.open(sds_name) as sds:
                band_name = sds_name.split(":")[-1]
                geospatial[band_name] = {
                    "geotransform": sds.transform.to_gdal(),
                    "crs_wkt": sds.crs.wkt,
                }
    return geospatial


def download(
    filename, outdir,
):
    """downloads data from nci thread server"""
    url = f"{__base_url__}{filename}"

    cmd = ["wget", url, "-P", outdir]
    try:
        check_call(cmd)
    except subprocess.CalledProcessError as err:
        raise err

    return Path(outdir).joinpath(filename)


def write_img(
    data, outfile, transform, src_crs, driver="GTiff", nodata=None,
):
    shape = data.shape
    if len(shape) != 2:
        raise ValueError(
            f"shape is {shape}:" " only 2 dimensional array is implemented"
        )

    origin = (transform[0], transform[3])
    pixelsize = (abs(transform[1]), abs(transform[5]))
    if isinstance(src_crs, osr.SpatialReference):
        crs = src_crs
    else:
        crs = osr.SpatialReference()
        if crs == crs.SetFromUserInput(src_crs):
            raise ValueError(f"Invalid crs: {src_crs}")

    transform = Affine(pixelsize[0], 0, origin[0], 0, -pixelsize[1], origin[1])

    # dtype of the array
    dtype = data.dtype.name
    if dtype in ["int64", "int8", "uint64"]:
        raise TypeError(f"Datatype not supported: {dtype}")

    kwargs = {
        "count": 1,
        "width": shape[1],
        "height": shape[0],
        "crs": crs.ExportToWkt(),
        "transform": transform,
        "dtype": data.dtype.name,
        "nodata": nodata,
        "driver": driver,
    }
    with rasterio.open(outfile.as_posix(), "w", **kwargs) as dst:
        dst.write(data, 1)


def pack_data(
    src_filename,
    output_dir,
    arr,
    timestamps,
    composite_type,
    version,
    htile,
    vtile,
    year,
):
    geo_spatial = read_gospatial(src_filename)
    for time_idx, dt in enumerate(timestamps):
        composite_map = __composite_map__[composite_type]
        for composite, composite_months in composite_map.items():
            if dt.month - 1 in composite_months:
                for fc_type, fc_index in __fc__.items():
                    # print(fc_type, fc_index, composite, geo_spatial[fc_type])
                    outfile = __outfile_fmt__.format(
                        c=f"{composite}_{fc_type}",
                        ver=version,
                        h=htile,
                        v=vtile,
                        year=year,
                    )
                    write_img(
                        arr[time_idx, fc_index, :, :].astype("uint8"),
                        output_dir.joinpath(outfile),
                        geo_spatial[fc_type]["geotransform"],
                        geo_spatial[fc_type]["crs_wkt"],
                        nodata=255,
                    )


def compute_medoid(
    filename, output_dir, ver, composite_type, htile, vtile, year
):

    with netCDF4.Dataset(filename) as ds:
        month_idx_tmp = [[] for _ in range(12)]
        season_idx_tmp = [[] for _ in range(4)]

        for its, ts in enumerate(ds["time"]):
            date = netCDF4.num2date(ts, ds["time"].units)
            month_idx_tmp[date.month - 1].append(its)

            # gather data index if data falls in season's month
            for _its, season in enumerate(
                [__spring__, __summer__, __autumn__, __winter__]
            ):
                if date.month - 1 in season:
                    season_idx_tmp[_its].append(its)
        # The following checks that if the current month is complete
        # We only compute medoids if the current month is complete
        for i in range(len(month_idx_tmp)):
            m = month_idx_tmp[i]
            if len(m) == 0:
                continue
            ts = netCDF4.num2date(ds["time"][m[-1]], ds["time"].units)
            next_date = ts + datetime.timedelta(days=8)

            if ts.month == next_date.month:
                if ts.year == 2001 and ts.month == 6:
                    print("Problem here incomplete files but still will do")
                else:
                    month_idx_tmp[i] = []

        month_idx = [m for m in month_idx_tmp if len(m) > 0]
        season_idx = [s for s in season_idx_tmp if len(s) > 0]

        pv = np.asarray(ds["phot_veg"][:], dtype=np.float32)
        npv = np.asarray(ds["nphot_veg"][:], dtype=np.float32)
        soil = np.asarray(ds["bare_soil"][:], dtype=np.float32)

        data = np.empty(
            (pv.shape[0], 3, pv.shape[1], pv.shape[2]), dtype=pv.dtype
        )
        data[:, 0, :, :] = pv
        data[:, 1, :, :] = npv
        data[:, 2, :, :] = soil

        medoid_data_list = []
        medoid_timestamp_list = []
        if composite_type == "month":
            # compute monthly medoid
            for m, m_idx in enumerate(month_idx):
                d = data[m_idx, ...]
                month_medoid = medoid(d)
                medoid_data_list.append(month_medoid)
                ts = cftime.num2pydate(ds["time"][m_idx[0]], ds["time"].units)
                ts = ts.replace(day=1)
                medoid_timestamp_list.append(ts)
        elif composite_type == "season":
            # compute seasonal medoid
            for s, s_idx in enumerate(season_idx):
                sdata = data[s_idx, ...]
                season_medoid = medoid(sdata)
                medoid_data_list.append(season_medoid)
                ts = cftime.num2pydate(ds["time"][s_idx[0]], ds["time"].units)
                ts = ts.replace(day=1)
                medoid_timestamp_list.append(ts)
        else:
            raise NotImplementedError(
                f"Composite type '{composite_type}' not implemented"
            )

        if len(medoid_data_list) == 0:
            print("no data for %s" % filename)
            return

        if len(medoid_data_list) == 1:
            medoid_data = np.expand_dims(medoid_data_list[0], axis=0)
        else:
            medoid_data = np.stack(medoid_data_list)

        pack_data(
            filename,
            output_dir,
            medoid_data,
            medoid_timestamp_list,
            composite_type,
            ver,
            htile,
            vtile,
            year,
        )


def build_vrt_mosaic(
    indir,
    outdir,
    composite_type,
    region,
    year,
    ver,
    overviews_level=5,
    resampling=Resampling.nearest,
):
    indir = Path(indir)
    outdir = Path(outdir)

    if region == "west":
        bounds = __WA_BOUNDS__
    else:
        raise NotImplementedError(f"{region} not implemented")

    composite = __composite_map__[composite_type]
    for comp in composite.keys():
        for cov in __fc__.keys():
            with tempfile.TemporaryDirectory() as tmpdir:
                inputfile = Path(tmpdir).joinpath(f"{comp}_{cov}.txt")
                with open(inputfile, "w") as fid:
                    out_vrt = Path(tmpdir).joinpath(f"{comp}_{cov}.vrt")
                    for fname in indir.rglob(f"FC_*{comp}_{cov}*.tif"):
                        fid.writelines(str(fname) + "\n")
                        fid.flush()
                    cmd = [
                        "gdalbuildvrt",
                        "-input_file_list",
                        inputfile.as_posix(),
                        out_vrt.as_posix(),
                    ]
                    check_call(cmd)

                    outfile = outdir.joinpath(
                        f"{comp}_{cov}_{year}_{region}.tif"
                    )
                    cmd = [
                        "gdalwarp",
                        "-t_srs",
                        "EPSG:4326",
                        "-te",
                        str(bounds[0]),
                        str(bounds[1]),
                        str(bounds[2]),
                        str(bounds[3]),
                        out_vrt.as_posix(),
                        outfile.as_posix(),
                    ]
                    check_call(cmd)

    with rasterio.open(outfile.as_posix()) as src:
        profile = src.profile

    profile["count"] = 3
    profile["tiled"] = True
    profile["blockxsize"] = 256
    profile["blockysize"] = 256
    profile["compress"] = "DEFLATE"
    profile["zlevel"] = 9
    profile["predictor"] = 2
    print(profile)
    for comp in composite.keys():
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = outdir.joinpath(
                f"FC_{comp}_Medoid.v{ver}.MCD43A4.{region}.{year}.006.tif"
            )
            with rasterio.open(outfile.as_posix(), "w", **profile) as outds:
                for idx, key in enumerate(
                    ["bare_soil", "phot_veg", "nphot_veg"]
                ):
                    fname = outdir.joinpath(
                        f"{comp}_{key}_{year}_{region}.tif"
                    )
                    with rasterio.open(fname.as_posix()) as src:
                        outds.write(src.read(1), idx + 1)

                    os.remove(fname.as_posix())

                # build overviews with 5 levels
                overviews = [2 ** j for j in range(1, overviews_level + 1)]
                outds.build_overviews(overviews, resampling)


@click.command(
    "--fc-medoid-process", help="Processing fractional cover medoid"
)
@click.option(
    "--src-root-dir",
    help="source root path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=os.getcwd(),
)
@click.option(
    "-r",
    "--region",
    help="region to process: west, east, australia",
    type=click.STRING,
    required=True,
)
@click.option(
    "--year", help="Year of source file", type=click.INT, required=True,
)
@click.option(
    "--output-dir",
    help="Full path to destination directroy.",
    type=click.Path(dir_okay=True, file_okay=False),
    default=os.getcwd(),
)
@click.option(
    "--composite-type",
    help="FC composite to be computed: month or season",
    type=click.STRING,
    default="month",
    show_default=True,
)
@click.option(
    "--version",
    help="Product version",
    type=click.STRING,
    default="310",
    show_default=True,
)
def main(
    src_root_dir: click.Path,
    region: click.STRING,
    year: click.INT,
    output_dir: click.Path,
    composite_type: click.STRING,
    version: click.STRING,
):
    if region == "west":
        tiles = __west__
    else:
        raise NotImplementedError(f"{region} not implemented")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(output_dir).joinpath(f"{region}{year}")
        outdir.mkdir(exist_ok=True)
        indir = Path(tmpdir)
        for htile, vtile in tiles:
            fname = (
                f"FC.v{version}.MCD43A4.h{htile:02d}"
                f"v{vtile:02}.{year}.006.nc"
            )

            fc_path = Path(src_root_dir).joinpath(fname)

            # if file does not exists then download from nci
            # thread serer
            if not fc_path.exists():
                fc_path = download(fname, tmpdir)

            compute_medoid(
                fc_path, indir, version, composite_type, htile, vtile, year
            )
        build_vrt_mosaic(indir, outdir, composite_type, region, year, version)


if __name__ == "__main__":
    main()
