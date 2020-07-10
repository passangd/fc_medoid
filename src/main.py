#!/usr/bin/env python3

import os
import sys
import warnings
import datetime
import cftime
import tempfile
import subprocess
import uuid
import logging
import logging.config
import shutil
from subprocess import check_call
from pathlib import Path
from typing import Optional, Union, List

import yaml
import boto3
import osr
import click
import netCDF4
import rasterio
import numpy as np
from rasterio import Affine
from rasterio.enums import Resampling
from botocore.exceptions import ClientError

from fast_medoid import medoid

__fc__ = {"phot_veg": 0, "nphot_veg": 1, "bare_soil": 2}
__spring__ = [8, 9, 10]  # September, October and November
__summer__ = [11, 0, 1]  # December, January and February
__autumn__ = [2, 3, 4]  # March, April and May
__winter__ = [5, 6, 7]  # June, July and August

__composite_map__ = {
    "season": {
        "Spring": __spring__,
        "Summer": __summer__,
        "Autumn": __autumn__,
        "Winter": __winter__,
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
__index_map__ = {
    "season": {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"},
    "month": {
        0: "January",
        1: "February",
        2: "March",
        3: "April",
        4: "May",
        5: "June",
        6: "July",
        7: "August",
        8: "September",
        9: "October",
        10: "November",
        11: "December",
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
    (30, 10),
    (30, 11),
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


def s3_upload(
    up_file: Union[str, Path],
    s3_client: boto3.Session.client,
    bucket: str,
    prefix: Optional[str] = "",
    replace: Optional[bool] = True,
):
    """Uploads file into s3_bucket in a prefix folder.

    :param up_file: Full path to file that will be uploaded to s3.
    :param s3_client: s3 client to upload the files.
    :param bucket: The name of s3 bucket to upload the files into.
    :param prefix: The output directory to put file in s3 bucket.
    :param replace: Optional parameter to specify if files in s3 bucket
                    is to be replaced if it exists.
    :return:
        True if uploaded, else False
    :raises:
        ValueError if file fails to upload to s3 bucket.
    """

    filename = Path(up_file).name
    s3_path = f"{prefix}/{filename}"
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_path)
        if replace:
            s3_client.upload_file(up_file.as_posix(), bucket, Key=s3_path)
            return True
        else:
            STATUS_LOGGER.info(
                f"{filename} exists in {bucket}/{prefix}, skipped upload"
            )
            return False
    except ClientError:
        try:
            s3_client.upload_file(up_file.as_posix(), bucket, Key=s3_path)
            return True
        except ClientError:
            raise ValueError(
                f"failed to upload {filename} at {bucket}/{prefix}"
            )


def download(
    filename, outdir,
):
    """downloads data from nci thread server"""
    url = f"{__base_url__}{filename}"

    cmd = ["wget", '-q', url, "-P", outdir]
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


def _munge_metadata(metadata_doc: dict, outfile: Path):
    """Consolidates the metadata used in processing stormsurge.

    :param metadata_doc: The lineage informations related to the
                              stormsurge product.
    :param outfile: The output filename where metadata will be written.
    """
    lineage = dict()
    for param, val in metadata_doc["lineage"].items():
        if isinstance(val, dict):
            for _param, _val in val.items():
                lineage[_param] = _val
        else:
            lineage[param] = val

    metadata_doc["lineage"] = lineage
    with open(outfile.as_posix(), "w") as fid:
        yaml.dump(metadata_doc, fid, sort_keys=False)


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
    filename, output_dir, ver, composite_type, htile, vtile, year,
):
    with netCDF4.Dataset(filename) as ds:
        month_idx_tmp = [[] for _ in range(12)]
        season_idx_tmp = [[] for _ in range(4)]

        month_idx_dates = dict()
        season_idx_dates = dict()

        for its, ts in enumerate(ds["time"]):
            date = cftime.num2pydate(ts, ds["time"].units)
            month_idx_tmp[date.month - 1].append(its)

            try:
                month_idx_dates[__index_map__["month"][date.month - 1]].append(
                    date
                )
            except KeyError:
                month_idx_dates[__index_map__["month"][date.month - 1]] = [
                    date
                ]

            # gather data index if data falls in season's month
            for _its, season in enumerate(
                [__spring__, __summer__, __autumn__, __winter__]
            ):
                if date.month - 1 in season:
                    season_idx_tmp[_its].append(its)
                    try:
                        season_idx_dates[__index_map__["season"][_its]].append(
                            date
                        )
                    except KeyError:
                        season_idx_dates[__index_map__["season"][_its]] = [
                            date
                        ]

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
                    STATUS_LOGGER.info(
                        "Incomplete files encountered, still processed"
                    )
                else:
                    month_idx_tmp[i] = []

        # checks if season has all three month's data to form
        # a seasonal composite
        for i in range(len(season_idx_tmp)):

            s = season_idx_tmp[i]
            if len(s) == 0:
                continue
            ts = [
                    cftime.num2date(ds["time"][s[j]], ds["time"].units)
                    for j in range(len(s))
            ]
            months = set([t.month for t in ts])
            if len(months) != 3:
                season_idx_tmp[i] = []

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
            dates_composite = month_idx_dates
            for m, m_idx in enumerate(month_idx):
                d = data[m_idx, ...]
                month_medoid = medoid(d)
                medoid_data_list.append(month_medoid)
                ts = cftime.num2pydate(ds["time"][m_idx[0]], ds["time"].units)
                ts = ts.replace(day=1)
                medoid_timestamp_list.append(ts)
        elif composite_type == "season":
            # compute seasonal medoid
            dates_composite = season_idx_dates
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
            STATUS_LOGGER.info(f"no data for {filename}")
            return

        if len(medoid_data_list) == 1:
            medoid_data = np.expand_dims(medoid_data_list[0], axis=0)
        else:
            medoid_data = np.stack(medoid_data_list)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=rasterio.errors.NotGeoreferencedWarning)
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
        return dates_composite


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
    valid_comp_keys = []
    for comp in composite.keys():
        for cov in __fc__.keys():
            with tempfile.TemporaryDirectory() as tmpdir:
                inputfile = Path(tmpdir).joinpath(f"{comp}_{cov}.txt")
                with open(inputfile, "w") as fid:
                    out_vrt = Path(tmpdir).joinpath(f"{comp}_{cov}.vrt")
                    fnames_list = []
                    for fname in indir.rglob(f"FC_*{comp}_{cov}*.tif"):
                        fnames_list.append(fname)
                        fid.writelines(str(fname) + "\n")
                        fid.flush()
                    if not fnames_list:
                        break
                    valid_comp_keys.append(comp)
                    cmd = [
                        "gdalbuildvrt",
                        "-q",
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
                        "-q",
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

    composite_names = dict()
    for comp in set(valid_comp_keys):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = outdir.joinpath(
                f"FC_{comp}_Medoid.v{ver}.MCD43A4.{region}.{year}.006.tif"
            )
            composite_names[comp] = outfile
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
    return composite_names


def sns_send(
    aws_session: boto3.Session,
    topic_arn: str,
    messages: Optional[str] = "",
    subject: Optional[str] = "",
    msg_attributes: Optional[dict] = None,
):
    """Simple notification service to report pgr status.

    :param aws_session: The aws session to invoke sns client.
    :param topic_arn: The sns topic arn to publish the message.
    :param messages: The messages to be published.
    :param subject: The subject of an sns message.
    :param msg_attributes: The MessageAttributes for sns service.
    """
    if msg_attributes is None:
        msg_attributes = {}

    sns_client = aws_session.client("sns")
    sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=messages,
        MessageAttributes=msg_attributes,
    )


def post_process(
    cleanup_dir: Optional[Path] = None,
    up_files: Optional[List[Path]] = None,
    aws_session: Optional[boto3.Session.client] = None,
    s3_bucket: Optional[str] = None,
    bucket_prefix: Optional[str] = "",
    ingest_database: Optional[bool] = False,
    db_table_name: Optional[bool] = "",
    sns_arn: Optional[str] = None,
):
    """Cleans up the directory created by the process and exit"""

    def _s3_upload(fids):
        dynb_db = boto3.resource('dynamodb')
        table = dynb_db.Table(db_table_name)
        s3_client = aws_session.client("s3")
        for f in fids:
            up_flag = s3_upload(
                f, s3_client, s3_bucket, prefix=bucket_prefix, replace=False
            )
            if not up_flag:
                continue

            if ingest_database:
                if f.suffix == ".yaml":
                    with open(f.as_posix()) as fid:
                        meta = yaml.safe_load(fid)
                        item = {
                            "id": meta["id"],
                            "composite": meta["lineage"]["composite_type"],
                            "region": meta["properties"]["region"],
                            "composite_year": meta["lineage"]["composite_dates"][0].year,
                            "sensor": meta["properties"]["instrument"],
                            "creation_datetime": meta["properties"]["creation_datetime"],
                            "path": meta["measurement"]
                        }
                        table.put_item(Item=item)

            if f.suffix == ".tif":
                if sns_arn is not None:
                    sns_send(
                        aws_session,
                        sns_arn,
                        messages=f"",
                        subject=f"FC Processing Completed: {f.name}",
                        msg_attributes=None
                    )

    if up_files is not None:
        if (aws_session is None) | (s3_bucket is None):
            STATUS_LOGGER.error(
                f"Missing aws session and bucket object to upload files"
                f" s3_bucket {s3_bucket} and aws_session is {aws_session}"
            )
        else:
            _s3_upload(up_files)

    if cleanup_dir is not None:
        try:
            shutil.rmtree(cleanup_dir)
        except Exception:
            STATUS_LOGGER.info(
                f"failed at clean up of {cleanup_dir}", exc_info=True
            )

    sys.exit(0)


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
    "--s3-bucket",
    help="The name of s3-bucket to save output.",
    type=click.STRING,
    required=True,
)
@click.option(
    "--ingest-database",
    help="Whether to update processing records to database",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--db-table-name",
    help="The AWS dynamo  database table name",
    type=click.STRING,
    default="DSR_FractionalCover"
)
@click.option(
    "--sns-arn",
    help="AWS SNS Topic Arn for notification service",
    type=click.STRING,
    default=None,
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
    s3_bucket: click.STRING,
    ingest_database: click.BOOL,
    db_table_name: click.STRING,
    sns_arn: click.STRING,
    version: click.STRING,
):
    if region == "west":
        tiles = __west__
    else:
        raise NotImplementedError(f"{region} not implemented")

    STATUS_LOGGER.info(f"Processing {composite_type} Fractional Cover: {region.upper()}")
    # base metadata document for fractional cover composite
    metadata_doc = {
        "id": "",
        "extents": [float(b) for b in __WA_BOUNDS__],
        "product": "Fractional Cover",
        "product_type": "medoid",
        "properties": {
            "instrument": "MODIS",
            "platform": ["AQU", "TER"],
            "gsd": 500.0,
            "epsg": 4326,
            "provider": {"name": "Landgate", "roles": ["processor", "host"]},
            "creation_datetime": datetime.datetime.utcnow().isoformat(),
            "fileformat": "GTiff",
            "region": f"{region.upper()}",
        },
        "Software": [
            "Landgate fc-medoid 0.1.0",
            "https://github.com/nci/geoglam.git:master",
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(output_dir).joinpath(f"{region}{year}")
        outdir.mkdir(exist_ok=True)
        indir = Path(tmpdir)
        source_metadata = []
        for htile, vtile in tiles:
            fname = (
                f"FC.v{version}.MCD43A4.h{htile:02d}"
                f"v{vtile:02}.{year}.006.nc"
            )

            fc_path = Path(src_root_dir).joinpath(fname)

            STATUS_LOGGER.info(f"computing medoid for {fname}")

            # if file does not exists then download from nci
            # thread serer
            if not fc_path.exists():
                fc_path = download(fname, tmpdir)
                source_metadata.append(f"{__base_url__}{fc_path.name}")
            else:
                source_metadata.append(str(fc_path))

            _dates = compute_medoid(
                fc_path, indir, version, composite_type, htile, vtile, year
            )
        STATUS_LOGGER.info(f"Creating {region} Australia FC medoid mosaic")
        _composites = build_vrt_mosaic(
            indir, outdir, composite_type, region, year, version
        )

        metadata_doc["lineage"] = (source_metadata,)

        up_files = []
        for _k, val in _composites.items():
            metadata_doc["id"] = str(uuid.uuid4())
            metadata_doc["lineage"] = {
                "source": source_metadata,
                "composite_type": _k,
                "composite_dates": _dates[_k],
            }
            metadata_doc["measurement"] = (
                f"s3://{s3_bucket}/{region}{year}/{val.name}"
            )
            metadata_doc["bands"] = {
                'band 1': 'bare soil',
                'band 2': 'photosynthetic vegetation',
                'band 3': 'non-photosynthetic vegetation'
            }
            up_files.append(val)
            yaml_file = val.with_suffix(".yaml")
            _munge_metadata(metadata_doc, yaml_file)
            up_files.append(yaml_file)

    # upload fractional cover mosiac to s3-bucket
    try:
        aws_session = boto3.Session()
    except Exception:
        STATUS_LOGGER.critical(
            f"failed to initialize aws session", exc_info=True
        )

    # post process
    post_process(
        **{
            "cleanup_dir": outdir,
            "up_files": up_files,
            "aws_session": aws_session,
            "s3_bucket": s3_bucket,
            "bucket_prefix": f"MODIS_FractionalCover/{outdir.name}",
            "ingest_database": ingest_database,
            "db_table_name": db_table_name,
            "sns_arn": sns_arn,
        }
    )


if __name__ == "__main__":
    LOG_CONFIG = Path(__file__).parent.joinpath("logging.cfg")
    logging.config.fileConfig(LOG_CONFIG.as_posix())
    STATUS_LOGGER = logging.getLogger()
    main()
