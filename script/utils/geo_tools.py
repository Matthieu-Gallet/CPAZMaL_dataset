from osgeo_utils import gdal_merge
from osgeo import gdal
from joblib import Parallel, delayed

import subprocess

from os import system, remove
from os.path import exists, dirname, join
import numpy as np
import glob
from tqdm import tqdm


def load_data(file_name, gdal_driver="GTiff"):
    """
    Converts a GDAL compatable file into a numpy array and associated geodata.
    The rray is provided so you can run with your processing - the geodata consists of the geotransform and gdal dataset object
    If you're using an ENVI binary as input, this willr equire an associated .hdr file otherwise this will fail.
    This needs modifying if you're dealing with multiple bands.

    VARIABLES
    file_name : file name and path of your file

    RETURNS
    image array
    (geotransform, inDs)
    """
    driver = gdal.GetDriverByName(gdal_driver)  ## http://www.gdal.org/formats_list.html
    driver.Register()

    inDs = gdal.Open(file_name, gdal.GA_ReadOnly)

    if inDs is None:
        print("Couldn't open this file: %s" % (file_name))
    else:
        pass
    # Extract some info form the inDs
    geotransform = inDs.GetGeoTransform()
    projection = inDs.GetProjection()

    # Get the data as a numpy array
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize

    channel = inDs.RasterCount
    image_array = np.zeros((rows, cols, channel), dtype=np.float32)
    for i in range(channel):
        band = inDs.GetRasterBand(i + 1)
        data_array = band.ReadAsArray(0, 0, cols, rows)
        # Obtenir la valeur nodata de la bande
        nodata = band.GetNoDataValue()
        if nodata is not None:
            # Remplacer les valeurs nodata par NaN
            data_array = np.where(data_array == nodata, np.nan, data_array)
        image_array[:, :, i] = data_array
    inDs = None
    return image_array, (geotransform, projection)


def array2raster(data_array, geodata, file_out, gdal_driver="GTiff", nodata_value=-999):
    """
    Converts a numpy array to a specific geospatial output
    If you provide the geodata of the original input dataset, then the output array will match this exactly.
    If you've changed any extents/cell sizes, then you need to amend the geodata variable contents (see below)

    VARIABLES
    data_array = the numpy array of your data
    geodata = (geotransform, inDs) # this is a combined variable of components when you opened the dataset
                            inDs = gdal.Open(file_name, GA_ReadOnly)
                            geotransform = inDs.GetGeoTransform()
                            see data2array()
    file_out = name of file to output to (directory must exist)
    gdal_driver = the gdal driver to use to write out the data (default is geotif) - see: http://www.gdal.org/formats_list.html

    RETURNS
    None
    """

    if not exists(dirname(file_out)):
        print("Your output directory doesn't exist - please create it")
        print("No further processing will take place.")
    else:
        post = geodata[0][1]
        original_geotransform, projection = geodata

        rows, cols, bands = data_array.shape
        # adapt number of bands to input data

        # Set the gedal driver to use
        driver = gdal.GetDriverByName(gdal_driver)
        driver.Register()

        # Creates a new raster data source
        outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

        # Write metadata
        originX = original_geotransform[0]
        originY = original_geotransform[3]

        outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
        outDs.SetProjection(projection)

        # Write raster datasets
        for i in range(bands):
            outBand = outDs.GetRasterBand(i + 1)
            outBand.SetNoDataValue(nodata_value)
            outBand.WriteArray(data_array[:, :, i])

        # print("Output saved: %s" % file_out)


def gdal_resolution(inraster):
    ds = gdal.Open(inraster)
    gt = ds.GetGeoTransform()
    res = gt[1]
    return res


def check_resolution(inref, inraster):
    raster_ref = gdal_resolution(inref)
    raster_c = gdal_resolution(inraster)
    return raster_ref == raster_c


def gdal_projection(inraster):
    ds = gdal.Open(inraster)
    pr = ds.GetProjection()
    return pr


def check_projection(inref, inraster):
    pr_ref = gdal_projection(inref)
    pr_c = gdal_projection(inraster)
    return pr_ref == pr_c


def check_ResProj_files(aux_path, data_path):
    re_d, pr_d = check_data(data_path)
    re_a, pr_a = check_data(aux_path)
    if re_d == re_a and pr_d == pr_a:
        return 1
    else:
        print("Resolution or projection not compatible")
        print(re_d, re_a)
        print(pr_d, pr_a)
        return 0


def check_data(data_path):
    data_files = glob.glob(data_path)
    dataR = data_files[0]
    for i in data_files[1:]:
        condR = check_resolution(dataR, i)
        condP = check_projection(dataR, i)
        if condR and condP:
            dataR = i
        else:
            print(f"data not compatible in projection {condP} or resolution {condR}")
            print(i)
            return 0
    return gdal_resolution(dataR), gdal_projection(dataR)


def check_with_nan(img):
    f = False
    try:
        c2, _ = load_data(img)
        for i in range(2):
            for j in range(2):
                cond = ((c2[:, :, i][c2[:, :, j] == -999] == -999).all()) & (
                    len(c2[:, :, i][c2[:, :, j] == -999] == -999) > 0
                )
                f = f | cond
    except:
        f = True
    return f


def clean_data_nan(path):
    list_files = glob.glob(path)
    for i in tqdm(list_files):
        if check_with_nan(i):
            remove(i)


def stats_dataset(path, dico):
    list_files = glob.glob(join(path, "*.tif"))
    for i in tqdm(list_files):
        c, _ = load_data(i)
        nbands = c.shape[2]
        try:
            dico["mean"].append(
                [np.mean(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["std"].append(
                [np.std(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["min"].append(
                [np.min(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["max"].append(
                [np.max(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
        except:
            print(i)
    return dico


def gdal_clip_shp_raster(inraster, inshape, outraster, country_name):
    result = subprocess.run(
        [
            "gdalwarp",
            "-overwrite",  # Permet d'écraser les fichiers existants
            "-of",
            "GTiff",
            "-dstnodata",
            "-999",
            "-ot",
            "Float32",
            inraster,
            outraster,
            "-cutline",
            inshape,
            "-crop_to_cutline",
            "-cwhere",
            f"id='{country_name}'",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Afficher les erreurs si la commande échoue
    if result.returncode != 0:
        print(f"Erreur gdalwarp pour {outraster}:")
        print(f"  stderr: {result.stderr}")
    
    return result.returncode


def gdal_merge_rasters(in_R1, in_R2, outraster):
    gdal_merge.main(
        [
            "",
            "-o",
            outraster,
            "-separate",
            "-ot",
            "Float32",
            "-of",
            "GTiff",
            "-n",
            "0",
            "-a_nodata",
            "-999",
            "-ot",
            "Float32",
            "-of",
            "GTiff",
            "-co",
            "COMPRESS=NONE",
            "-co",
            "BIGTIFF=IF_NEEDED",
            in_R1,
            in_R2,
        ]
    )


def count_value_altitude(prediction, dem, alti_min, alti_max, step, condition):
    lower = alti_min
    total = 0
    n_pix = []
    for upper in np.arange(alti_min + step, alti_max, step):
        count_alt = prediction[(dem >= lower) & (dem < upper) & condition]
        n_pix.append(np.nansum(count_alt))
        total += count_alt.size
        lower = upper
    count_alt = prediction[(dem >= lower) & condition]
    n_pix.append(np.nansum(count_alt))
    total += count_alt.size
    alti = np.array(n_pix) * 100 / total
    return alti


def count_value_orien_alti(
    prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
):
    lower_o = orientation[0]
    alti_theta = []
    for upper_o in orientation[1:]:
        cond_theta = (theta >= lower_o) & (theta < upper_o) & cond_pente
        array_alti = count_value_altitude(
            prediction, dem, alti_min, alti_max, step, cond_theta
        )
        alti_theta.append(array_alti)
        lower_o = upper_o
    cond_theta = (theta >= lower_o) & cond_pente
    array_alti = count_value_altitude(
        prediction, dem, alti_min, alti_max, step, cond_theta
    )
    alti_theta.append(array_alti)
    return np.array(alti_theta)


def count_value_pen_ori_alti(
    prediction, gradient, dem, theta, orientation, pente, alti_min, alti_max, step
):
    alpeor = []
    lower_p = pente[0]
    for upper_p in tqdm(pente[1:]):
        cond_pente = (gradient >= lower_p) & (gradient < upper_p)
        alpeor.append(
            count_value_orien_alti(
                prediction,
                theta,
                dem,
                orientation,
                alti_min,
                alti_max,
                step,
                cond_pente,
            )
        )
        lower_p = upper_p
    cond_pente = gradient >= lower_p
    alpeor.append(
        count_value_orien_alti(
            prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
        )
    )
    alpeor = np.array(alpeor)
    return np.moveaxis(alpeor, 0, 2)


def gdal_resampling(i_path, resolution):
    o_path = i_path[:-4] + "_resampled_" + str(resolution) + ".tif"
    cmd = f"gdalwarp -tr {resolution} {resolution} -of GTiff {i_path} {o_path}"
    system(cmd)
    remove(i_path)
    return o_path
