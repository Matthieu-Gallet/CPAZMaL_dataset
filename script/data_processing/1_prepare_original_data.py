import sys
sys.path.append("../")
from utils.geo_tools import *
import numpy as np
import os
import re
from joblib import Parallel, delayed
import tqdm
from osgeo import ogr


def get_stations_id(i):
    ds = ogr.Open(i)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    ft = lyr.GetNextFeature()
    while ft:
        stations_id = ft.GetFieldAsString("id")
        ft = lyr.GetNextFeature()
    return stations_id


def extract_new_name(filename, acq, target):
    filename = os.path.basename(filename)
    # Use regular expression to extract the first character and the date
    match = re.search(r"([PT])\w*.*?(\d{8})", filename)
    if match:
        satellite_initial = match.group(1)
        date = match.group(2)
        return f"{satellite_initial}{acq}_{target}_{date}.tif"
    return None


def clip_shp_img(img, acq, dirs, s):
    t_n, t_i = os.path.basename(s).split("_")[0:2]
    name = t_n + t_i
    out_dir = os.path.join(dirs, name)
    os.makedirs(out_dir, exist_ok=True)
    id = get_stations_id(s)
    outname = extract_new_name(img, acq, name)
    gdal_clip_shp_raster(img, s, os.path.join(out_dir, outname), country_name=id)


def split_channel(temp):
    img, meta = load_data(temp)
    if img.shape[-1] == 2:
        # Image avec 2 bandes : HH et masque
        name_comp = os.path.basename(temp).split("_")
        name = name_comp[0] + "_" + name_comp[1] + f"_HH_{name_comp[-1]}"
        os.rename(temp, os.path.join(os.path.dirname(temp), name))

    if img.shape[-1] >= 3:
        # Image avec 3+ bandes : HH, HV, masque
        name_comp = os.path.basename(temp).split("_")
        name_0 = name_comp[0] + "_" + name_comp[1] + f"_HH_{name_comp[-1]}"
        name_1 = name_comp[0] + "_" + name_comp[1] + f"_HV_{name_comp[-1]}"
        
        # Extraire les canaux correctement (canal 0 = HH, canal 1 = HV, dernier canal = masque)
        # Ne pas inclure le masque dans les nouvelles images, ou l'inclure comme 2e bande
        array2raster(
            img[:, :, 0], meta, os.path.join(os.path.dirname(temp), name_0)
        )
        array2raster(
            img[:, :, 1], meta, os.path.join(os.path.dirname(temp), name_1)
        )
        os.remove(temp)


if __name__ == "__main__":

    shp_path = "../../DATASET/SHP/"
    output = "../../DATA/"

    for acq in ["asc", "dsc"]:
        acq_upper = acq.upper()  # ASC ou DSC
        os.makedirs(os.path.join(output, acq_upper), exist_ok=True)
        path = f"/media/mgallet/BACK UP/DATA/X_SAR/{acq}/*.tif"
        files = sorted(glob.glob(path))
        acq_initial = acq_upper[0]  # A ou D (pour le nom de fichier)
        classe = [os.path.basename(x) for x in glob.glob(os.path.join(shp_path, "*"))]

        for img in tqdm.tqdm(files):
            for c in classe:
                dirs = os.path.join(output, acq_upper, c)  # Utiliser acq_upper (ASC/DSC)
                os.makedirs(dirs, exist_ok=True)
                shp = os.path.join(shp_path, f"{c}/*.shp")
                shp_files = glob.glob(shp)

                Parallel(n_jobs=-1)(
                    delayed(clip_shp_img)(img, acq_initial, dirs, s) for s in shp_files  # Utiliser acq_initial (A/D)
                )
files_proc = glob.glob(f"{output}**/*.tif", recursive=True)
for temp in tqdm.tqdm(files_proc):
    split_channel(temp)
# Erreur pour l'instant propablement du au Parallel
# Parallel(n_jobs=-1)(
#     delayed(split_channel)(temp) for temp in tqdm.tqdm(files_proc)
# )
