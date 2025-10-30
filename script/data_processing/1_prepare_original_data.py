import sys
sys.path.append("../")
from utils.geo_tools import *
import numpy as np
import os
import re
from joblib import Parallel, delayed
import tqdm
from osgeo import ogr, gdal
from collections import defaultdict
import subprocess


def extract_acquisition_key(filename):
    """
    Extrait une clé unique pour identifier les acquisitions similaires.
    Clé = (satellite, date, heure, type_acq) sans les minutes
    """
    basename = os.path.basename(filename)
    # Exemples: 
    # - subset_20_of_TDX1_SAR_SSC_SM_S_SRA_20131114T172538_20131114T172543_Cal_TF_TC.tif
    # - subset_0_of_PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536_Cal_Tf_Tc_TC.tif
    # - subset_0_of_TSX1_SAR_SSC_SM_S_SRA_20091010T172514_20091010T172522_Cal_TF_TC.tif
    # On cherche: satellite (TDX1/TSX1/PAZ1), date (YYYYMMDD), heure (HH) du début
    match = re.search(r"([TP][DASX][ZX]\d).*?(\d{8})T(\d{2})\d{4}_\d{8}T\d{6}", basename)
    if match:
        satellite = match.group(1)  # TDX1, TSX1 ou PAZ1
        date = match.group(2)       # 20131114
        hour = match.group(3)       # 17
        return (satellite, date, hour)
    return None


def smart_merge_images(file_list, output_path):
    """
    Fusionne intelligemment plusieurs images en gérant les zones de 0.
    - Si les deux pixels sont non-nuls: moyenne
    - Si un seul est non-nul: prend celui qui est non-nul
    - Si les deux sont nuls: 0
    """
    # Ouvrir toutes les images
    datasets = [gdal.Open(f, gdal.GA_ReadOnly) for f in file_list]
    
    # Récupérer les informations de géoréférencement de la première image
    first_ds = datasets[0]
    geotransform = first_ds.GetGeoTransform()
    projection = first_ds.GetProjection()
    
    # Déterminer l'étendue combinée de toutes les images
    # Pour simplifier, on utilise l'union des étendues
    x_min, x_max, y_min, y_max = None, None, None, None
    
    for ds in datasets:
        gt = ds.GetGeoTransform()
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        
        x_left = gt[0]
        x_right = gt[0] + x_size * gt[1]
        y_top = gt[3]
        y_bottom = gt[3] + y_size * gt[5]
        
        if x_min is None:
            x_min, x_max = min(x_left, x_right), max(x_left, x_right)
            y_min, y_max = min(y_top, y_bottom), max(y_top, y_bottom)
        else:
            x_min = min(x_min, x_left, x_right)
            x_max = max(x_max, x_left, x_right)
            y_min = min(y_min, y_top, y_bottom)
            y_max = max(y_max, y_top, y_bottom)
    
    # Calculer la taille de l'image de sortie
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])
    out_width = int((x_max - x_min) / pixel_width)
    out_height = int((y_max - y_min) / pixel_height)
    
    # Obtenir le nombre de bandes
    n_bands = first_ds.RasterCount
    
    # Créer les arrays de sortie
    merged_data = np.zeros((out_height, out_width, n_bands), dtype=np.float32)
    count_data = np.zeros((out_height, out_width, n_bands), dtype=np.float32)
    
    # Pour chaque image source
    for ds in datasets:
        gt = ds.GetGeoTransform()
        
        # Calculer la position de cette image dans l'image de sortie
        # Pour gérer le cas où pixel_height est négatif
        x_offset = int(round((gt[0] - x_min) / pixel_width))
        y_offset = int(round((y_max - gt[3]) / pixel_height))
        
        # Lire chaque bande
        for band_idx in range(n_bands):
            band = ds.GetRasterBand(band_idx + 1)
            data = band.ReadAsArray()
            
            if data is None:
                continue
            
            rows, cols = data.shape
            
            # Masque des pixels non-nuls (excluant 0 et nodata)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                valid_mask = (data != 0) & (data != nodata) & ~np.isnan(data)
            else:
                valid_mask = (data != 0) & ~np.isnan(data)
            
            # Calculer les indices de destination
            dst_y_start = max(0, y_offset)
            dst_y_end = min(out_height, y_offset + rows)
            dst_x_start = max(0, x_offset)
            dst_x_end = min(out_width, x_offset + cols)
            
            # Calculer les indices source correspondants
            src_y_start = dst_y_start - y_offset
            src_y_end = src_y_start + (dst_y_end - dst_y_start)
            src_x_start = dst_x_start - x_offset
            src_x_end = src_x_start + (dst_x_end - dst_x_start)
            
            # Vérifier la validité des dimensions
            if dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
                continue
            if src_y_end <= src_y_start or src_x_end <= src_x_start:
                continue
            if src_y_end > rows or src_x_end > cols:
                continue
            
            # Accumuler les valeurs valides
            src_valid = valid_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            src_data = data[src_y_start:src_y_end, src_x_start:src_x_end]
            
            merged_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end, band_idx] += src_data * src_valid
            count_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end, band_idx] += src_valid.astype(np.float32)
    
    # Calculer la moyenne là où on a des données
    final_data = np.zeros_like(merged_data)
    for band_idx in range(n_bands):
        mask = count_data[:, :, band_idx] > 0
        final_data[:, :, band_idx][mask] = merged_data[:, :, band_idx][mask] / count_data[:, :, band_idx][mask]
    
    # Créer le fichier de sortie
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, out_width, out_height, n_bands, gdal.GDT_Float32, 
                           options=['COMPRESS=LZW'])
    
    # Définir la géotransformation de sortie
    out_geotransform = list(geotransform)
    out_geotransform[0] = x_min
    out_geotransform[3] = y_max
    out_ds.SetGeoTransform(out_geotransform)
    out_ds.SetProjection(projection)
    
    # Écrire les données
    for band_idx in range(n_bands):
        out_band = out_ds.GetRasterBand(band_idx + 1)
        out_band.WriteArray(final_data[:, :, band_idx])
        out_band.SetNoDataValue(-999)
        out_band.FlushCache()
    
    # Fermer les datasets
    out_ds = None
    for ds in datasets:
        ds = None


def merge_similar_acquisitions(files, temp_dir):
    """
    Fusionne les images qui correspondent à la même acquisition (même satellite, date, heure).
    Retourne la liste des fichiers fusionnés (ou originaux si pas de merge nécessaire).
    """
    # Filtrer les fichiers .tif uniquement (pas les .ovr ni autres)
    files = [f for f in files if f.lower().endswith('.tif') and not f.lower().endswith('.ovr')]
    
    # Grouper les fichiers par clé d'acquisition
    groups = defaultdict(list)
    files_without_key = []
    
    for f in files:
        key = extract_acquisition_key(f)
        if key:
            groups[key].append(f)
        else:
            # Fichier qui ne correspond pas au pattern, on le garde tel quel
            files_without_key.append(f)
    
    merged_files = []
    os.makedirs(temp_dir, exist_ok=True)
    
    # Traiter les groupes identifiés
    for key, group_files in groups.items():
        if len(group_files) > 1:
            # Plusieurs fichiers pour la même acquisition -> merge intelligent
            print(f"Fusion intelligente de {len(group_files)} images pour {key}")
            # Utiliser le nom du premier fichier comme base
            first_file = sorted(group_files)[0]
            output_name = os.path.basename(first_file)
            output_path = os.path.join(temp_dir, output_name)
            
            try:
                smart_merge_images(sorted(group_files), output_path)
                merged_files.append(output_path)
                print(f"  -> Fusion réussie: {output_name}")
            except Exception as e:
                print(f"Erreur lors du merge: {e}")
                # En cas d'erreur, garder le premier fichier
                merged_files.append(first_file)
        else:
            # Un seul fichier pour cette acquisition
            merged_files.append(group_files[0])
    
    # Ajouter les fichiers sans clé
    merged_files.extend(files_without_key)
    
    if files_without_key:
        print(f"Note: {len(files_without_key)} fichiers n'ont pas pu être analysés (format non standard), ils seront traités tels quels")
    
    return merged_files


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
        # array2raster(
        #     img[:, :, 0], meta, os.path.join(os.path.dirname(temp), name_0)
        # )
        # array2raster(
        #     img[:, :, 1], meta, os.path.join(os.path.dirname(temp), name_1)
        # )
        array2raster(
            img[:, :, [0, -1]], meta, os.path.join(os.path.dirname(temp), name_0)
        )
        array2raster(
            img[:, :, [1, -1]], meta, os.path.join(os.path.dirname(temp), name_1)
        )
        os.remove(temp)


if __name__ == "__main__":

    shp_path = "../../DATASET/SHP/"
    output = "../../DATA/"
    temp_merge_dir = "../../DATA/TEMP_MERGED/"

    for acq in ["asc", "dsc"]:
        acq_upper = acq.upper()  # ASC ou DSC
        os.makedirs(os.path.join(output, acq_upper), exist_ok=True)
        path = f"/media/mgallet/BACK UP/DATA/X_SAR/{acq}/*.tif"
        all_files = sorted(glob.glob(path))
        
        # Étape 1: Fusionner les acquisitions similaires
        print(f"\n=== Fusion des acquisitions similaires pour {acq_upper} ===")
        merged_files = merge_similar_acquisitions(all_files, os.path.join(temp_merge_dir, acq))
        print(f"Fichiers après fusion: {len(merged_files)} (originaux: {len(all_files)})")
        
        acq_initial = acq_upper[0]  # A ou D (pour le nom de fichier)
        classe = [os.path.basename(x) for x in glob.glob(os.path.join(shp_path, "*"))]

        # Étape 2: Traiter les fichiers fusionnés
        print(f"\n=== Découpage des images pour {acq_upper} ===")
        for img in tqdm.tqdm(merged_files):
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
os.rmdir(temp_merge_dir)
# Erreur pour l'instant propablement du au Parallel
# Parallel(n_jobs=-1)(
#     delayed(split_channel)(temp) for temp in tqdm.tqdm(files_proc)
# )
