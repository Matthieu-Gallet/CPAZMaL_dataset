#!/usr/bin/env python3
"""
Script optimisé pour créer un dataset HDF5 structuré pour le Machine Learning.

Améliorations par rapport à 2_create_dataset.py:
1. Structure HDF5 optimisée avec index pour accès rapide
2. Métadonnées enrichies (satellite, résolution, angle d'incidence)
3. Organisation par satellite et période temporelle
4. Chunking et compression optimisés pour ML
5. Attributs de normalisation pré-calculés
6. Index temporels pour extraction rapide de séquences
"""

# Configuration GDAL pour éviter les warnings
import os
os.environ['GDAL_PAM_ENABLED'] = 'NO'

from datetime import datetime
import h5py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import rasterio

# Importer les fonctions utilitaires
from script.utils.processing_utils import resize_image, load_data_rasterio, compute_mask_statistics, get_date_ranges_by_satellite


def extract_metadata_from_filename(filename):
    """
    Extrait les métadonnées du nom de fichier.
    Format: PA_ABL001_HH_20200110.tif ou TX_ABL001_HH_20071024.tif
    
    Returns:
        dict: satellite_code, group, polarisation, date
    """
    basename = os.path.basename(filename)
    parts = basename.replace(".tif", "").split("_")
    
    return {
        'satellite_code': parts[0],  # PA (PAZ) ou TX (TerraSAR-X) ou TD (TanDEM-X)
        'group': parts[1],
        'polarisation': parts[2],
        'date': parts[3]
    }


def get_satellite_info(date_str, acquisition_df):
    """
    Récupère les informations satellite depuis data.csv pour une date donnée.
    
    Args:
        date_str: Date au format YYYYMMDD
        acquisition_df: DataFrame avec les données d'acquisition
    
    Returns:
        dict: Informations satellite (nom, angle, résolution, orbite)
    """
    # Convertir date_str YYYYMMDD en format comparable
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    date_formatted = date_obj.strftime('%Y-%m-%d')
    
    # Chercher dans le DataFrame
    match = acquisition_df[acquisition_df['Date'] == date_formatted]
    
    if len(match) > 0:
        row = match.iloc[0]
        return {
            'satellite': row['Satellite'],
            'pass': row['Pass'],
            'angle_incidence': float(row['Angle Incidence'].replace('°', '')),
            'resolution_range': float(row['Résolution Range'].replace(' m', '')),
            'resolution_azimuth': float(row['Résolution Azimuth'].replace(' m', ''))
        }
    else:
        # Valeurs par défaut si non trouvé
        return {
            'satellite': 'Unknown',
            'pass': 'Unknown',
            'angle_incidence': 0.0,
            'resolution_range': 0.0,
            'resolution_azimuth': 0.0
        }


def open_files_with_metadata(group_folder, polarisation, acquisition_df, nan=-999):
    """
    Charge les fichiers avec métadonnées enrichies depuis data.csv
    
    Returns:
        tuple: (tensor_img, tensor_mask, tensor_time, metadata_list, mean_geo, mean_shape)
    """
    tensor_data = []
    tensor_shape = []
    tensor_geo = []
    tensor_time = []
    metadata_list = []
    
    files = sorted(glob.glob(os.path.join(group_folder, f"*{polarisation}*.tif")))
    
    for file in files:
        # Charger l'image avec rasterio
        img, geo = load_data_rasterio(file)
        tensor_data.append(img)
        tensor_shape.append(img.shape)
        tensor_geo.append(geo[0])
        
        # Extraire métadonnées du nom de fichier
        file_meta = extract_metadata_from_filename(file)
        dt = file_meta['date']
        tensor_time.append(dt)
        
        # Récupérer infos satellite depuis CSV
        sat_info = get_satellite_info(dt, acquisition_df)
        
        # Combiner les métadonnées
        metadata_list.append({
            'date': dt,
            'satellite': sat_info['satellite'],
            'pass': sat_info['pass'],
            'angle_incidence': sat_info['angle_incidence'],
            'resolution_range': sat_info['resolution_range'],
            'resolution_azimuth': sat_info['resolution_azimuth'],
            'file': os.path.basename(file)
        })
    
    if len(tensor_data) == 0:
        return None, None, None, None, None, None
    
    # Calculer la forme moyenne
    mean_shape = np.mean(tensor_shape, axis=0, dtype=int)
    mean_geo = (tuple(np.mean(tensor_geo, axis=0)), geo[1])
    
    # Redimensionner les images
    tensor_img = np.array(
        [resize_image(img[:, :, 0], new_shape=mean_shape[:-1]) for img in tensor_data],
        dtype=np.float32,
    )
    tensor_img = np.where(tensor_img <= 0, nan, tensor_img)
    
    # Traiter les masques : nettoyer et convertir nodata
    mask_list = []
    for img in tensor_data:
        mask = resize_image(img[:, :, 1], new_shape=mean_shape[:-1], order=0)  # order=0 pour nearest neighbor
        
        # Nettoyer le masque : garder seulement 0, 1, 2, 3 et nodata
        # Convertir -999.0 en -127 (valeur spéciale pour int8)
        mask_cleaned = np.where(mask == nan, -127, mask)
        
        # Forcer les valeurs valides à être dans [0, 3]
        # Toute autre valeur devient -127 (nodata)
        valid_mask = np.isin(mask_cleaned, [0, 1, 2, 3, -127])
        mask_cleaned = np.where(valid_mask, mask_cleaned, -127)
        
        mask_list.append(mask_cleaned)
    
    tensor_mask = np.array(mask_list, dtype=np.int8)
    
    # Réorganiser: (time, height, width) -> (height, width, time)
    tensor_img = np.moveaxis(tensor_img, 0, -1)
    tensor_mask = np.moveaxis(tensor_mask, 0, -1)
    tensor_time = np.array(tensor_time, dtype='S8')
    
    return tensor_img, tensor_mask, tensor_time, metadata_list, mean_geo, mean_shape


def sort_by_time(tensor_img, tensor_mask, tensor_time, metadata_list):
    """Trie les tenseurs par ordre chronologique"""
    idx = np.argsort(tensor_time)
    tensor_img = tensor_img[:, :, idx]
    tensor_mask = tensor_mask[:, :, idx]
    tensor_time = tensor_time[idx]
    metadata_list = [metadata_list[i] for i in idx]
    return tensor_img, tensor_mask, tensor_time, metadata_list


def compute_statistics(data, nodata=-999):
    """Calcule les statistiques pour normalisation"""
    valid_data = data[data != nodata]
    
    if len(valid_data) == 0:
        return {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 1.0,
            'percentile_1': 0.0,
            'percentile_99': 1.0
        }
    
    return {
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'percentile_1': float(np.percentile(valid_data, 1)),
        'percentile_99': float(np.percentile(valid_data, 99))
    }


def create_optimized_dataset(output_file, data_dir, metadata_csv, acquisition_csv, nodata=-999.0):
    """
    Crée un dataset HDF5 optimisé pour le ML.
    
    Structure HDF5:
    /
    ├── metadata/
    │   ├── classes.json (liste des classes)
    │   ├── groups.json (liste des groupes par classe)
    │   ├── statistics.json (statistiques globales)
    │   └── acquisition_info (table des acquisitions)
    ├── data/
    │   ├── {GROUP_NAME}/
    │   │   ├── {ORBIT}/
    │   │   │   ├── {POLARISATION}/
    │   │   │   │   ├── images (H×W×T, chunked, compressed)
    │   │   │   │   ├── masks (H×W×T)
    │   │   │   │   ├── timestamps (T,)
    │   │   │   │   ├── satellite_codes (T,) ["PAZ", "TSX", "TDX"]
    │   │   │   │   ├── angles_incidence (T,)
    │   │   │   │   └── attributes: stats, geo, shape
    └── index/
        ├── by_satellite/ (index groupé par satellite)
        ├── by_class/ (index groupé par classe)
        ├── by_period/ (index par période temporelle)
        └── temporal_ranges (début/fin par groupe)
    """
    
    # Charger les métadonnées
    metadata_df = pd.read_csv(metadata_csv)
    acquisition_df = pd.read_csv(acquisition_csv)
    
    print("  Création du dataset HDF5 optimisé pour ML...")
    print(f"  Output: {output_file}")
    print(f"  Data directory: {data_dir}")
    
    with h5py.File(output_file, 'w') as hdf5_file:
        # Créer les groupes principaux
        meta_group = hdf5_file.create_group('metadata')
        data_group = hdf5_file.create_group('data')
        index_group = hdf5_file.create_group('index')
        
        # Structures pour les index
        satellite_index = defaultdict(list)
        class_index = defaultdict(list)
        temporal_ranges = {}
        
        # Parcourir les orbites
        for orbit in ['ASC', 'DSC']:
            orbit_path = os.path.join(data_dir, orbit)
            if not os.path.exists(orbit_path):
                continue
            
            class_folders = glob.glob(os.path.join(orbit_path, '*'))
            
            for class_folder in class_folders:
                class_name = os.path.basename(class_folder)
                group_folders = sorted(glob.glob(os.path.join(class_folder, '*')))
                
                for group_folder in tqdm(group_folders, desc=f"Processing {orbit}/{class_name}"):
                    group_name = os.path.basename(group_folder)
                    
                    # Récupérer les métadonnées géographiques
                    try:
                        group_num = group_name[-3:]
                        if group_num[0] == '0':
                            group_num = int(group_num)
                        else:
                            group_num = group_name
                        
                        meta_row = metadata_df.loc[
                            np.logical_and(
                                metadata_df['classe'] == class_name,
                                metadata_df['group'].astype(str).str.replace('_', '') == str(group_num)
                            )
                        ]
                        
                        if len(meta_row) == 0:
                            print(f"  ⚠️ Métadonnées manquantes pour {group_name}")
                            continue
                        
                        latitude, longitude, elevation, orientation, slope, class_id = meta_row[[
                            'latitude', 'longitude', 'altitude', 'exposition', 'pente', 'id'
                        ]].values[0]
                        
                    except Exception as e:
                        print(f"  ⚠️ Erreur métadonnées {group_name}: {e}")
                        continue
                    
                    # Créer le groupe dans HDF5
                    if group_name not in data_group:
                        group_hdf = data_group.create_group(group_name)
                        group_hdf.attrs['class'] = class_name
                        group_hdf.attrs['group_num'] = str(group_num)
                        group_hdf.attrs['latitude'] = latitude
                        group_hdf.attrs['longitude'] = longitude
                        group_hdf.attrs['elevation'] = elevation
                        group_hdf.attrs['orientation'] = orientation
                        group_hdf.attrs['slope'] = slope
                        group_hdf.attrs['id'] = int(class_id)
                        group_hdf.attrs['nodata'] = nodata
                    else:
                        group_hdf = data_group[group_name]
                    
                    # Créer le sous-groupe pour l'orbite
                    orbit_hdf = group_hdf.create_group(orbit)
                    orbit_hdf.attrs['orbit'] = 'ascending' if orbit == 'ASC' else 'descending'
                    
                    # Traiter chaque polarisation
                    for polarisation in ['HH', 'HV']:
                        result = open_files_with_metadata(
                            group_folder, polarisation, acquisition_df, nan=nodata
                        )
                        
                        if result[0] is None:
                            continue
                        
                        data, mask, timestamps, metadata_list, geoinfo, shape_data = result
                        
                        # Trier par ordre chronologique
                        data, mask, timestamps, metadata_list = sort_by_time(
                            data, mask, timestamps, metadata_list
                        )
                        
                        # Créer le groupe de polarisation
                        pol_hdf = orbit_hdf.create_group(polarisation)
                        
                        # Stocker les données avec chunking optimisé pour ML
                        # Chunks: accès à une image complète à un instant donné
                        chunk_shape = (data.shape[0], data.shape[1], 1)
                        
                        pol_hdf.create_dataset(
                            'images',
                            data=data,
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=chunk_shape
                        )
                        
                        pol_hdf.create_dataset(
                            'masks',
                            data=mask,
                            dtype=np.int8,
                            compression='gzip',
                            compression_opts=4,
                            chunks=chunk_shape
                        )
                        
                        pol_hdf.create_dataset('timestamps', data=timestamps)
                        
                        # Extraire et stocker les métadonnées satellite
                        satellites = np.array([m['satellite'] for m in metadata_list], dtype='S20')
                        passes = np.array([m['pass'] for m in metadata_list], dtype='S20')
                        angles = np.array([m['angle_incidence'] for m in metadata_list], dtype=np.float32)
                        res_range = np.array([m['resolution_range'] for m in metadata_list], dtype=np.float32)
                        res_azimuth = np.array([m['resolution_azimuth'] for m in metadata_list], dtype=np.float32)
                        
                        pol_hdf.create_dataset('satellites', data=satellites)
                        pol_hdf.create_dataset('passes', data=passes)
                        pol_hdf.create_dataset('angles_incidence', data=angles)
                        pol_hdf.create_dataset('resolution_range', data=res_range)
                        pol_hdf.create_dataset('resolution_azimuth', data=res_azimuth)
                        
                        # Calculer et stocker les statistiques
                        stats = compute_statistics(data, nodata)
                        for key, value in stats.items():
                            pol_hdf.attrs[f'stat_{key}'] = value
                        
                        # Attributs géographiques
                        # Convertir le CRS en string pour HDF5
                        crs_str = str(geoinfo[1]) if geoinfo[1] is not None else 'EPSG:32631'
                        pol_hdf.attrs['geo_projection'] = crs_str
                        pol_hdf.attrs['geo_transform'] = geoinfo[0]
                        pol_hdf.attrs['shape_original'] = shape_data
                        pol_hdf.attrs['n_timestamps'] = len(timestamps)
                        
                        # Construire les index
                        path = f"{group_name}/{orbit}/{polarisation}"
                        
                        # Index par satellite
                        for i, sat in enumerate(satellites):
                            sat_name = sat.decode('utf-8')
                            satellite_index[sat_name].append({
                                'path': path,
                                'timestamp_idx': i,
                                'timestamp': timestamps[i].decode('utf-8'),
                                'class': class_name,
                                'group': group_name
                            })
                        
                        # Index par classe
                        class_index[class_name].append({
                            'path': path,
                            'group': group_name,
                            'orbit': orbit,
                            'polarisation': polarisation,
                            'n_samples': len(timestamps)
                        })
                        
                        # Ranges temporels
                        temporal_ranges[path] = {
                            'start': timestamps[0].decode('utf-8'),
                            'end': timestamps[-1].decode('utf-8'),
                            'n_samples': len(timestamps)
                        }
        
        # Sauvegarder les index
        print("\n Création des index...")
        
        # Index par satellite
        sat_index_group = index_group.create_group('by_satellite')
        for sat_name, entries in satellite_index.items():
            sat_group = sat_index_group.create_group(sat_name)
            sat_group.attrs['n_entries'] = len(entries)
            sat_group.attrs['entries_json'] = json.dumps(entries)
        
        # Index par classe
        class_index_group = index_group.create_group('by_class')
        for class_name, entries in class_index.items():
            cls_group = class_index_group.create_group(class_name)
            cls_group.attrs['n_groups'] = len(entries)
            cls_group.attrs['entries_json'] = json.dumps(entries)
        
        # Ranges temporels
        temp_range_group = index_group.create_group('temporal_ranges')
        temp_range_group.attrs['ranges_json'] = json.dumps(temporal_ranges)
        
        # Métadonnées globales
        meta_group.attrs['classes'] = json.dumps(list(class_index.keys()))
        meta_group.attrs['satellites'] = json.dumps(list(satellite_index.keys()))
        meta_group.attrs['n_total_groups'] = len(temporal_ranges)
        meta_group.attrs['creation_date'] = datetime.now().isoformat()
        meta_group.attrs['nodata_value'] = nodata
        
        print(f"\nDataset créé avec succès!")
        print(f"  {len(temporal_ranges)} groupes traités")
        print(f"  {len(class_index)} classes: {list(class_index.keys())}")
        print(f"  {len(satellite_index)} satellites: {list(satellite_index.keys())}")



def generate_txt_report(json_file, output_txt):
    """Génère un rapport texte lisible depuis le JSON détaillé."""
    
    with open(json_file, 'r') as f:
        report = json.load(f)
    
    with open(output_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT DETAILLE DU DATASET PAZTSX_CRYO_ML\n")
        f.write("="*80 + "\n\n")
        
        # Métadonnées globales
        meta = report['metadata']
        f.write("METADONNEES GLOBALES:\n")
        f.write(f"  Classes: {', '.join(meta['classes'])}\n")
        f.write(f"  Satellites: {', '.join(meta['satellites'])}\n")
        f.write(f"  Nombre total de groupes: {meta['n_total_groups']}\n")
        f.write(f"  Date de creation: {meta['creation_date']}\n")
        f.write(f"  Valeur nodata: {meta['nodata_value']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILS PAR GROUPE\n")
        f.write("="*80 + "\n\n")
        
        # Parcourir tous les groupes
        for group_name, group_data in sorted(report['groups'].items()):
            f.write(f"Groupe: {group_data['group_name']}\n")
            f.write(f"Classe: {group_data['class']}\n")
            f.write(f"Position: ({group_data['latitude']:.4f}, {group_data['longitude']:.4f})\n")
            f.write(f"Altitude: {group_data['elevation']:.1f} m\n")
            f.write(f"Orientation: {group_data['orientation']:.1f} deg\n")
            f.write(f"Pente: {group_data['slope']:.1f} deg\n")
            
            # Parcourir les orbites
            for orbit_name in ['ASC', 'DSC']:
                if orbit_name in group_data['orbits']:
                    orbit_data = group_data['orbits'][orbit_name]
                    f.write(f"{orbit_name}:\n")
                    
                    # Parcourir les polarisations
                    for pol_name in ['HH', 'HV']:
                        if pol_name in orbit_data:
                            pol_data = orbit_data[pol_name]
                            f.write(f"  {pol_name}: {tuple(pol_data['shape'])} - {pol_data['n_acquisitions']} acquisitions\n")
                            f.write(f"    Stats: mean={pol_data['stats']['mean']:.2f}, std={pol_data['stats']['std']:.2f}\n")
                            f.write(f"    NaN: {pol_data['nan_info']['percentage']:.2f}% ({pol_data['nan_info']['count']} pixels)\n")
                            f.write(f"    Dates: {pol_data['date_min']} -> {pol_data['date_max']}\n")
                            
                            # Plages par satellite
                            if pol_data['date_ranges']:
                                for sat_name in ['PAZ', 'TerraSAR-X', 'TanDEM-X']:
                                    if sat_name in pol_data['date_ranges']:
                                        dates = pol_data['date_ranges'][sat_name]
                                        f.write(f"      {sat_name}: {dates['min']} -> {dates['max']} ({dates['count']} acq)\n")
                            
                            # Statistiques des masques (simplifié - seulement classes principales)
                            mask_stats = pol_data['mask_statistics']
                            f.write(f"    Classes masques (principales):\n")
                            
                            # Trier par pourcentage décroissant et prendre les 5 premiers
                            sorted_classes = sorted(mask_stats.items(), 
                                                  key=lambda x: x[1]['percentage'], 
                                                  reverse=True)[:5]
                            
                            for class_name, stats in sorted_classes:
                                f.write(f"      {class_name}: {stats['percentage']:.2f}% ({stats['count']} pixels)\n")
            
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"Rapport texte sauvegarde: {output_txt}")


def print_dataset_summary(hdf5_file):
    """Affiche un résumé du dataset créé"""
    print("\n" + "="*60)
    print("RESUME DU DATASET")
    print("="*60)
    
    with h5py.File(hdf5_file, 'r') as f:
        # Métadonnées globales
        meta = f['metadata']
        print(f"\nClasses: {json.loads(meta.attrs['classes'])}")
        print(f"Satellites: {json.loads(meta.attrs['satellites'])}")
        print(f"Nombre de groupes: {meta.attrs['n_total_groups']}")
        print(f"Cree le: {meta.attrs['creation_date']}")
        
        # Exemple de données
        print("\nExemple de structure (premier groupe):")
        data_group = f['data']
        first_group = list(data_group.keys())[0]
        print(f"  Groupe: {first_group}")
        
        group = data_group[first_group]
        print(f"    Classe: {group.attrs['class']}")
        print(f"    Position: ({group.attrs['latitude']:.4f}, {group.attrs['longitude']:.4f})")
        print(f"    Altitude: {group.attrs['elevation']:.1f} m")
        
        for orbit in group.keys():
            print(f"    {orbit}:")
            for pol in group[orbit].keys():
                pol_data = group[orbit][pol]
                print(f"      {pol}: {pol_data['images'].shape} - {pol_data.attrs['n_timestamps']} acquisitions")
                print(f"        Stats: mean={pol_data.attrs['stat_mean']:.2f}, std={pol_data.attrs['stat_std']:.2f}")


if __name__ == "__main__":
    # Configuration
    output_file = "../DATASET/PAZTSX_CRYO_ML.hdf5"
    data_dir = "../ORIGINAL_DATA/"
    metadata_csv = "../ORIGINAL_DATA/metadata/details_shp.csv"
    acquisition_csv = "../ORIGINAL_DATA/metadata/data.csv"
    nodata = -999.0
    
    # Créer le dataset
    create_optimized_dataset(
        output_file=output_file,
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        acquisition_csv=acquisition_csv,
        nodata=nodata
    )
    
    # Afficher le résumé
    print_dataset_summary(output_file)
    
    # Générer le rapport texte lisible
    txt_report = output_file.replace('.hdf5', '_report.txt')
    json_report = output_file.replace('.hdf5', '_detailed_report.json')
    generate_txt_report(json_report, txt_report)
    
    print("\n" + "="*60)
    print("TERMINE!")
    print("="*60)
