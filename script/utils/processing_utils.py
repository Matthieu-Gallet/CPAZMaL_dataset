#!/usr/bin/env python3
"""
Utilitaires de traitement pour les datasets SAR.
Utilise rasterio au lieu de GDAL pour éviter les warnings.
"""

import numpy as np
import rasterio
from scipy.ndimage import zoom


def resize_image(array, new_shape=(100, 100), order=1):
    """
    Redimensionne une image avec interpolation.
    
    Args:
        array: Image à redimensionner (2D ou 3D)
        new_shape: Nouvelle taille (height, width)
        order: Ordre d'interpolation (0=nearest, 1=bilinear, 3=cubic)
    
    Returns:
        np.ndarray: Image redimensionnée
    """
    zoom_factors = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])
    resized_array = zoom(array, zoom_factors, order=order)
    return resized_array


def load_data_rasterio(file_path):
    """
    Charge un fichier raster avec rasterio.
    
    Args:
        file_path: Chemin vers le fichier GeoTIFF
    
    Returns:
        tuple: (data_array, geo_info)
            - data_array: (height, width, n_bands)
            - geo_info: (transform, crs)
    """
    with rasterio.open(file_path) as src:
        # Lire toutes les bandes
        data = src.read()  # Shape: (n_bands, height, width)
        
        # Réorganiser en (height, width, n_bands)
        data = np.moveaxis(data, 0, -1)
        
        # Récupérer les informations géographiques
        transform = src.transform
        crs = src.crs
        
        geo_info = (transform, crs)
    
    return data, geo_info


def save_geotiff_rasterio(file_path, data, transform, crs, nodata=None):
    """
    Sauvegarde un GeoTIFF avec rasterio.
    
    Args:
        file_path: Chemin de sortie
        data: Array à sauvegarder (height, width) ou (height, width, n_bands)
        transform: Affine transform de rasterio
        crs: Système de coordonnées
        nodata: Valeur nodata
    """
    # Gérer les dimensions
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # Ajouter dimension band
        count = 1
    else:
        data = np.moveaxis(data, -1, 0)  # (H,W,C) -> (C,H,W)
        count = data.shape[0]
    
    height, width = data.shape[1], data.shape[2]
    
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data)


def compute_mask_statistics(mask):
    """
    Calcule les statistiques d'un masque de classification.
    
    Args:
        mask: Array 2D ou 3D avec des valeurs entières représentant les classes
              Valeurs attendues: 0, 1, 2, 3 (classes valides) et -127 (nodata)
    
    Returns:
        dict: Pourcentage de pixels par classe
    """
    unique_values, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    stats = {}
    for value, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        
        # Nommer les classes de manière claire
        if value == -127:
            class_name = 'nodata'
        elif value in [0, 1, 2, 3]:
            class_name = f'class_{int(value)}'
        else:
            class_name = f'unknown_{int(value)}'
        
        stats[class_name] = {
            'count': int(count),
            'percentage': float(percentage)
        }
    
    return stats


def get_date_ranges_by_satellite(timestamps, satellites):
    """
    Calcule les plages de dates par satellite.
    
    Args:
        timestamps: Array de timestamps (format YYYYMMDD en bytes)
        satellites: Array de noms de satellites (en bytes)
    
    Returns:
        dict: Plages de dates par satellite
    """
    from datetime import datetime
    
    # Décoder les arrays
    timestamps_str = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]
    satellites_str = [sat.decode('utf-8') if isinstance(sat, bytes) else sat for sat in satellites]
    
    # Grouper par satellite
    sat_dates = {}
    for sat, ts in zip(satellites_str, timestamps_str):
        if sat not in sat_dates:
            sat_dates[sat] = []
        sat_dates[sat].append(ts)
    
    # Calculer min/max pour chaque satellite
    ranges = {}
    for sat, dates in sat_dates.items():
        if dates:
            ranges[sat] = {
                'min': min(dates),
                'max': max(dates),
                'count': len(dates)
            }
    
    return ranges


def analyze_group_data(group_hdf):
    """
    Analyse complète d'un groupe HDF5.
    
    Args:
        group_hdf: Groupe HDF5 contenant les données
    
    Returns:
        dict: Informations détaillées du groupe
    """
    info = {
        'group_name': group_hdf.name.split('/')[-1],
        'class': group_hdf.attrs.get('class', 'Unknown'),
        'latitude': float(group_hdf.attrs.get('latitude', 0.0)),
        'longitude': float(group_hdf.attrs.get('longitude', 0.0)),
        'elevation': float(group_hdf.attrs.get('elevation', 0.0)),
        'orientation': float(group_hdf.attrs.get('orientation', 0.0)),
        'slope': float(group_hdf.attrs.get('slope', 0.0)),
        'orbits': {}
    }
    
    # Parcourir les orbites
    for orbit_name in group_hdf.keys():
        orbit_hdf = group_hdf[orbit_name]
        info['orbits'][orbit_name] = {}
        
        # Parcourir les polarisations
        for pol_name in orbit_hdf.keys():
            pol_hdf = orbit_hdf[pol_name]
            
            # Données de base
            images = pol_hdf['images'][:]
            masks = pol_hdf['masks'][:]
            timestamps = pol_hdf['timestamps'][:]
            satellites = pol_hdf['satellites'][:]
            
            # Statistiques
            stats_mean = pol_hdf.attrs.get('stat_mean', 0.0)
            stats_std = pol_hdf.attrs.get('stat_std', 0.0)
            
            # Compter les NaN
            nodata = group_hdf.attrs.get('nodata', -999.0)
            n_nan = np.sum(images == nodata)
            total_pixels = images.size
            nan_percentage = (n_nan / total_pixels) * 100
            
            # Plages de dates par satellite
            date_ranges = get_date_ranges_by_satellite(timestamps, satellites)
            
            # Statistiques des masques
            mask_stats = compute_mask_statistics(masks)
            
            info['orbits'][orbit_name][pol_name] = {
                'shape': images.shape,
                'n_acquisitions': len(timestamps),
                'stats': {
                    'mean': float(stats_mean),
                    'std': float(stats_std)
                },
                'nan_info': {
                    'count': int(n_nan),
                    'percentage': float(nan_percentage)
                },
                'date_ranges': date_ranges,
                'mask_statistics': mask_stats
            }
    
    return info
