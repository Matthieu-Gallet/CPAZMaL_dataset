#!/usr/bin/env python3
"""
Optimized script to create a structured HDF5 dataset for Machine Learning.

Improvements compared to 2_create_dataset.py:
1. Optimized HDF5 structure with indexes for fast access
2. Enriched metadata (resolution, incidence angle)
3. Organization by time period
4. Chunking and compression optimized for ML
5. Pre-calculated normalization attributes
6. Temporal indexes for fast sequence extraction
"""

# GDAL configuration to avoid warnings and PROJ errors
import os
os.environ['GDAL_PAM_ENABLED'] = 'NO'

# Force the use of pyproj's PROJ database (more recent)
import sys
import pyproj
proj_data_dir = pyproj.datadir.get_data_dir()
os.environ['PROJ_DATA'] = proj_data_dir
os.environ['PROJ_LIB'] = proj_data_dir
print(f"Using PROJ data from: {proj_data_dir}")


from datetime import datetime
import h5py
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Import utility functions
from src.utils.processing_utils import resize_image, load_data_rasterio


def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename.
    Format: PA_ABL001_HH_20200110.tif or TX_ABL001_HH_20071024.tif
    
    Returns:
        dict: group, polarisation, date
    """
    basename = os.path.basename(filename)
    parts = basename.replace(".tif", "").split("_")
    
    return {
        'group': parts[1],
        'polarisation': parts[2],
        'date': parts[3]
    }


def get_satellite_info(date_str, acquisition_df):
    """
    Retrieve satellite information from data.csv for a given date.
    
    Args:
        date_str: Date in YYYYMMDD format
        acquisition_df: DataFrame with acquisition data
    
    Returns:
        dict: Satellite information (name, angle, resolution, orbit)
    """
    # Convert date_str YYYYMMDD to comparable format
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    date_formatted = date_obj.strftime('%Y-%m-%d')
    
    # Search in DataFrame
    match = acquisition_df[acquisition_df['Date'] == date_formatted]
    
    if len(match) > 0:
        row = match.iloc[0]
        return {
            'pass': row['Pass'],
            'angle_incidence': float(row['Angle Incidence'].replace('°', '')),
            'resolution_range': float(row['Résolution Range'].replace(' m', '')),
            'resolution_azimuth': float(row['Résolution Azimuth'].replace(' m', ''))
        }
    else:
        # Default values if not found
        return {
            'pass': 'Unknown',
            'angle_incidence': 0.0,
            'resolution_range': 0.0,
            'resolution_azimuth': 0.0
        }


def open_files_with_metadata(group_folder, polarisation, acquisition_df, nan=-999):
    """
    Load files with enriched metadata from data.csv
    
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
        # Load image with rasterio
        img, geo = load_data_rasterio(file)
        tensor_data.append(img)
        tensor_shape.append(img.shape)
        tensor_geo.append(geo[0])
        
        # Extract metadata from filename
        file_meta = extract_metadata_from_filename(file)
        dt = file_meta['date']
        tensor_time.append(dt)
        
        # Retrieve satellite info from CSV
        sat_info = get_satellite_info(dt, acquisition_df)
        
        # Combine metadata
        metadata_list.append({
            'date': dt,
            'pass': sat_info['pass'],
            'angle_incidence': sat_info['angle_incidence'],
            'resolution_range': sat_info['resolution_range'],
            'resolution_azimuth': sat_info['resolution_azimuth'],
            'file': os.path.basename(file)
        })
    
    if len(tensor_data) == 0:
        return None, None, None, None, None, None
    
    # Calculate mean shape
    mean_shape = np.mean(tensor_shape, axis=0, dtype=int)
    mean_geo = (tuple(np.mean(tensor_geo, axis=0)), geo[1])
    
    # Resize images
    resized_images = []
    for img in tensor_data:
        img_channel = img[:, :, 0].copy()
        # Replace problematic values BEFORE resize to avoid NaN propagation
        img_channel = np.where(np.isfinite(img_channel), img_channel, nan)
        img_channel = np.where(img_channel <= 0, nan, img_channel)
        resized = resize_image(img_channel, new_shape=mean_shape[:-1])
        resized_images.append(resized)
    
    tensor_img = np.array(resized_images, dtype=np.float32)
    
    # Process masks: clean and convert nodata
    mask_list = []
    for img in tensor_data:
        mask = resize_image(img[:, :, 1], new_shape=mean_shape[:-1], order=0)  # order=0 for nearest neighbor
        
        # Clean mask: keep only 0, 1, 2, 3 and nodata
        # Convert -999.0 to -127 (special value for int8)
        mask_cleaned = np.where(mask == nan, -127, mask)
        
        # Force valid values to be in [0, 3]
        # Any other value becomes -127 (nodata)
        valid_mask = np.isin(mask_cleaned, [0, 1, 2, 3, -127])
        mask_cleaned = np.where(valid_mask, mask_cleaned, -127)
        
        mask_list.append(mask_cleaned)
    
    tensor_mask = np.array(mask_list, dtype=np.int8)
    
    # Reorganize: (time, height, width) -> (height, width, time)
    tensor_img = np.moveaxis(tensor_img, 0, -1)
    tensor_mask = np.moveaxis(tensor_mask, 0, -1)
    tensor_time = np.array(tensor_time, dtype='S8')
    
    return tensor_img, tensor_mask, tensor_time, metadata_list, mean_geo, mean_shape


def sort_by_time(tensor_img, tensor_mask, tensor_time, metadata_list):
    """Sort tensors by chronological order"""
    idx = np.argsort(tensor_time)
    tensor_img = tensor_img[:, :, idx]
    tensor_mask = tensor_mask[:, :, idx]
    tensor_time = tensor_time[idx]
    metadata_list = [metadata_list[i] for i in idx]
    return tensor_img, tensor_mask, tensor_time, metadata_list


def compute_statistics(data, nodata=-999):
    """
    Calculate statistics for normalization.
    Filter nodata values (with tolerance), NaN and infinities.

    Uses an absolute tolerance when comparing to `nodata` to account for
    small perturbations introduced by resampling/interpolation.
    """
    # Mask out non-finite values and values close to nodata
    nodata_mask = np.isclose(data, nodata, atol=5)
    valid_mask = np.isfinite(data) & (~nodata_mask)
    valid_data = data[valid_mask]

    if valid_data.size == 0:
        # No valid data: return safe defaults
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
    Create an optimized HDF5 dataset for ML.
    
    HDF5 Structure:
    /
    ├── metadata/
    │   ├── classes.json (list of classes)
    │   ├── groups.json (list of groups per class)
    │   ├── statistics.json (global statistics)
    │   └── acquisition_info (acquisition table)
    ├── data/
    │   ├── {GROUP_NAME}/
    │   │   ├── {ORBIT}/
    │   │   │   ├── {POLARISATION}/
    │   │   │   │   ├── images (H×W×T, chunked, compressed)
    │   │   │   │   ├── masks (H×W×T)
    │   │   │   │   ├── timestamps (T,)
    │   │   │   │   ├── angles_incidence (T,)
    │   │   │   │   └── attributes: stats, geo, shape
    └── index/
        ├── by_class/ (class-grouped index)
        ├── by_period/ (period index)
        └── temporal_ranges (start/end per group)
    """
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv)
    acquisition_df = pd.read_csv(acquisition_csv)
    
    print("  Creating optimized HDF5 dataset for ML...")
    print(f"  Output: {output_file}")
    print(f"  Data directory: {data_dir}")
    
    with h5py.File(output_file, 'w') as hdf5_file:
        # Create main groups
        meta_group = hdf5_file.create_group('metadata')
        data_group = hdf5_file.create_group('data')
        index_group = hdf5_file.create_group('index')
        
        # Structures for indexes
        class_index = defaultdict(list)
        temporal_ranges = {}
        
        # Browse orbits
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
                    
                    # Retrieve geographical metadata
                    try:
                        # Extract group number (last 3 characters) or full name
                        # Groups in CSV are in format '001', '002', etc. (3 digits with zeros)
                        # or special names like 'ALL_GLACIERS', 'ARGENTIER_TOP'
                        
                        # Mapping for truncated names -> full names
                        special_names = {
                            'ALLGLA': 'ALLGLACIERS',
                            'ARGTOP': 'ARGENTIERTOP'
                        }
                        
                        if len(group_name) >= 6 and group_name[-3:].isdigit():
                            # Normal case: FOR006 -> '006'
                            group_num = group_name[-3:]
                        elif group_name in special_names:
                            # Special cases: ALLGLA -> 'ALLGLACIERS'
                            group_num = special_names[group_name]
                        else:
                            # Other cases
                            group_num = group_name
                        
                        meta_row = metadata_df.loc[
                            np.logical_and(
                                metadata_df['classe'] == class_name,
                                metadata_df['group'].astype(str).str.replace('_', '') == group_num
                            )
                        ]
                        
                        if len(meta_row) == 0:
                            print(f"  ⚠️ Missing metadata for {group_name}")
                            continue
                        
                        latitude, longitude, elevation, orientation, slope, class_id = meta_row[[
                            'latitude', 'longitude', 'altitude', 'exposition', 'pente', 'id'
                        ]].values[0]
                        
                    except Exception as e:
                        print(f"  ⚠️ Metadata error {group_name}: {e}")
                        continue
                    
                    # Create group in HDF5
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
                    
                    # Create orbit subgroup
                    orbit_hdf = group_hdf.create_group(orbit)
                    orbit_hdf.attrs['orbit'] = 'ascending' if orbit == 'ASC' else 'descending'
                    
                    # Process each polarisation
                    for polarisation in ['HH', 'HV']:
                        result = open_files_with_metadata(
                            group_folder, polarisation, acquisition_df, nan=nodata
                        )
                        
                        if result[0] is None:
                            continue
                        
                        data, mask, timestamps, metadata_list, geoinfo, shape_data = result
                        
                        # Sort by chronological order
                        data, mask, timestamps, metadata_list = sort_by_time(
                            data, mask, timestamps, metadata_list
                        )
                        
                        # Create polarisation group
                        pol_hdf = orbit_hdf.create_group(polarisation)
                        
                        # Store data with ML-optimized chunking
                        # Chunks: access to a complete image at a given time
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
                        
                        # Extract and store satellite metadata
                        passes = np.array([m['pass'] for m in metadata_list], dtype='S20')
                        angles = np.array([m['angle_incidence'] for m in metadata_list], dtype=np.float32)
                        res_range = np.array([m['resolution_range'] for m in metadata_list], dtype=np.float32)
                        res_azimuth = np.array([m['resolution_azimuth'] for m in metadata_list], dtype=np.float32)
                        
                        pol_hdf.create_dataset('passes', data=passes)
                        pol_hdf.create_dataset('angles_incidence', data=angles)
                        pol_hdf.create_dataset('resolution_range', data=res_range)
                        pol_hdf.create_dataset('resolution_azimuth', data=res_azimuth)
                        
                        # Calculate and store statistics
                        # Debug: check values before stats calculation
                        n_total = data.size
                        # nodata exact match (may be zero if resampling altered values)
                        n_nodata_exact = int(np.sum(data == nodata))
                        # nodata approximate match (to catch -998.99 etc. after interpolation)
                        n_nodata_close = int(np.sum(np.isclose(data, nodata, atol=0.5)))
                        n_nan = int(np.sum(np.isnan(data)))
                        # Consider valid those that are finite and not close to nodata
                        n_valid = int(np.sum(np.isfinite(data) & (~np.isclose(data, nodata, atol=0.5))))

                        if n_valid == 0:
                            print(f"         {group_name}/{orbit}/{polarisation}: No valid data!")
                            print(f"         Total pixels: {n_total}, nodata_exact: {n_nodata_exact}, nodata_approx: {n_nodata_close}, NaN: {n_nan}")

                        stats = compute_statistics(data, nodata)
                        for key, value in stats.items():
                            pol_hdf.attrs[f'stat_{key}'] = value
                        
                        # Geographical attributes
                        # Convert CRS to string for HDF5
                        crs_str = str(geoinfo[1]) if geoinfo[1] is not None else 'EPSG:32631'
                        pol_hdf.attrs['geo_projection'] = crs_str
                        pol_hdf.attrs['geo_transform'] = geoinfo[0]
                        pol_hdf.attrs['shape_original'] = shape_data
                        pol_hdf.attrs['n_timestamps'] = len(timestamps)
                        
                        # Build indexes
                        path = f"{group_name}/{orbit}/{polarisation}"
                        
   
                        # Index by class
                        class_index[class_name].append({
                            'path': path,
                            'group': group_name,
                            'orbit': orbit,
                            'polarisation': polarisation,
                            'n_samples': len(timestamps)
                        })
                        
                        # Temporal ranges
                        temporal_ranges[path] = {
                            'start': timestamps[0].decode('utf-8'),
                            'end': timestamps[-1].decode('utf-8'),
                            'n_samples': len(timestamps)
                        }
        
        # Save indexes
        print("\n Creating indexes...")
        

        
        # Index by class
        class_index_group = index_group.create_group('by_class')
        for class_name, entries in class_index.items():
            cls_group = class_index_group.create_group(class_name)
            cls_group.attrs['n_groups'] = len(entries)
            cls_group.attrs['entries_json'] = json.dumps(entries)
        
        # Temporal ranges
        temp_range_group = index_group.create_group('temporal_ranges')
        temp_range_group.attrs['ranges_json'] = json.dumps(temporal_ranges)
        
        # Global metadata
        meta_group.attrs['classes'] = json.dumps(list(class_index.keys()))
        meta_group.attrs['n_total_groups'] = len(temporal_ranges)
        meta_group.attrs['creation_date'] = datetime.now().isoformat()
        meta_group.attrs['nodata_value'] = nodata
        
        print(f"\nDataset created successfully!")
        print(f"  {len(temporal_ranges)} groups processed")
        print(f"  {len(class_index)} classes: {list(class_index.keys())}")



def generate_txt_report(json_file, output_txt):
    """Generate a readable text report from the detailed JSON."""
    
    with open(json_file, 'r') as f:
        report = json.load(f)
    
    with open(output_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED REPORT OF PAZ_CRYO_ML DATASET\n")
        f.write("="*80 + "\n\n")
        
        # Global metadata
        meta = report['metadata']
        f.write("GLOBAL METADATA:\n")
        f.write(f"  Classes: {', '.join(meta['classes'])}\n")
        f.write(f"  Total number of groups: {meta['n_total_groups']}\n")
        f.write(f"  Creation date: {meta['creation_date']}\n")
        f.write(f"  Nodata value: {meta['nodata_value']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILS BY GROUP\n")
        f.write("="*80 + "\n\n")
        
        # Browse all groups
        for group_name, group_data in sorted(report['groups'].items()):
            f.write(f"Group: {group_data['group_name']}\n")
            f.write(f"Class: {group_data['class']}\n")
            f.write(f"Position: ({group_data['latitude']:.4f}, {group_data['longitude']:.4f})\n")
            f.write(f"Elevation: {group_data['elevation']:.1f} m\n")
            f.write(f"Orientation: {group_data['orientation']:.1f} deg\n")
            f.write(f"Slope: {group_data['slope']:.1f} deg\n")
            
            # Browse orbits
            for orbit_name in ['ASC', 'DSC']:
                if orbit_name in group_data['orbits']:
                    orbit_data = group_data['orbits'][orbit_name]
                    f.write(f"{orbit_name}:\n")
                    
                    # Browse polarisations
                    for pol_name in ['HH', 'HV']:
                        if pol_name in orbit_data:
                            pol_data = orbit_data[pol_name]
                            f.write(f"  {pol_name}: {tuple(pol_data['shape'])} - {pol_data['n_acquisitions']} acquisitions\n")
                            f.write(f"    Stats: mean={pol_data['stats']['mean']:.2f}, std={pol_data['stats']['std']:.2f}\n")
                            f.write(f"    NaN: {pol_data['nan_info']['percentage']:.2f}% ({pol_data['nan_info']['count']} pixels)\n")
                            f.write(f"    Dates: {pol_data['date_min']} -> {pol_data['date_max']}\n")
                            
                            # Ranges by satellite
                            if pol_data['date_ranges']:
                                for sat_name in ['PAZ']:
                                    if sat_name in pol_data['date_ranges']:
                                        dates = pol_data['date_ranges'][sat_name]
                                        f.write(f"      {sat_name}: {dates['min']} -> {dates['max']} ({dates['count']} acq)\n")
                            
                            # Mask statistics (simplified - only main classes)
                            mask_stats = pol_data['mask_statistics']
                            f.write(f"    Mask classes (main):\n")
                            
                            # Sort by decreasing percentage and take first 5
                            sorted_classes = sorted(mask_stats.items(), 
                                                  key=lambda x: x[1]['percentage'], 
                                                  reverse=True)[:5]
                            
                            for class_name, stats in sorted_classes:
                                f.write(f"      {class_name}: {stats['percentage']:.2f}% ({stats['count']} pixels)\n")
            
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"Text report saved: {output_txt}")


def save_dataset_summary_to_txt(hdf5_file, output_txt):
    """Save a summary of the created dataset to a text file"""
    
    with open(output_txt, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("="*60 + "\n")
        
        with h5py.File(hdf5_file, 'r') as hdf5_f:
            # Global metadata
            meta = hdf5_f['metadata']
            f.write(f"\nClasses: {json.loads(meta.attrs['classes'])}\n")
            f.write(f"Number of groups: {meta.attrs['n_total_groups']}\n")
            f.write(f"Created on: {meta.attrs['creation_date']}\n")
            
            # Example data
            f.write("\nExample structure (first group):\n")
            data_group = hdf5_f['data']
            first_group = list(data_group.keys())[0]
            f.write(f"  Group: {first_group}\n")
            
            group = data_group[first_group]
            f.write(f"    Class: {group.attrs['class']}\n")
            f.write(f"    Position: ({group.attrs['latitude']:.4f}, {group.attrs['longitude']:.4f})\n")
            f.write(f"    Elevation: {group.attrs['elevation']:.1f} m\n")
            
            for orbit in group.keys():
                f.write(f"    {orbit}:\n")
                for pol in group[orbit].keys():
                    pol_data = group[orbit][pol]
                    f.write(f"      {pol}: {pol_data['images'].shape} - {pol_data.attrs['n_timestamps']} acquisitions\n")
                    f.write(f"        Stats: min={pol_data.attrs['stat_min']:.2f}, max={pol_data.attrs['stat_max']:.2f}, mean={pol_data.attrs['stat_mean']:.2f}, std={pol_data.attrs['stat_std']:.2f}\n")
                    f.write(f"        Percentiles: p1={pol_data.attrs['stat_percentile_1']:.2f}, p99={pol_data.attrs['stat_percentile_99']:.2f}\n")
    
    print(f"Dataset summary saved to: {output_txt}")


if __name__ == "__main__":
    
    # Configuration
    output_file = "DATASET/CPAZMaL.hdf5"
    data_dir = "ORIGINAL_DATA/"
    metadata_csv = "ORIGINAL_DATA/metadata/shapefile_statistics.csv"
    acquisition_csv = "ORIGINAL_DATA/metadata/data.csv"
    nodata = -999.0
    
    # Create the dataset
    create_optimized_dataset(
        output_file=output_file,
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        acquisition_csv=acquisition_csv,
        nodata=nodata
    )
    
    # Save the summary to text file
    summary_txt = output_file.replace('.hdf5', '_summary.txt')
    save_dataset_summary_to_txt(output_file, summary_txt)
    

    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

