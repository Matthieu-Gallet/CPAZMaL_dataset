from src.utils.geo_tools import *
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
    Extract a unique key to identify similar acquisitions.
    Key = (satellite, date, hour, acq_type) without minutes
    """
    basename = os.path.basename(filename)
    # Examples: 
    # - subset_20_of_TDX1_SAR_SSC_SM_S_SRA_20131114T172538_20131114T172543_Cal_TF_TC.tif
    # - subset_0_of_PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536_Cal_Tf_Tc_TC.tif
    # - subset_0_of_TSX1_SAR_SSC_SM_S_SRA_20091010T172514_20091010T172522_Cal_TF_TC.tif
    # Looking for: satellite (TDX1/TSX1/PAZ1), date (YYYYMMDD), start hour (HH)
    match = re.search(r"([TP][DASX][ZX]\d).*?(\d{8})T(\d{2})\d{4}_\d{8}T\d{6}", basename)
    if match:
        satellite = match.group(1)  # TDX1, TSX1 or PAZ1
        date = match.group(2)       # 20131114
        hour = match.group(3)       # 17
        return (satellite, date, hour)
    return None


def smart_merge_images(file_list, output_path):
    """
    Intelligently merges multiple images by handling zero areas.
    - If both pixels are non-zero: average
    - If only one is non-zero: take the non-zero one
    - If both are zero: 0
    """
    # Open all images
    datasets = [gdal.Open(f, gdal.GA_ReadOnly) for f in file_list]
    
    # Get georeferencing information from the first image
    first_ds = datasets[0]
    geotransform = first_ds.GetGeoTransform()
    projection = first_ds.GetProjection()
    
    # Determine the combined extent of all images
    # For simplicity, we use the union of extents
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
    
    # Calculate output image size
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])
    out_width = int((x_max - x_min) / pixel_width)
    out_height = int((y_max - y_min) / pixel_height)
    
    # Get number of bands
    n_bands = first_ds.RasterCount
    
    # Create output arrays
    merged_data = np.zeros((out_height, out_width, n_bands), dtype=np.float32)
    count_data = np.zeros((out_height, out_width, n_bands), dtype=np.float32)
    
    # For each source image
    for ds in datasets:
        gt = ds.GetGeoTransform()
        
        # Calculate the position of this image in the output image
        # To handle the case where pixel_height is negative
        x_offset = int(round((gt[0] - x_min) / pixel_width))
        y_offset = int(round((y_max - gt[3]) / pixel_height))
        
        # Read each band
        for band_idx in range(n_bands):
            band = ds.GetRasterBand(band_idx + 1)
            data = band.ReadAsArray()
            
            if data is None:
                continue
            
            rows, cols = data.shape
            
            # Mask of non-zero pixels (excluding 0 and nodata)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                valid_mask = (data != 0) & (data != nodata) & ~np.isnan(data)
            else:
                valid_mask = (data != 0) & ~np.isnan(data)
            
            # Calculate destination indices
            dst_y_start = max(0, y_offset)
            dst_y_end = min(out_height, y_offset + rows)
            dst_x_start = max(0, x_offset)
            dst_x_end = min(out_width, x_offset + cols)
            
            # Calculate corresponding source indices
            src_y_start = dst_y_start - y_offset
            src_y_end = src_y_start + (dst_y_end - dst_y_start)
            src_x_start = dst_x_start - x_offset
            src_x_end = src_x_start + (dst_x_end - dst_x_start)
            
            # Check dimension validity
            if dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
                continue
            if src_y_end <= src_y_start or src_x_end <= src_x_start:
                continue
            if src_y_end > rows or src_x_end > cols:
                continue
            
            # Accumulate valid values
            src_valid = valid_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            src_data = data[src_y_start:src_y_end, src_x_start:src_x_end]
            
            merged_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end, band_idx] += src_data * src_valid
            count_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end, band_idx] += src_valid.astype(np.float32)
    
    # Calculate average where we have data
    final_data = np.zeros_like(merged_data)
    for band_idx in range(n_bands):
        mask = count_data[:, :, band_idx] > 0
        final_data[:, :, band_idx][mask] = merged_data[:, :, band_idx][mask] / count_data[:, :, band_idx][mask]
    
    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, out_width, out_height, n_bands, gdal.GDT_Float32, 
                           options=['COMPRESS=LZW'])
    
    # Set output geotransform
    out_geotransform = list(geotransform)
    out_geotransform[0] = x_min
    out_geotransform[3] = y_max
    out_ds.SetGeoTransform(out_geotransform)
    out_ds.SetProjection(projection)
    
    # Write data
    for band_idx in range(n_bands):
        out_band = out_ds.GetRasterBand(band_idx + 1)
        out_band.WriteArray(final_data[:, :, band_idx])
        out_band.SetNoDataValue(-999)
        out_band.FlushCache()
    
    # Close datasets
    out_ds = None
    for ds in datasets:
        ds = None


def merge_similar_acquisitions(files, temp_dir):
    """
    Merge images that correspond to the same acquisition (same satellite, date, hour).
    Returns the list of merged files (or original if no merge needed).
    """
    # Filter only .tif files (not .ovr or others)
    files = [f for f in files if f.lower().endswith('.tif') and not f.lower().endswith('.ovr')]
    
    # Group files by acquisition key
    groups = defaultdict(list)
    files_without_key = []
    
    for f in files:
        key = extract_acquisition_key(f)
        if key:
            groups[key].append(f)
        else:
            # File that doesn't match the pattern, keep as is
            files_without_key.append(f)
    
    merged_files = []
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process identified groups
    for key, group_files in groups.items():
        if len(group_files) > 1:
            # Multiple files for the same acquisition -> smart merge
            print(f"Smart merge of {len(group_files)} images for {key}")
            # Use the first file name as base
            first_file = sorted(group_files)[0]
            output_name = os.path.basename(first_file)
            output_path = os.path.join(temp_dir, output_name)
            
            try:
                smart_merge_images(sorted(group_files), output_path)
                merged_files.append(output_path)
                print(f"  -> Merge successful: {output_name}")
            except Exception as e:
                print(f"Error during merge: {e}")
                # In case of error, keep the first file
                merged_files.append(first_file)
        else:
            # Single file for this acquisition
            merged_files.append(group_files[0])
    
    # Add files without key
    merged_files.extend(files_without_key)
    
    if files_without_key:
        print(f"Note: {len(files_without_key)} files could not be analyzed (non-standard format), they will be processed as is")
    
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
        # Image with 2 bands: HH and mask
        name_comp = os.path.basename(temp).split("_")
        name = name_comp[0] + "_" + name_comp[1] + f"_HH_{name_comp[-1]}"
        os.rename(temp, os.path.join(os.path.dirname(temp), name))

    if img.shape[-1] >= 3:
        # Image with 3+ bands: HH, HV, mask
        name_comp = os.path.basename(temp).split("_")
        name_0 = name_comp[0] + "_" + name_comp[1] + f"_HH_{name_comp[-1]}"
        name_1 = name_comp[0] + "_" + name_comp[1] + f"_HV_{name_comp[-1]}"
        
        # Extract channels correctly (channel 0 = HH, channel 1 = HV, last channel = mask)
        # Do not include mask in new images, or include it as 2nd band
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

    shp_path = "DATASET/SHP/"
    output = "DATA/"
    temp_merge_dir = "DATA/TEMP_MERGED/"

    for acq in ["asc", "dsc"]:
        acq_upper = acq.upper()  # ASC or DSC
        os.makedirs(os.path.join(output, acq_upper), exist_ok=True)
        path = f"/media/mgallet/BACK UP/DATA/X_SAR/{acq}/*.tif"
        all_files = sorted(glob.glob(path))
        
        # Step 1: Merge similar acquisitions
        print(f"\n=== Merging similar acquisitions for {acq_upper} ===")
        merged_files = merge_similar_acquisitions(all_files, os.path.join(temp_merge_dir, acq))
        print(f"Files after merge: {len(merged_files)} (original: {len(all_files)})")
        
        acq_initial = acq_upper[0]  # A or D (for filename)
        classe = [os.path.basename(x) for x in glob.glob(os.path.join(shp_path, "*"))]

        # Step 2: Process merged files
        print(f"\n=== Cropping images for {acq_upper} ===")
        for img in tqdm.tqdm(merged_files):
            for c in classe:
                dirs = os.path.join(output, acq_upper, c)  # Use acq_upper (ASC/DSC)
                os.makedirs(dirs, exist_ok=True)
                shp = os.path.join(shp_path, f"{c}/*.shp")
                shp_files = glob.glob(shp)

                Parallel(n_jobs=-1)(
                    delayed(clip_shp_img)(img, acq_initial, dirs, s) for s in shp_files  # Use acq_initial (A/D)
                )
    files_proc = glob.glob(f"{output}**/*.tif", recursive=True)
    for temp in tqdm.tqdm(files_proc):
        split_channel(temp)
    os.rmdir(temp_merge_dir)
