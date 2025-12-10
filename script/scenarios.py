#!/usr/bin/env python3
"""
Learning scenarios for PAZ/TSX HDF5 dataset.

This module contains 4 machine learning scenarios that use
an MLDatasetLoader to extract and prepare data.

Scenarios:
1. Temporal stacking classification (group k-fold)
2. Temporal prediction LSTM (time series)
3. Domain adaptation HH vs HV
4. Domain adaptation PAZ vs TerraSAR-X
"""

import numpy as np
from typing import Dict, Optional
from tqdm import tqdm


def scenario_1_temporal_stacking_classification(
    loader,
    window_size: int = 32,
    max_mask_value: int = 1,
    min_valid_percentage: float = 50.0,
    max_mask_percentage: float = 10.0,
    orbit: str = 'ASC',
    start_date: str = '20200101',
    end_date: str = '20201231',
    scale_type: str = 'intensity',
    skip_optim_offset: bool = True,
    verbose: bool = True
) -> Dict:
    """
    SCENARIO 1: Temporal stacking classification.
    
    Objective: Classify groups on 32x32 dual-pol (HH+HV) windows
    in ascending orbit over year 2020.
    
    Args:
        loader: MLDatasetLoader instance
        window_size: Window size (default: 32)
        max_mask_value: Max accepted mask value (0-3)
        max_mask_percentage: Max % of pixels with mask > max_mask_value
        min_valid_percentage: Min % of valid pixels (non nodata)
        orbit: Orbit ('ASC' or 'DSC')
        start_date: Start date ('YYYYMMDD')
        end_date: End date ('YYYYMMDD')
        scale_type: 'intensity' (default), 'amplitude' (data**0.5), or 'log10'
        verbose: Display detailed information
    
    Returns:
        Dict with:
            - X: Array (N,) of arrays - dual-pol data (window_size, window_size, T, 2)
            - y: Array (N,) - labels (classes encoded as int)
            - groups: Array (N,) - group identifiers encoded as int
            - masks: Array (N,) of arrays - masks (window_size, window_size, T)
            - satellites: Array (N,) of arrays - satellites per timestamp
            - class_names: Dict - mapping int -> class name
            - group_names: Dict - mapping int -> group name
            - positions: Array (N,) of tuples - window positions (y, x)
    """
    print(f"\n{'='*70}")
    print("SCENARIO 1: Temporal Stacking Classification (Dual-Pol)")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Windows: {window_size}x{window_size}")
    print(f"  - Polarization: Dual (HH + HV)")
    print(f"  - Period: {start_date} - {end_date}")
    print(f"  - Scale type: {scale_type}")
    print(f"  - Scale type: {scale_type}")
    print(f"  - Mask max: {max_mask_value}, {max_mask_percentage}%\n")
    
    # Filter learning classes (exclude STUDY)
    learning_classes = [c for c in loader.classes if c != 'STUDY']
    
    # Create class encoding
    class_to_int = {c: i for i, c in enumerate(learning_classes)}
    
    X_all = []
    y_all = []
    groups_all = []
    masks_all = []
    satellites_all = []
    group_names_all = []
    positions_all = []
    
    # Create group -> int mapping AND get unique group list
    all_groups = []
    for class_name in learning_classes:
        all_groups.extend(loader.get_groups_by_class(class_name))
    unique_groups = sorted(list(set(all_groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    
    # Create group -> class mapping for display
    group_to_class = {}
    for class_name in learning_classes:
        for group in loader.get_groups_by_class(class_name):
            group_to_class[group] = class_name
    
    # Loop directly on unique groups
    pbar = tqdm(unique_groups, desc="Groups", unit="grp")
    for group_name in pbar:
        class_name = group_to_class[group_name]
        
        # Update bar description with current group (rewrites in place)
        if verbose:
            pbar.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            # Load in dual-pol
            data = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation=['HH', 'HV'],
                start_date=start_date,
                end_date=end_date,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            images = data['images']  # (H, W, T, 2)
            masks = data['masks']    # (H, W, T)
            
            if images.shape[2] == 0:
                continue
            
            # Extract windows
            windows, window_masks, positions = loader.extract_windows(
                image=images,
                mask=masks,
                window_size=window_size,
                stride=window_size,  # Non-overlapping
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows is None:
                continue
            
            n_windows = len(windows)
            
            # Store each window individually with its metadata
            for idx in range(n_windows):
                X_all.append(windows[idx])  # (window_size, window_size, T, 2)
                y_all.append(class_to_int[class_name])
                groups_all.append(group_to_int[group_name])
                masks_all.append(window_masks[idx])
                group_names_all.append(group_name)
                positions_all.append(positions[idx])
                
                # Satellites for this window
                sat_array = np.array(data['satellites'])
                satellites_all.append(sat_array)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Skipping {group_name}: {str(e)}")
            continue
    
    if len(X_all) == 0:
        raise ValueError("No windows extracted. Check parameters.")
    
    # Convert to arrays with dtype=float32 for X
    X = np.array([x.astype(np.float32) for x in X_all], dtype=object)
    y = np.array(y_all)
    groups = np.array(groups_all)
    masks = np.array(masks_all, dtype=object)
    satellites = np.array(satellites_all, dtype=object)
    positions = np.array(positions_all, dtype=object)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  - Total windows: {len(X)}")
        print(f"  - X.shape: {X.shape}")
        print(f"  - y.shape: {y.shape}")
        print(f"  - groups.shape: {groups.shape}")
        print(f"  - Unique classes: {np.unique(y, return_counts=True)}")
        print(f"  - Unique groups: {len(np.unique(groups))}")
    
    # Timestamp distribution
    t_sizes = [x.shape[2] for x in X]
    print(f"  - Timestamps per window: min={min(t_sizes)}, max={max(t_sizes)}, mean={np.mean(t_sizes):.1f}")
    
    return {
        'X': X,  # Array (N,) of arrays (window_size, window_size, T, 2)
        'y': y,
        'groups': groups,  # Array (N,) - encoded as int
        'masks': masks,  # Array (N,) of arrays
        'satellites': satellites,  # Array (N,) of arrays
        'class_names': {v: k for k, v in class_to_int.items()},
        'group_names': {v: k for k, v in group_to_int.items()},
        'positions': positions,
        'metadata': {
            'window_size': window_size,
            'orbit': orbit,
            'period': (start_date, end_date),
            'polarization': 'Dual (HH+HV)',
            'scale_type': scale_type,
            'note': 'X and masks are object arrays due to variable T between windows'
        }
    }


def scenario_2_temporal_prediction_lstm(
    loader,
    window_size: int = 32,
    max_mask_value: int = 1,
    max_mask_percentage: float = 10.0,
    min_valid_percentage: float = 50.0, 
    orbit: str = 'DSC',
    polarization: str = 'HH',
    train_start: str = '20200101',
    train_end: str = '20201031',
    predict_start: str = '20201101',
    predict_end: str = '20201231',
    scale_type: str = 'intensity',
    skip_optim_offset: bool = True,
    verbose: bool = True
) -> Dict:
    """
    SCENARIO 2: Temporal prediction with LSTM.
    
    Objective: Learn on jan-oct 2020 and predict nov-dec 2020.
    Mean HH in descending orbit for time series.
    
    Args:
        loader: MLDatasetLoader instance
        window_size: Window size
        max_mask_value: Max accepted mask value
        max_mask_percentage: Max % of pixels with mask > max_mask_value
        min_valid_percentage: Min % of valid pixels
        orbit: Orbit ('DSC')
        polarization: 'HH' or 'HV'
        train_start: Training period start
        train_end: Training period end
        predict_start: Prediction period start
        predict_end: Prediction period end
        scale_type: 'intensity' (default), 'amplitude' (data**0.5), or 'log10'
        verbose: Display detailed information
    
    Returns:
        Dict with:
            - X_train: Array (N,) of arrays - train sequences (window_size, window_size, T_train)
            - X_predict: Array (N,) of arrays - sequences to predict (window_size, window_size, T_predict)
            - groups: Array (N,) - group identifiers encoded as int
            - masks_train: Array (N,) of arrays (window_size, window_size, T_train)
            - masks_predict: Array (N,) of arrays (window_size, window_size, T_predict)
            - timestamps_train: Array (N,) of arrays - dates
            - timestamps_predict: Array (N,) of arrays - dates
            - class_labels: Array (N,) - classes for analysis
            - group_names: Dict - mapping int -> group name
    """
    print(f"\n{'='*70}")
    print("SCENARIO 2: Temporal Prediction LSTM")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Windows: {window_size}x{window_size}")
    print(f"  - Polarization: {polarization}")
    print(f"  - Orbit: {orbit}")
    print(f"  - Train: {train_start} - {train_end}")
    print(f"  - Predict: {predict_start} - {predict_end}")
    print(f"  - Scale type: {scale_type}")
    print(f"  - Mask max: {max_mask_value}, {max_mask_percentage}%\n")
    
    learning_classes = [c for c in loader.classes if c != 'STUDY']
    class_to_int = {c: i for i, c in enumerate(learning_classes)}
    
    X_train_all = []
    X_predict_all = []
    groups_all = []
    masks_train_all = []
    masks_predict_all = []
    timestamps_train_all = []
    timestamps_predict_all = []
    class_labels_all = []
    group_names_all = []
    
    # Create mapping group -> int and get the unique group list
    all_groups = []
    for class_name in learning_classes:
        all_groups.extend(loader.get_groups_by_class(class_name))
    unique_groups = sorted(list(set(all_groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    
    # Create group -> class mapping for display
    group_to_class = {}
    for class_name in learning_classes:
        for group in loader.get_groups_by_class(class_name):
            group_to_class[group] = class_name
    
    # Loop directly on the unique groups
    pbar = tqdm(unique_groups, desc="Groups", unit="grp")
    for group_name in pbar:
        class_name = group_to_class[group_name]
        
        # Update progress bar description with current group
        if verbose:
            pbar.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            # Load training period
            data_train = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation=polarization,
                start_date=train_start,
                end_date=train_end,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            # Load prediction period
            data_predict = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation=polarization,
                start_date=predict_start,
                end_date=predict_end,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            if data_train['images'].shape[2] == 0 or data_predict['images'].shape[2] == 0:
                continue
            
            # Calculate the spatial mean over the whole image (no windowing)
            # Option: use windows
            img_train = data_train['images']  # (H, W, T_train)
            img_predict = data_predict['images']  # (H, W, T_predict)
            mask_train = data_train['masks']
            mask_predict = data_predict['masks']
            
            # Extract windows for training
            windows_train, window_masks_train, positions = loader.extract_windows(
                image=img_train,
                mask=mask_train,
                window_size=window_size,
                stride=window_size,
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows_train is None:
                continue
            
            # For each training window, extract the same position for prediction
            n_windows = len(windows_train)
            windows_predict_list = []
            window_masks_predict_list = []
            valid_indices = []
            
            for idx, (y, x) in enumerate(positions):
                # Check that the position exists in the prediction set
                if y + window_size <= img_predict.shape[0] and x + window_size <= img_predict.shape[1]:
                    win_pred = img_predict[y:y+window_size, x:x+window_size, :]
                    mask_pred = mask_predict[y:y+window_size, x:x+window_size, :]
                    
                    # Check mask criterion
                    bad_pixels = np.any(mask_pred > max_mask_value, axis=-1)
                    bad_pct = (np.sum(bad_pixels) / (window_size * window_size)) * 100
                    
                    if bad_pct <= max_mask_percentage:
                        windows_predict_list.append(win_pred)
                        window_masks_predict_list.append(mask_pred)
                        valid_indices.append(idx)
            
            if len(windows_predict_list) == 0:
                continue
            
            # Filter windows_train to keep only valid positions
            windows_train = windows_train[valid_indices]
            window_masks_train = window_masks_train[valid_indices]
            windows_predict = np.array(windows_predict_list)
            window_masks_predict = np.array(window_masks_predict_list)
            
            n_valid = len(windows_train)
            
            # Store each window individually
            ts_train = np.array(data_train['timestamps'])
            ts_predict = np.array(data_predict['timestamps'])
            
            for idx in range(n_valid):
                X_train_all.append(windows_train[idx])
                X_predict_all.append(windows_predict[idx])
                groups_all.append(group_to_int[group_name])
                masks_train_all.append(window_masks_train[idx])
                masks_predict_all.append(window_masks_predict[idx])
                class_labels_all.append(class_to_int[class_name])
                group_names_all.append(group_name)
                timestamps_train_all.append(ts_train)
                timestamps_predict_all.append(ts_predict)
            
        except Exception as e:
            continue
    
    if len(X_train_all) == 0:
        raise ValueError("No windows extracted. Check parameters.")
    
    X_train = np.array([x.astype(np.float32) for x in X_train_all], dtype=object)
    X_predict = np.array([x.astype(np.float32) for x in X_predict_all], dtype=object)
    groups = np.array(groups_all)
    class_labels = np.array(class_labels_all)
    masks_train = np.array(masks_train_all, dtype=object)
    masks_predict = np.array(masks_predict_all, dtype=object)
    timestamps_train = np.array(timestamps_train_all, dtype=object)
    timestamps_predict = np.array(timestamps_predict_all, dtype=object)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  - Total windows: {len(X_train)}")
        print(f"  - X_train.shape: {X_train.shape}")
        print(f"  - X_predict.shape: {X_predict.shape}")
        print(f"  - groups.shape: {groups.shape}")
        print(f"  - class_labels uniques: {np.unique(class_labels, return_counts=True)}")
        print(f"  - Unique groups: {len(np.unique(groups))}")
    
    return {
    'X_train': X_train,  # Array (N,) of arrays (window_size, window_size, T_train)
    'X_predict': X_predict,  # Array (N,) of arrays (window_size, window_size, T_predict)
    'groups': groups,  # Array (N,) - encoded as int
    'masks_train': masks_train,  # Array (N,) of arrays
    'masks_predict': masks_predict,  # Array (N,) of arrays
        'timestamps_train': timestamps_train,
        'timestamps_predict': timestamps_predict,
        'class_labels': class_labels,
        'class_names': {v: k for k, v in class_to_int.items()},
        'group_names': {v: k for k, v in group_to_int.items()},
        'metadata': {
            'window_size': window_size,
            'orbit': orbit,
            'polarization': polarization,
            'train_period': (train_start, train_end),
            'predict_period': (predict_start, predict_end),
            'note': 'X_train and X_predict are object arrays because T is variable'
        }
    }


def scenario_3_domain_adaptation_pol(
    loader,
    window_size: int = 32,
    max_mask_value: int = 1,
    max_mask_percentage: float = 10.0,
    min_valid_percentage: float = 50.0,
    orbit: str = 'DSC',
    target_date: str = '20200804',
    scale_type: str = 'intensity',
    skip_optim_offset: bool = True,
    verbose: bool = True
) -> Dict:
    """
    SCENARIO 3: Domain Adaptation HH vs HV (same date).
    
    Objective: Train on HH and test on HV for the same acquisition.
    Date: 2020-08-04, all classes.
    
    Args:
        loader: MLDatasetLoader instance
        window_size: Window size
        max_mask_value: Max mask value
        max_mask_percentage: Max % of pixels with mask > max_mask_value
        min_valid_percentage: Min % of valid pixels
        orbit: Orbit
        target_date: Acquisition date ('YYYYMMDD')
        scale_type: 'intensity' (default), 'amplitude' (data**0.5), or 'log10'
        verbose: Display detailed information
    
    Returns:
        Dict with:
            - X_source (HH): Array (N, window_size, window_size)
            - X_target (HV): Array (N, window_size, window_size)
            - y: Array (N,) - labels
            - groups: Array (N,) - group identifiers encoded as int
            - masks_source: Array (N, window_size, window_size)
            - masks_target: Array (N, window_size, window_size)
            - satellites: Array (N,) - satellites
            - group_names: Dict - mapping int -> group name
    """
    print(f"\n{'='*70}")
    print("SCENARIO 3: Domain Adaptation HH -> HV")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Windows: {window_size}x{window_size}")
    print(f"  - Date: {target_date}")
    print(f"  - Orbit: {orbit}")
    print(f"  - Source: HH -> Target: HV")
    print(f"  - Mask max: {max_mask_value}, {max_mask_percentage}%\n")
    
    learning_classes = [c for c in loader.classes if c != 'STUDY']
    class_to_int = {c: i for i, c in enumerate(learning_classes)}
    
    X_source_all = []
    X_target_all = []
    y_all = []
    groups_all = []
    masks_source_all = []
    masks_target_all = []
    satellites_all = []
    group_names_all = []
    
    # Create mapping group -> int and get the unique group list
    all_groups = []
    for class_name in learning_classes:
        all_groups.extend(loader.get_groups_by_class(class_name))
    unique_groups = sorted(list(set(all_groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    
    # Create group -> class mapping for display
    group_to_class = {}
    for class_name in learning_classes:
        for group in loader.get_groups_by_class(class_name):
            group_to_class[group] = class_name
    
    # Loop directly on the unique groups
    pbar = tqdm(unique_groups, desc="Groups", unit="grp")
    for group_name in pbar:
        class_name = group_to_class[group_name]
        
        # Update progress bar description with current group
        if verbose:
            pbar.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            # Load HH (source)
            data_hh = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation='HH',
                start_date=target_date,
                end_date=target_date,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            # Load HV (target)
            data_hv = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation='HV',
                start_date=target_date,
                end_date=target_date,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            if data_hh['images'].shape[2] == 0 or data_hv['images'].shape[2] == 0:
                continue
            
            # Take the first (and only) acquisition
            img_hh = data_hh['images'][:, :, 0]
            img_hv = data_hv['images'][:, :, 0]
            mask_hh = data_hh['masks'][:, :, 0]
            mask_hv = data_hv['masks'][:, :, 0]
            
            # Extract HH windows
            windows_hh, window_masks_hh, positions = loader.extract_windows(
                image=img_hh,
                mask=mask_hh,
                window_size=window_size,
                stride=window_size,
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows_hh is None:
                continue
            
            # Extract the same positions for HV
            windows_hv_list = []
            window_masks_hv_list = []
            valid_indices = []
            
            for idx, (y, x) in enumerate(positions):
                if y + window_size <= img_hv.shape[0] and x + window_size <= img_hv.shape[1]:
                    win_hv = img_hv[y:y+window_size, x:x+window_size]
                    mask_hv_win = mask_hv[y:y+window_size, x:x+window_size]
                    
                    # Check criterion
                    bad_pixels = mask_hv_win > max_mask_value
                    bad_pct = (np.sum(bad_pixels) / (window_size * window_size)) * 100
                    
                    if bad_pct <= max_mask_percentage:
                        windows_hv_list.append(win_hv)
                        window_masks_hv_list.append(mask_hv_win)
                        valid_indices.append(idx)
            
            if len(windows_hv_list) == 0:
                continue
            
            windows_hh = windows_hh[valid_indices]
            window_masks_hh = window_masks_hh[valid_indices]
            windows_hv = np.array(windows_hv_list)
            window_masks_hv = np.array(window_masks_hv_list)
            
            n_valid = len(windows_hh)
            
            # Store individually
            sat = data_hh['satellites'][0]
            for idx in range(n_valid):
                X_source_all.append(windows_hh[idx])
                X_target_all.append(windows_hv[idx])
                y_all.append(class_to_int[class_name])
                groups_all.append(group_to_int[group_name])
                masks_source_all.append(window_masks_hh[idx])
                masks_target_all.append(window_masks_hv[idx])
                group_names_all.append(group_name)
                satellites_all.append(sat)
            
        except Exception as e:
            continue
    
    if len(X_source_all) == 0:
        raise ValueError("No windows extracted. Check parameters.")
    
    # Convert to arrays (same shape since T=1)
    X_source = np.array(X_source_all, dtype=np.float32)
    X_target = np.array(X_target_all, dtype=np.float32)
    y = np.array(y_all)
    groups = np.array(groups_all)
    masks_source = np.array(masks_source_all)
    masks_target = np.array(masks_target_all)
    satellites = np.array(satellites_all)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  - Total windows: {len(X_source)}")
        print(f"  - X_source.shape: {X_source.shape}")
        print(f"  - X_target.shape: {X_target.shape}")
        print(f"  - y.shape: {y.shape}")
        print(f"  - groups.shape: {groups.shape}")
        print(f"  - Unique classes: {np.unique(y, return_counts=True)}")
        print(f"  - Unique groups: {len(np.unique(groups))}")
        
    return {
        'X_source': X_source,
        'X_target': X_target,
        'y': y,
    'groups': groups,  # Array (N,) - encoded as int
        'masks_source': masks_source,
        'masks_target': masks_target,
        'satellites': satellites,
        'class_names': {v: k for k, v in class_to_int.items()},
        'group_names': {v: k for k, v in group_to_int.items()},
        'metadata': {
            'window_size': window_size,
            'date': target_date,
            'orbit': orbit,
            'source_pol': 'HH',
            'target_pol': 'HV'
        }
    }


def scenario_4_domain_adaptation_satellite(
    loader,
    window_size: int = 32,
    max_mask_value: int = 1,
    max_mask_percentage: float = 10.0,
    min_valid_percentage: float = 50.0,
    source_orbit: str = 'DSC',
    target_orbit: str = 'ASC',
    source_date: str = '20210127',
    target_date: str = '20210214',
    source_polarization: str = 'HH',  # Can be 'HH', 'HV', or ['HH', 'HV']
    target_polarization: str = 'HH',  # Can be 'HH', 'HV', or ['HH', 'HV']
    scale_type: str = 'intensity',
    skip_optim_offset: bool = True,
    verbose: bool = True
) -> Dict:
    """
    SCENARIO 4: Domain Adaptation between different acquisition geometries.
    
    Objective: Learn on source geometry (labeled) and adapt to target geometry (unlabeled).
    Default: Source: PAZ DSC 2021-01-27 HH -> Target: PAZ ASC 2021-02-14 HH
    
    Args:
        loader: Instance of MLDatasetLoader
        window_size: Window size
        max_mask_value: Max mask value
        max_mask_percentage: Max % of pixels with mask > max_mask_value
        min_valid_percentage: Min % of valid pixels
        source_orbit: Source orbit ('ASC' or 'DSC')
        target_orbit: Target orbit ('ASC' or 'DSC')
        source_date: Source acquisition date ('YYYYMMDD')
        target_date: Target acquisition date ('YYYYMMDD')
        source_polarization: Source polarization ('HH', 'HV', or ['HH', 'HV'])
        target_polarization: Target polarization ('HH', 'HV', or ['HH', 'HV'])
        scale_type: 'intensity' (default), 'amplitude', or 'log10'
        skip_optim_offset: Skip window offset optimization
        verbose: Display detailed information
    
    Returns:
        Dict with:
            - X_source: Array (N_source, window_size, window_size, n_pol_source)
            - X_target: Array (N_target, window_size, window_size, n_pol_target)
            - y_source: Array (N_source,) - labels source
            - y_target: Array (N_target,) - labels target (for analysis only)
            - groups_source: Array (N_source,) - group identifiers for source
            - groups_target: Array (N_target,) - group identifiers for target
            - masks_source: Array (N_source, window_size, window_size)
            - masks_target: Array (N_target, window_size, window_size)
            - class_names: Dict - mapping int -> class name
            - group_names: Dict - mapping int -> group name
    """
    print(f"\n{'='*70}")
    print("SCENARIO 4: Domain Adaptation - Different Acquisition Geometries")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Windows: {window_size}x{window_size}")
    print(f"  - Source: PAZ {source_orbit} {source_date} {source_polarization}")
    print(f"  - Target: PAZ {target_orbit} {target_date} {target_polarization}")
    print(f"  - Scale type: {scale_type}")
    print(f"  - Mask max: {max_mask_value}, {max_mask_percentage}%\n")
    
    learning_classes = [c for c in loader.classes if c != 'STUDY']
    class_to_int = {c: i for i, c in enumerate(learning_classes)}
    
    X_source_all = []
    X_target_all = []
    y_source_all = []
    y_target_all = []
    groups_source_all = []
    groups_target_all = []
    masks_source_all = []
    masks_target_all = []
    group_names_source_all = []
    group_names_target_all = []
    
    # Create mapping group -> int (common for source and target) and get the unique group list
    all_groups = []
    for class_name in learning_classes:
        all_groups.extend(loader.get_groups_by_class(class_name))
    unique_groups = sorted(list(set(all_groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    
    # Create group -> class mapping for display
    group_to_class = {}
    for class_name in learning_classes:
        for group in loader.get_groups_by_class(class_name):
            group_to_class[group] = class_name
    
    # SOURCE
    print("=" * 70)
    print(f"Loading SOURCE ({source_orbit} {source_date} {source_polarization})...")
    pbar_source = tqdm(unique_groups, desc="Source Groups", unit="grp")
    for group_name in pbar_source:
        class_name = group_to_class[group_name]
        
        # Update progress bar description with current group
        if verbose:
            pbar_source.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            # Load source data
            data = loader.load_data(
                group_name=group_name,
                orbit=source_orbit,
                polarisation=source_polarization,
                start_date=source_date,
                end_date=source_date,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            if data['images'].shape[2] == 0:
                continue
            
            # Filter by PAZ satellite
            satellites = np.array(data['satellites'])
            paz_indices = np.where(satellites == 'PAZ')[0]
            
            if len(paz_indices) == 0:
                continue
            
            # Take the first PAZ acquisition
            idx = paz_indices[0]
            
            # Handle both single-pol and dual-pol
            if len(data['images'].shape) == 4:  # Dual-pol: (H, W, T, 2)
                img = data['images'][:, :, idx, :]  # (H, W, 2)
            else:  # Single-pol: (H, W, T)
                img = data['images'][:, :, idx]  # (H, W)
            
            mask = data['masks'][:, :, idx]
            
            # Extract windows
            windows, window_masks, positions = loader.extract_windows(
                image=img,
                mask=mask,
                window_size=window_size,
                stride=window_size,
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows is None:
                continue
            
            n_windows = len(windows)
            
            # Store individually
            for idx_win in range(n_windows):
                X_source_all.append(windows[idx_win])
                y_source_all.append(class_to_int[class_name])
                groups_source_all.append(group_to_int[group_name])
                masks_source_all.append(window_masks[idx_win])
                group_names_source_all.append(group_name)
            
        except Exception as e:
            continue
    
    # TARGET
    print("\n" + "=" * 70)
    print(f"Loading TARGET ({target_orbit} {target_date} {target_polarization})...")
    pbar_target = tqdm(unique_groups, desc="Target Groups", unit="grp")
    for group_name in pbar_target:
        class_name = group_to_class[group_name]
        
        # Update progress bar description with current group
        if verbose:
            pbar_target.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            data = loader.load_data(
                group_name=group_name,
                orbit=target_orbit,
                polarisation=target_polarization,
                start_date=target_date,
                end_date=target_date,
                normalize=False,
                remove_nodata=False,
                scale_type=scale_type
            )
            
            if data['images'].shape[2] == 0:
                continue
            
            # Filter by PAZ satellite
            satellites = np.array(data['satellites'])
            paz_indices = np.where(satellites == 'PAZ')[0]
            
            if len(paz_indices) == 0:
                continue
            
            idx = paz_indices[0]
            
            # Handle both single-pol and dual-pol
            if len(data['images'].shape) == 4:  # Dual-pol: (H, W, T, 2)
                img = data['images'][:, :, idx, :]  # (H, W, 2)
            else:  # Single-pol: (H, W, T)
                img = data['images'][:, :, idx]  # (H, W)
            
            mask = data['masks'][:, :, idx]
            
            windows, window_masks, positions = loader.extract_windows(
                image=img,
                mask=mask,
                window_size=window_size,
                stride=window_size,
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows is None:
                continue
            
            n_windows = len(windows)
            
            # Store individually
            for idx_win in range(n_windows):
                X_target_all.append(windows[idx_win])
                y_target_all.append(class_to_int[class_name])  # For analysis only
                groups_target_all.append(group_to_int[group_name])
                masks_target_all.append(window_masks[idx_win])
                group_names_target_all.append(group_name)
            
        except Exception as e:
            continue
    
    if len(X_source_all) == 0 or len(X_target_all) == 0:
        raise ValueError("No source or target data found")
    
    X_source = np.array(X_source_all, dtype=np.float32)
    X_target = np.array(X_target_all, dtype=np.float32)
    y_source = np.array(y_source_all)
    y_target = np.array(y_target_all)
    groups_source = np.array(groups_source_all)
    groups_target = np.array(groups_target_all)
    masks_source = np.array(masks_source_all)
    masks_target = np.array(masks_target_all)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  - SOURCE ({source_orbit} {source_date} {source_polarization}):")
        print(f"    X_source.shape: {X_source.shape}")
        print(f"    y_source.shape: {y_source.shape}")
        print(f"    groups_source.shape: {groups_source.shape}")
        print(f"    Unique classes: {np.unique(y_source, return_counts=True)}")
        print(f"    Unique groups: {len(np.unique(groups_source))}")
        print(f"  - TARGET ({target_orbit} {target_date} {target_polarization}):")
        print(f"    X_target.shape: {X_target.shape}")
        print(f"    y_target.shape: {y_target.shape}")
        print(f"    groups_target.shape: {groups_target.shape}")
        print(f"    Unique groups: {len(np.unique(groups_target))}")
        
    return {
        'X_source': X_source,
        'X_target': X_target,
        'y_source': y_source,
        'y_target': y_target,
        'groups_source': groups_source,  # Array (N_source,) - encoded as int
        'groups_target': groups_target,  # Array (N_target,) - encoded as int
        'masks_source': masks_source,
        'masks_target': masks_target,
        'class_names': {v: k for k, v in class_to_int.items()},
        'group_names': {v: k for k, v in group_to_int.items()},
        'metadata': {
            'window_size': window_size,
            'source_date': source_date,
            'target_date': target_date,
            'source_orbit': source_orbit,
            'target_orbit': target_orbit,
            'source_polarization': source_polarization,
            'target_polarization': target_polarization,
            'satellite': 'PAZ'
        }
    }




def example_usage(selec: int = None):
    """Example usage of the 4 learning scenarios"""
    # Initialize the loader
    from load_dataset import MLDatasetLoader
    loader = MLDatasetLoader('../DATASET/PAZTSX_CRYO_ML.hdf5')
    
    # 1. Display dataset statistics
    print("\n Dataset statistics:")
    stats = loader.get_statistics_summary()
    print(f"  Classes: {loader.classes}")
    print(f"  Satellites: {loader.satellites}")
    print(f"  - Total groups: {loader.n_groups}")
    
    # ==========================================================================
    # SCENARIO 1: Temporal Stacking Classification
    # ==========================================================================
    if selec == 1:
        scenario1_data = scenario_1_temporal_stacking_classification(
            loader=loader,
            window_size=64,
            max_mask_value=1,
            max_mask_percentage=10.0,
            min_valid_percentage=100.0,  # Changed from 100.0 to 50.0
            orbit='ASC',
            start_date='20200101',
            end_date='20201231',
            scale_type='amplitude',
            skip_optim_offset=True,
        )
    
    # ==========================================================================
    # SCENARIO 2: Temporal Prediction LSTM
    # ==========================================================================
    if selec == 2:
        scenario2_data = scenario_2_temporal_prediction_lstm(
            loader=loader,
            window_size=32,
            max_mask_value=1,
            max_mask_percentage=10.0,
            min_valid_percentage=100.0,  # Changed from 100.0 to 50.0
            orbit='DSC',
            polarization='HH',
            train_start='20200101',
            train_end='20201031',
            predict_start='20201101',
            predict_end='20201231',
            scale_type='amplitude',
            skip_optim_offset=True,
        )
    
    # ==========================================================================
    # SCENARIO 3: Domain Adaptation HH vs HV
    # ==========================================================================    
    if selec == 3:
        scenario3_data = scenario_3_domain_adaptation_pol(
            loader=loader,
            window_size=32,
            max_mask_value=1,
            max_mask_percentage=10.0,
            min_valid_percentage=100.0,  # Changed from 100.0 to 50.0
            orbit='DSC',
            target_date='20200804',
            scale_type='intensity',
            skip_optim_offset=True
        )
        
    # ==========================================================================
    # SCENARIO 4: Domain Adaptation Different Geometries
    # ==========================================================================
    if selec == 4:
        scenario4_data = scenario_4_domain_adaptation_satellite(
            loader=loader,
            window_size=32,
            max_mask_value=1,
            max_mask_percentage=10.0,
            min_valid_percentage=100.0,
            source_orbit='DSC',
            target_orbit='ASC',
            source_date='20210127',
            target_date='20210214',
            source_polarization='HH',
            target_polarization='HH',
            scale_type='amplitude',
            skip_optim_offset=True
        )

if __name__ == "__main__":
    example_usage(selec=1)  # Choose the scenario to run (1 to 4)
    example_usage(selec=2)
    example_usage(selec=3)
    example_usage(selec=4)