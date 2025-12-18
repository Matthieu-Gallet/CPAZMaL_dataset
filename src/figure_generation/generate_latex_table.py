#!/usr/bin/env python3
"""
Script pour générer un tableau LaTeX à partir des données details_shp.csv et HDF5
"""

import pandas as pd
import numpy as np
import h5py
from collections import defaultdict

def create_latex_table_from_csv():
    """Crée un tableau LaTeX à partir du fichier details_shp.csv et HDF5"""
    
    # Lire le fichier CSV
    df = pd.read_csv("../../ORIGINAL_DATA/metadata/shapefile_statistics.csv")
    
    # Extraire les statistiques HDF5
    h5_file = "../../DATASET/dataset/PAZTSX_CRYO_ML.hdf5"
    group_stats = {}
    class_stats = {}
    
    with h5py.File(h5_file, 'r') as hf:
        if 'data' in hf:
            data_group = hf['data']
            
            for group_name in data_group.keys():
                group = data_group[group_name]
                
                if 'class' not in group.attrs:
                    continue
                
                classe = group.attrs['class']
                
                # Initialiser les stats pour ce groupe
                group_stats[group_name] = {
                    'classe': classe,
                    'size_h': [],
                    'size_w': [],
                    'valid_pct': [],
                    'n_acq_ASC_HH': 0,
                    'n_acq_ASC_HV': 0,
                    'n_acq_DSC_HH': 0,
                    'n_acq_DSC_HV': 0,
                    'satellites_ASC': defaultdict(int),
                    'satellites_DSC': defaultdict(int)
                }
                
                # Parcourir les orbites et polarisations
                for orbit in ['ASC', 'DSC']:
                    if orbit not in group:
                        continue
                    
                    for pol in ['HH', 'HV']:
                        path = f"{orbit}/{pol}"
                        if path not in group:
                            continue
                        
                        pol_group = group[f"{orbit}/{pol}"]
                        
                        # Taille en pixels (H, W)
                        images = pol_group['images']
                        h, w, t = images.shape
                        group_stats[group_name]['size_h'].append(h)
                        group_stats[group_name]['size_w'].append(w)
                        
                        # Pourcentage de pixels valides
                        sample_image = images[:, :, 0]
                        valid_mask = (sample_image != -999) & np.isfinite(sample_image)
                        valid_pct = (np.sum(valid_mask) / (h * w)) * 100
                        group_stats[group_name]['valid_pct'].append(valid_pct)
                        
                        # Nombre d'acquisitions par orbite et polarisation
                        if orbit == 'ASC':
                            if pol == 'HH':
                                group_stats[group_name]['n_acq_ASC_HH'] = t
                            else:
                                group_stats[group_name]['n_acq_ASC_HV'] = t
                        else:  # DSC
                            if pol == 'HH':
                                group_stats[group_name]['n_acq_DSC_HH'] = t
                            else:
                                group_stats[group_name]['n_acq_DSC_HV'] = t
                        
                        # Satellites par orbite
                        if 'satellites' in pol_group:
                            satellites = pol_group['satellites'][:]
                            for sat in satellites:
                                sat_name = sat.decode('utf-8') if isinstance(sat, bytes) else sat
                                # Mapper les noms de satellites
                                if 'TerraSAR' in sat_name:
                                    sat_key = 'TSX'
                                elif 'TanDEM' in sat_name:
                                    sat_key = 'TDX'
                                elif 'PAZ' in sat_name:
                                    sat_key = 'PAZ'
                                else:
                                    sat_key = sat_name
                                
                                if orbit == 'ASC':
                                    group_stats[group_name]['satellites_ASC'][sat_key] += 1
                                else:
                                    group_stats[group_name]['satellites_DSC'][sat_key] += 1
                
                # Moyennes pour ce groupe
                if group_stats[group_name]['size_h']:
                    group_stats[group_name]['avg_size_h'] = int(np.mean(group_stats[group_name]['size_h']))
                    group_stats[group_name]['avg_size_w'] = int(np.mean(group_stats[group_name]['size_w']))
                    group_stats[group_name]['avg_valid_pct'] = np.mean(group_stats[group_name]['valid_pct'])
    
    # Agréger par classe
    for group_name, gstats in group_stats.items():
        classe = gstats['classe']
        if classe not in class_stats:
            class_stats[classe] = {
                'size_h': [],
                'size_w': [],
                'valid_pct': [],
                'n_acq_ASC_HH': [],
                'n_acq_ASC_HV': [],
                'n_acq_DSC_HH': [],
                'n_acq_DSC_HV': [],
                'satellites_ASC': defaultdict(int),
                'satellites_DSC': defaultdict(int)
            }
        
        class_stats[classe]['size_h'].append(gstats['avg_size_h'])
        class_stats[classe]['size_w'].append(gstats['avg_size_w'])
        class_stats[classe]['valid_pct'].append(gstats['avg_valid_pct'])
        class_stats[classe]['n_acq_ASC_HH'].append(gstats['n_acq_ASC_HH'])
        class_stats[classe]['n_acq_ASC_HV'].append(gstats['n_acq_ASC_HV'])
        class_stats[classe]['n_acq_DSC_HH'].append(gstats['n_acq_DSC_HH'])
        class_stats[classe]['n_acq_DSC_HV'].append(gstats['n_acq_DSC_HV'])
        
        for sat, count in gstats['satellites_ASC'].items():
            class_stats[classe]['satellites_ASC'][sat] += count
        for sat, count in gstats['satellites_DSC'].items():
            class_stats[classe]['satellites_DSC'][sat] += count
    
    # Grouper par classe pour avoir des statistiques par classe
    grouped = df.groupby('classe').agg({
        'latitude': ['mean', 'std', 'count'],
        'longitude': ['mean', 'std'],
        'altitude': ['mean', 'std', 'min', 'max'],
        'exposition': ['mean', 'std'],
        'pente': ['mean', 'std'],
        'group': 'nunique'
    }).round(2)
    
    # Créer un tableau plus simple avec les données principales
    latex_table = r"""
\begin{table*}[htbp]
\centering
\caption{Detail of the mean value over each classes of spatial, temporal and topographic parameters}
\label{tab:study_areas}
\resizebox{\textwidth}{!}{%
\begin{NiceTabular}{lccccccccccccccccc}
\toprule
\textbf{Class} & \textbf{Number} & \textbf{Size} & \textbf{Valid} & \Block{1-5}{\textbf{Ascending (n. acq.)}} & & & & & \Block{1-5}{\textbf{Descending (n. acq.)}} & & & & & \textbf{Alt} & \textbf{Slope} & \textbf{Asp} & \textbf{Lat/Lon} \\
& \textbf{of groups} & \textbf{(H$\times$W)} & \textbf{(\%)} & \Block{1-2}{\textit{Polarization}} & & \Block{1-3}{\textit{Satellites}} & & & \Block{1-2}{\textit{Polarization}} & & \Block{1-3}{\textit{Satellites}} & & & \textbf{(m)} & \textbf{(°)} & \textbf{(°)} & \\
& & & & \text{HH} & \text{HV} & \text{PAZ} & \text{TSX} & \text{TDX} & \text{HH} & \text{HV} & \text{PAZ} & \text{TSX} & \text{TDX} & & & & \\
\midrule
"""
    
    # Ajouter les données pour chaque classe
    for classe in sorted(df['classe'].unique()):
        classe_data = df[df['classe'] == classe]
        
        n_sites = len(classe_data)
        alt_mean = classe_data['altitude'].mean()
        alt_std = classe_data['altitude'].std()
        pente_mean = classe_data['pente'].mean()
        pente_std = classe_data['pente'].std()
        exp_mean = classe_data['exposition'].mean()
        exp_std = classe_data['exposition'].std()
        lat_mean = classe_data['latitude'].mean()
        lon_mean = classe_data['longitude'].mean()
        
        # Stats HDF5
        if classe in class_stats:
            cstats = class_stats[classe]
            avg_size_h = int(np.mean(cstats['size_h'])) if cstats['size_h'] else 0
            avg_size_w = int(np.mean(cstats['size_w'])) if cstats['size_w'] else 0
            avg_valid = np.mean(cstats['valid_pct']) if cstats['valid_pct'] else 0
            
            # Moyennes des acquisitions
            avg_asc_hh = int(np.mean(cstats['n_acq_ASC_HH'])) if cstats['n_acq_ASC_HH'] else 0
            avg_asc_hv = int(np.mean(cstats['n_acq_ASC_HV'])) if cstats['n_acq_ASC_HV'] else 0
            avg_dsc_hh = int(np.mean(cstats['n_acq_DSC_HH'])) if cstats['n_acq_DSC_HH'] else 0
            avg_dsc_hv = int(np.mean(cstats['n_acq_DSC_HV'])) if cstats['n_acq_DSC_HV'] else 0
            
            # Moyennes des satellites (diviser par le nombre de groupes)
            n_groups = len(cstats['size_h'])
            paz_asc = int(cstats['satellites_ASC']['PAZ'] / n_groups) if n_groups > 0 else 0
            tsx_asc = int(cstats['satellites_ASC']['TSX'] / n_groups) if n_groups > 0 else 0
            tdx_asc = int(cstats['satellites_ASC']['TDX'] / n_groups) if n_groups > 0 else 0
            paz_dsc = int(cstats['satellites_DSC']['PAZ'] / n_groups) if n_groups > 0 else 0
            tsx_dsc = int(cstats['satellites_DSC']['TSX'] / n_groups) if n_groups > 0 else 0
            tdx_dsc = int(cstats['satellites_DSC']['TDX'] / n_groups) if n_groups > 0 else 0
        else:
            avg_size_h = 0
            avg_size_w = 0
            avg_valid = 0
            avg_asc_hh = 0
            avg_asc_hv = 0
            avg_dsc_hh = 0
            avg_dsc_hv = 0
            paz_asc = 0
            tsx_asc = 0
            tdx_asc = 0
            paz_dsc = 0
            tsx_dsc = 0
            tdx_dsc = 0
        
        latex_table += f"{classe} & {n_sites} & "
        latex_table += f"${avg_size_h} \\times {avg_size_w}$ & {avg_valid:.1f} & "
        latex_table += f"{avg_asc_hh} & {avg_asc_hv} & "
        latex_table += f"{paz_asc} & {tsx_asc} & {tdx_asc} & "
        latex_table += f"{avg_dsc_hh} & {avg_dsc_hv} & "
        latex_table += f"{paz_dsc} & {tsx_dsc} & {tdx_dsc} & "
        latex_table += f"{alt_mean:.0f}$\\pm${alt_std:.0f} & "
        latex_table += f"{pente_mean:.1f}$\\pm${pente_std:.1f} & "
        latex_table += f"{exp_mean:.0f}$\\pm${exp_std:.0f} & "
        latex_table += f"{lat_mean:.2f}/{lon_mean:.2f} \\\\\n"
    
    latex_table += r"""
\bottomrule
\end{NiceTabular}%
}
\end{table*}
"""
    
    # Sauvegarder le tableau
    with open("../../figure/study_areas_table.tex", "w") as f:
        f.write(latex_table)
    
    print("Tableau LaTeX sauvegardé: figure/study_areas_table.tex")
    
    # Créer aussi un tableau détaillé avec toutes les zones
    detailed_table = r"""
\begin{table*}[htbp]
\centering
\caption{Detail of the individual study areas with spatial, temporal and topographic parameters}
\label{tab:detailed_study_areas}
\resizebox{\textwidth}{!}{%
\begin{NiceTabular}{llccccccccccccccccc}
\toprule
\textbf{ID} & \textbf{Class} & \textbf{Group} & \textbf{Size} & \textbf{Valid} & \Block{1-5}{\textbf{Ascending (n. acq.)}} & & & & & \Block{1-5}{\textbf{Descending (n. acq.)}} & & & & & \textbf{Alt} & \textbf{Slope} & \textbf{Asp} & \textbf{Lat/Lon} \\
& & & \textbf{(H$\times$W)} & \textbf{(\%)} & \Block{1-2}{\textit{Polarization}} & & \Block{1-3}{\textit{Satellites}} & & & \Block{1-2}{\textit{Polarization}} & & \Block{1-3}{\textit{Satellites}} & & & \textbf{(m)} & \textbf{(°)} & \textbf{(°)} & \\
& & & & & \text{HH} & \text{HV} & \text{PAZ} & \text{TSX} & \text{TDX} & \text{HH} & \text{HV} & \text{PAZ} & \text{TSX} & \text{TDX} & & & & \\
\midrule
"""
    
    # Mapping pour les noms de groupes STUDY (CSV -> HDF5)
    study_mapping = {
        'ALL_GLACIERS': 'ALLGLA',
        'ARGENTIER_TOP': 'ARGTOP'
    }
    
    # Trier par classe puis par ID
    df_sorted = df.sort_values(['classe', 'id'])
    
    for _, row in df_sorted.iterrows():
        # Mapper le nom du groupe pour STUDY
        if row['classe'] == 'STUDY' and row['group'] in study_mapping:
            group_name = study_mapping[row['group']]
        else:
            group_name = f"{row['classe']}{row['group']}"
        
        # Stats HDF5 pour ce groupe
        if group_name in group_stats:
            gstats = group_stats[group_name]
            size_h = gstats['avg_size_h']
            size_w = gstats['avg_size_w']
            valid = gstats['avg_valid_pct']
            n_asc_hh = gstats['n_acq_ASC_HH']
            n_asc_hv = gstats['n_acq_ASC_HV']
            n_dsc_hh = gstats['n_acq_DSC_HH']
            n_dsc_hv = gstats['n_acq_DSC_HV']
            paz_asc = gstats['satellites_ASC']['PAZ']
            tsx_asc = gstats['satellites_ASC']['TSX']
            tdx_asc = gstats['satellites_ASC']['TDX']
            paz_dsc = gstats['satellites_DSC']['PAZ']
            tsx_dsc = gstats['satellites_DSC']['TSX']
            tdx_dsc = gstats['satellites_DSC']['TDX']
        else:
            size_h = 0
            size_w = 0
            valid = 0
            n_asc_hh = 0
            n_asc_hv = 0
            n_dsc_hh = 0
            n_dsc_hv = 0
            paz_asc = 0
            tsx_asc = 0
            tdx_asc = 0
            paz_dsc = 0
            tsx_dsc = 0
            tdx_dsc = 0
        
        detailed_table += f"{row['id']} & {row['classe']} & {row['group']} & "
        detailed_table += f"${size_h} \\times {size_w}$ & {valid:.1f} & "
        detailed_table += f"{n_asc_hh} & {n_asc_hv} & "
        detailed_table += f"{paz_asc} & {tsx_asc} & {tdx_asc} & "
        detailed_table += f"{n_dsc_hh} & {n_dsc_hv} & "
        detailed_table += f"{paz_dsc} & {tsx_dsc} & {tdx_dsc} & "
        detailed_table += f"{row['altitude']:.0f} & {row['pente']:.1f} & {row['exposition']:.0f} & "
        detailed_table += f"{row['latitude']:.2f}/{row['longitude']:.2f} \\\\\n"
    
    detailed_table += r"""
\bottomrule
\end{NiceTabular}%
}
\end{table*}
"""
    
    # Sauvegarder le tableau détaillé
    with open("../../figure/detailed_study_areas_table.tex", "w") as f:
        f.write(detailed_table)
    
    print("Tableau détaillé sauvegardé: figure/detailed_study_areas_table.tex")
    
    # Créer un tableau de statistiques descriptives
    stats_table = r"""
\begin{table}[htbp]
\centering
\caption{Statistiques descriptives des zones d'étude}
\label{tab:study_areas_stats}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Variable} & \textbf{Min} & \textbf{Max} & \textbf{Moyenne} & \textbf{Médiane} & \textbf{Écart-type} \\
\hline
"""
    
    variables = [
        ('Altitude (m)', 'altitude'),
        ('Pente (°)', 'pente'),
        ('Exposition (°)', 'exposition'),
        ('Latitude', 'latitude'),
        ('Longitude', 'longitude')
    ]
    
    for var_name, col_name in variables:
        data = df[col_name]
        stats_table += f"{var_name} & {data.min():.1f} & {data.max():.1f} & "
        stats_table += f"{data.mean():.1f} & {data.median():.1f} & {data.std():.1f} \\\\\n"
    
    stats_table += r"""
\hline
\end{tabular}
\end{table}
"""
    
if __name__ == "__main__":
    create_latex_table_from_csv()