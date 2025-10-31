#!/usr/bin/env python3
"""
Script pour générer un tableau LaTeX à partir des données details_shp.csv
"""

import pandas as pd
import numpy as np

def create_latex_table_from_csv():
    """Crée un tableau LaTeX à partir du fichier details_shp.csv"""
    
    # Lire le fichier CSV
    df = pd.read_csv("../../ORIGINAL_DATA/metadata/shapefile_statistics.csv")
    
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
\begin{table}[htbp]
\centering
\caption{Caractéristiques des zones d'étude par classe}
\label{tab:study_areas}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Classe} & \textbf{N} & \textbf{Altitude (m)} & \textbf{Pente (°)} & \textbf{Exposition (°)} & \textbf{Latitude} & \textbf{Longitude} \\
\hline
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
        
        latex_table += f"{classe} & {n_sites} & "
        latex_table += f"{alt_mean:.0f} $\\pm$ {alt_std:.0f} & "
        latex_table += f"{pente_mean:.1f} $\\pm$ {pente_std:.1f} & "
        latex_table += f"{exp_mean:.0f} $\\pm$ {exp_std:.0f} & "
        latex_table += f"{lat_mean:.3f} & {lon_mean:.3f} \\\\\n"
    
    latex_table += r"""
\hline
\end{tabular}
\end{table}
"""
    
    # Sauvegarder le tableau
    with open("../../figure/study_areas_table.tex", "w") as f:
        f.write(latex_table)
    
    print("Tableau LaTeX sauvegardé: figure/study_areas_table.tex")
    
    # Créer aussi un tableau détaillé avec toutes les zones
    detailed_table = r"""
\begin{table}[htbp]
\centering
\caption{Détail des zones d'étude individuelles}
\label{tab:detailed_study_areas}
\begin{tabular}{|l|l|c|c|c|c|c|c|}
\hline
\textbf{ID} & \textbf{Classe} & \textbf{Groupe} & \textbf{Latitude} & \textbf{Longitude} & \textbf{Altitude (m)} & \textbf{Pente (°)} & \textbf{Exposition (°)} \\
\hline
"""
    
    # Trier par classe puis par ID
    df_sorted = df.sort_values(['classe', 'id'])
    
    for _, row in df_sorted.iterrows():
        detailed_table += f"{row['id']} & {row['classe']} & {row['group']} & "
        detailed_table += f"{row['latitude']:.4f} & {row['longitude']:.4f} & "
        detailed_table += f"{row['altitude']:.0f} & {row['pente']:.1f} & {row['exposition']:.0f} \\\\\n"
    
    detailed_table += r"""
\hline
\end{tabular}
\end{table}
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