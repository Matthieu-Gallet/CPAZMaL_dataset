#!/usr/bin/env python3
"""
Script pour créer une figure d'acquisition raffinée avec subplots verticaux
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import os

mpl.use("pgf")

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)

def create_enhanced_acquisition_figure():
    """Crée une figure d'acquisition améliorée avec deux subplots verticaux"""
    
    # Lire et préparer les données
    df = pd.read_csv("../../ORIGINAL_DATA/metadata/data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Incidence_Angle'] = df['Angle Incidence'].str.replace('°', '').astype(float)
    
    # Classifier la polarisation
    def classify_polarization(pol_str):
        if '/' in pol_str:
            pol_count = len(pol_str.split('/'))
            return 'Dual/Quad-pol' if pol_count >= 2 else 'Single-pol'
        return 'Single-pol'
    
    df['Pol_Type'] = df['Polarisations'].apply(classify_polarization)
    
    # Filtrer les données par période
    # Période 2: PAZ (novembre 2019 - février 2023)
    period2_start = datetime(2019, 11, 1)
    period2_end = datetime(2023, 2, 28)
    df_period2 = df[(df['Date'] >= period2_start) & (df['Date'] <= period2_end)]
    
    # Créer la figure avec un subplot
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    # Définir les couleurs et symboles
    colors = {'Single-pol': 'tab:blue', 'Dual/Quad-pol': 'tab:orange'}
    markers = {'Ascending': '^', 'Descending': 'v'}
    marker_size = 180  # Taille uniforme encore plus grande pour les triangles
    
    # Fonction pour ploter les données sur un axe
    def plot_period_data(ax, data, title, ylim, xlim):
        for satellite in data['Satellite'].unique():
            sat_data = data[data['Satellite'] == satellite]
            
            for pol_type in sat_data['Pol_Type'].unique():
                for orbit in sat_data['Pass'].unique():
                    subset = sat_data[(sat_data['Pol_Type'] == pol_type) & (sat_data['Pass'] == orbit)]
                    
                    if len(subset) > 0:
                        ax.scatter(subset['Date'], subset['Incidence_Angle'], 
                                  c=colors[pol_type], marker=markers[orbit], 
                                  s=marker_size, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Configuration de l'axe
        ax.set_ylabel('Incidence Angle (°)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        # Format des dates
        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
    
    # Ploter période 2: PAZ (nov 2019 - fév 2023) 
    plot_period_data(ax, df_period2, 
                    'PAZ (Nov 2019 - Feb 2023)', 
                    (35, 55.5), 
                    (period2_start, period2_end))
    
    # Légende personnalisée - horizontale sous les deux plots
    from matplotlib.lines import Line2D
    
    # Créer les éléments de légende
    legend_elements = []
    
    # Orbites (triangles plus gros)
    legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                                 markerfacecolor='gray', markersize=16,
                                 label='Ascending', markeredgecolor='black'))
    legend_elements.append(Line2D([0], [0], marker='v', color='w', 
                                 markerfacecolor='gray', markersize=16,
                                 label='Descending', markeredgecolor='black'))
    
    # Polarisations
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='tab:blue', markersize=14,
                                 label='Single-pol', markeredgecolor='black'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='tab:orange', markersize=14,
                                 label='Dual-pol', markeredgecolor='black'))
    
    # Ajouter la légende horizontale sous le subplot
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, -0.075), ncol=4, frameon=False, 
               fancybox=False, shadow=False)
    
    # Ajuster la mise en page
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Laisser de l'espace pour la légende
    
    # Sauvegarder
    os.makedirs("figure", exist_ok=True)
    plt.savefig("figure/enhanced_acquisition_timeline.pdf", dpi=300, bbox_inches="tight", backend="pgf")
    plt.savefig("figure/enhanced_acquisition_timeline.png", dpi=300, bbox_inches="tight")
    
    print("Figure améliorée sauvegardée: figure/enhanced_acquisition_timeline.pdf")
    
    # Statistiques détaillées
    print("\\n=== DETAILED ACQUISITION STATISTICS ===")
    print(f"Total acquisitions: {len(df)}")
    print(f"Time span: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Duration: {(df['Date'].max() - df['Date'].min()).days} days")
    
    print("\\n=== PERIOD 2 (PAZ Era): Nov 2019 - Feb 2023 ===")
    print(f"Period 2 acquisitions: {len(df_period2)}")
    for satellite in df_period2['Satellite'].unique():
        sat_data = df_period2[df_period2['Satellite'] == satellite]
        print(f"  {satellite}: {len(sat_data)} acquisitions ({len(sat_data)/len(df_period2)*100:.1f}%)")
        print(f"    Incidence range: {sat_data['Incidence_Angle'].min():.1f}° to {sat_data['Incidence_Angle'].max():.1f}°")
    
    print("\\nBy Orbit (Overall):")
    for orbit in df['Pass'].unique():
        orbit_data = df[df['Pass'] == orbit]
        mean_inc = orbit_data['Incidence_Angle'].mean()
        std_inc = orbit_data['Incidence_Angle'].std()
        print(f"  {orbit}: {len(orbit_data)} acquisitions, mean incidence: {mean_inc:.1f}° ± {std_inc:.1f}°")
    
    print("\\nBy Polarization (Overall):")
    for pol in df['Pol_Type'].unique():
        pol_data = df[df['Pol_Type'] == pol]
        print(f"  {pol}: {len(pol_data)} acquisitions ({len(pol_data)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    print("Génération de la figure d'acquisition améliorée...")
    create_enhanced_acquisition_figure()
    print("Terminé!")