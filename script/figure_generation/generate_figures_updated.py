#!/usr/bin/env python3
"""
Script de génération des figures adapté pour le nouveau dataset HDF5 ML optimisé.

Structure du nouveau dataset:
/data/{GROUP}/{ORBIT}/{POL}/images   (H×W×T, float32, nodata=-999)
/data/{GROUP}/{ORBIT}/{POL}/masks    (H×W×T, int8, values: -127=nodata, 0-3=classes)
/data/{GROUP}/{ORBIT}/{POL}/timestamps
/data/{GROUP}/{ORBIT}/{POL}/satellites

IMPORTANT: Filtrage sur mask == 0 strictement pour extraire les données valides.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from load_dataset import MLDatasetLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime
from scipy.ndimage import zoom

# Configuration matplotlib pour LaTeX
mpl.use("pgf")

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ]),
})


def load_group_data(h5_file, group_name, orbit, polarization):
    """
    Charge les données d'un groupe spécifique depuis le nouveau dataset.
    
    Args:
        h5_file: Chemin vers le fichier HDF5
        group_name: Nom du groupe (ex: "ABL001")
        orbit: "ASC" ou "DSC"
        polarization: "HH" ou "HV"
    
    Returns:
        dict avec 'images', 'masks', 'timestamps', 'satellites' ou None si absent
    """
    with h5py.File(h5_file, 'r') as hf:
        path = f"data/{group_name}/{orbit}/{polarization}"
        
        if path not in hf:
            return None
        
        group = hf[path]
        
        return {
            'images': group['images'][:],      # Shape: (H, W, T)
            'masks': group['masks'][:],        # Shape: (H, W, T)
            'timestamps': group['timestamps'][:],
            'satellites': group['satellites'][:] if 'satellites' in group else None
        }


def filter_data_by_mask(images, masks, mask_value=0):
    """
    Filtre les images pour ne garder que les pixels où mask == mask_value.
    Exclut également les valeurs nodata (-999) et NaN.
    
    Args:
        images: Array (H, W, T) ou (H, W)
        masks: Array (H, W, T) ou (H, W) 
        mask_value: Valeur du masque à conserver (par défaut 0)
    
    Returns:
        Array 1D avec uniquement les valeurs valides
    """
    # Créer le masque booléen: pixels valides = (mask == mask_value) 
    # ET (image != -999) ET (image n'est pas NaN)
    valid_mask = (masks == mask_value) & (images > 0) & np.isfinite(images)
    
    # Extraire les valeurs valides
    valid_values = images[valid_mask]
    # print(f'Valid values count: {len(valid_values)} - Min: {np.min(valid_values) if len(valid_values)>0 else "N/A"} - Max: {np.max(valid_values) if len(valid_values)>0 else "N/A"}')
    
    return valid_values


def get_available_groups_by_class(h5_file):
    """
    Liste tous les groupes disponibles organisés par classe.
    
    Returns:
        dict: {class_name: [group_names]}
    """
    groups_by_class = {}
    
    with h5py.File(h5_file, 'r') as hf:
        if 'data' not in hf:
            return groups_by_class
        
        data_group = hf['data']
        
        for group_name in data_group.keys():
            group = data_group[group_name]
            
            if 'class' in group.attrs:
                class_name = group.attrs['class']
                
                if class_name not in groups_by_class:
                    groups_by_class[class_name] = []
                
                groups_by_class[class_name].append(group_name)
    
    return groups_by_class


def plot_hist_updated(h5_file, polarization='HH', save=True):
    """
    Histogrammes avec le nouveau dataset - 3 périodes (janvier, avril, août 2020)
    Filtre: mask == 0 strictement
    Structure: 2x4 avec légende au 8ème subplot
    """
    
    print(f"\nGeneration histogramme pour DSC/{polarization}...")
    
    # Groupes à analyser (un par classe)
    groups = {
        'ICA': 'ICA001',
        'HAG': 'HAG002', 
        'ABL': 'ABL001',
        'ACC': 'ACC002',
        'FOR': 'FOR002',
        'PLA': 'PLA001',
        'ROC': 'ROC001',
        'LAC': 'LAC001'
    }
    
    orbit = 'DSC'
    
    # Périodes d'intérêt
    periods = {
        'jan_2020': (datetime(2020, 1, 1), datetime(2020, 1, 31)),
        'apr_2020': (datetime(2020, 4, 1), datetime(2020, 4, 30)),
        'aug_2020': (datetime(2020, 8, 1), datetime(2020, 8, 31))
    }
    
    # Collecter les données par période
    data_by_period = {}
    
    for class_name, group_name in groups.items():
        print(f"  Traitement {group_name}...")
        
        data_dict = load_group_data(h5_file, group_name, orbit, polarization)
        
        if data_dict is None:
            print(f"    Pas de donnees pour {group_name}/{orbit}/{polarization}")
            continue
        
        images = data_dict['images']
        masks = data_dict['masks']
        timestamps = data_dict['timestamps']
        
        # Convertir timestamps
        dates = [datetime.strptime(ts.decode('utf-8'), '%Y%m%d') for ts in timestamps]
        
        data_by_period[class_name] = {}
        
        # Extraire les données pour chaque période
        for period_name, (start_date, end_date) in periods.items():
            # Trouver les indices dans la période
            indices = [i for i, d in enumerate(dates) if start_date <= d <= end_date]
            
            if not indices:
                data_by_period[class_name][period_name] = None
                continue
            
            # Extraire images et masques pour cette période
            period_images = images[:, :, indices]
            period_masks = masks[:, :, indices]
            
            # Filtrer: garder uniquement pixels avec mask == 0
            valid_values = filter_data_by_mask(period_images, period_masks, mask_value=0)
            
            if len(valid_values) > 0:
                # Convertir en amplitude (sqrt de l'intensité)
                amplitude = np.sqrt(np.abs(valid_values))
                data_by_period[class_name][period_name] = amplitude
                print(f"    {period_name}: {len(amplitude)} pixels valides (mask==0)")
            else:
                data_by_period[class_name][period_name] = None
    
    # Créer la figure
    l = 1.25
    fig, ax = plt.subplots(2, 4, figsize=(17 / l, 6 / l), sharex=True, sharey=True)
    
    colors = ['tab:blue', 'tab:green', 'tab:orange']
    period_names = ['jan_2020', 'apr_2020', 'aug_2020']
    period_labels = ['January', 'April', 'August']
    
    # Limites selon polarisation
    ylim_max = 0.22 if polarization == 'HH' else 0.29
    xlim_max = 1.0
    
    class_list = list(groups.keys())
    
    for cl, class_name in enumerate(class_list):
        if cl >= 8:
            break
        
        row = cl // 4
        col = cl % 4
        
        for i, (period, label, color) in enumerate(zip(period_names, period_labels, colors)):
            if class_name in data_by_period and data_by_period[class_name][period] is not None:
                data = data_by_period[class_name][period]
                
                if len(data) > 0:
                    d, b = np.histogram(data, bins=75, range=(0, 1.5))
                    ax[row, col].bar(
                        b[:-1],
                        d / np.sum(d),
                        width=b[1] - b[0],
                        alpha=0.8 - i*0.15,
                        label=label,
                        edgecolor="black",
                        linewidth=0.2,
                        color=color,
                    )
        
        ax[row, col].set_title(class_name, fontsize=15, fontweight="bold")
        
        if col == 0:
            ax[row, col].set_ylabel("Density", fontsize=15)
        
        ax[row, col].set_xlim(0, xlim_max)
        ax[row, col].set_ylim(0, ylim_max)
        ax[row, col].grid(True, alpha=0.3, linestyle='-')
    
    # Légende globale sous les graphes
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, edgecolor='black', label=label, alpha=0.7) 
                     for color, label in zip(colors, period_labels)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=14, frameon=True, 
               fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.05))

    # # Supprimer le 8ème subplot (inutile maintenant)
    # ax[1, 3].axis('off')
    
    plt.subplots_adjust(hspace=0.22, wspace=0.1)
    
    if save:
        os.makedirs("figure/updated", exist_ok=True)
        filename = f"figure/updated/hist_DSC_{polarization}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight", backend="pgf")
        print(f"  Figure sauvegardee: {filename}")
        plt.close(fig)
    else:
        plt.show()


def create_temporal_plot_updated(h5_file, polarization='HH', save=True):
    """
    Plots temporels avec le nouveau dataset - période PAZ (2020-2023)
    Filtre: mask == 0 strictement
    Structure: 2x4 avec légende au 8ème subplot
    """
    
    print(f"\nGeneration plot temporel pour DSC/{polarization}...")
    
    groups = {
        'ICA': 'ICA001',
        'HAG': 'HAG002',
        'ABL': 'ABL001',
        'ACC': 'ACC002',
        'FOR': 'FOR002',
        'PLA': 'PLA001',
        'ROC': 'ROC001',
        'LAC': 'LAC001'
    }
    
    orbit = 'DSC'
    
    # Sauvegarder le backend actuel et passer en Agg pour l'affichage
    original_backend = mpl.get_backend()
    if save:
        mpl.use('Agg')  # Backend non-interactif pour la sauvegarde
    
    # Créer la figure
    l = 1.25
    fig, ax = plt.subplots(2, 4, figsize=(17 / l, 6 / l), sharex=True, sharey=True)
    
    class_list = list(groups.keys())
    
    for cl, class_name in enumerate(class_list):
        if cl >= 8:
            break
        
        row = cl // 4
        col = cl % 4
        
        group_name = groups[class_name]
        print(f"  Traitement {group_name}...")
        
        data_dict = load_group_data(h5_file, group_name, orbit, polarization)
        
        if data_dict is None:
            print(f"    Pas de donnees pour {group_name}/{orbit}/{polarization}")
            continue
        
        images = data_dict['images']
        masks = data_dict['masks']
        timestamps = data_dict['timestamps']
        
        # Convertir timestamps
        dates = np.array([datetime.strptime(ts.decode('utf-8'), '%Y%m%d') 
                         for ts in timestamps])
        
        # Filtrer période PAZ (>= 2020)
        year_2020 = datetime(2020, 1, 1)
        mask_paz = dates >= year_2020
        
        if not np.any(mask_paz):
            print(f"    Pas de donnees PAZ (>=2020) pour {group_name} - dates: {dates[0]} a {dates[-1]}")
            continue
        
        dates_paz = dates[mask_paz]
        images_paz = images[:, :, mask_paz]
        masks_paz = masks[:, :, mask_paz]
        
        
        # Calculer la moyenne temporelle pour chaque timestamp
        mean_series = []
        std_series = []
        valid_dates = []
        
        for t in range(images_paz.shape[2]):
            img_t = images_paz[:, :, t]
            mask_t = masks_paz[:, :, t]
            # print(f'{np.min(img_t), np.max(img_t), np.unique(img_t) }')
            # Filtrer: mask == 0 (pixels valides, exclut -127=nodata et classes 1,2,3)
            # ET image != -999 (valeurs valides)
            valid_values = filter_data_by_mask(img_t, mask_t, mask_value=0)
            
            if len(valid_values) > 0:
                mean_series.append(np.mean(valid_values))
                std_series.append(np.std(valid_values))
                valid_dates.append(dates_paz[t])
        
        if len(mean_series) == 0:
            print(f"    ATTENTION: Aucune donnee valide pour {group_name}")
            continue
        
        mean_series = np.array(mean_series)
        std_series = np.array(std_series)
        valid_dates = np.array(valid_dates)
        
        # Convertir en dB (10*log10) avec gestion robuste des valeurs invalides
        # Filtrer les valeurs <= 0 avant le log
        mean_series_positive = np.where(mean_series > 0, mean_series, np.nan)
        std_series_positive = np.where(std_series > 0, std_series, np.nan)
        
        mean_db = 10 * np.log10(mean_series_positive)
        std_db = 10 * np.log10(std_series_positive)
        
        # Plot avec les données valides uniquement
        ax[row, col].plot(valid_dates, mean_db, '+-', 
                         color='tab:blue', linewidth=1., 
                         markersize=4, markeredgewidth=1.5)
        ax[row, col].fill_between(valid_dates, 
                                  mean_db - std_db, 
                                  mean_db + std_db, 
                                  alpha=0.3, color='tab:blue')
        
        # Configuration
        ax[row, col].set_title(class_name, fontsize=15, fontweight="bold")
        ax[row, col].grid(True, alpha=0.3, linestyle='--')
        if polarization == 'HH':
            ax[row, col].set_ylim(-30, 2.5)
        else:
            ax[row, col].set_ylim(-35, 2.5)
        if col == 0:
            ax[row, col].set_ylabel("Amplitude (dB)", fontsize=12)
        
        if row == 1:
            # ax[row, col].tick_params(axis='x', rotation=45)
            if polarization == 'HH':
            # Configurer les x labels pour afficher seulement 3 dates
                ax[row, col].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6]))  # Janvier, Juillet
                ax[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format: YYYY-MM
            else:
                ax[row, col].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 8 ,12]))  # Janvier, Juillet
                ax[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format: YYYY-MM
        
        n_valid = len(mean_db)
        print(f"    {n_valid} timestamps avec donnees valides (mask==0)")
    
    # Légende globale en dessous des subplots (2 colonnes)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color='tab:blue', linewidth=2, label='Mean'),
        Patch(facecolor='tab:blue', alpha=0.3, label=r'$\pm$ Std')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=12,
               frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, 0.015))

    # Ajuster l'espacement pour laisser de la place à la légende
    plt.subplots_adjust(hspace=0.25, wspace=0.15, bottom=0.15)
    
    if save:
        os.makedirs("figure/updated", exist_ok=True)
        # Sauvegarder avec le backend PGF
        filename = f"figure/updated/temporal_DSC_{polarization}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight", backend="pgf")
        print(f"  Figure sauvegardee: {filename}")
        plt.close(fig)
        
        # Restaurer le backend original
        mpl.use(original_backend)
    else:
        plt.show()


def create_temporal_plot_dual_orbit_updated(h5_file, polarization='HH', save=True):
    """
    Plot temporel avec orbites DSC (bleu) et ASC (orange) sur le même graphe.
    Filtre: mask == 0 strictement
    
    Layout: 1 ligne x 4 colonnes (ACC, ICA, FOR, ROC)
    Axes X et Y partagés
    """
    
    print(f"\nGeneration plot temporel dual-orbit pour {polarization}...")
    
    # 4 groupes
    groups = {
        'ACC': 'ACC002',
        'ICA': 'ICA004',
        'FOR': 'FOR002',
        'ROC': 'ROC001'
    }
    
    # Créer la figure
    l = 1.25
    fig, ax = plt.subplots(1, 4, figsize=(17 / l, 3.5 / l), sharex=True, sharey=True)
    
    colors = {
        'DSC': 'tab:blue',
        'ASC': 'tab:orange'
    }
    
    class_list = list(groups.keys())
    
    for cl, class_name in enumerate(class_list):
        group_name = groups[class_name]
        print(f"  Traitement {group_name}...")
        
        # Traiter les deux orbites
        for orbit in ['DSC', 'ASC']:
            data_dict = load_group_data(h5_file, group_name, orbit, polarization)
            
            if data_dict is None:
                print(f"    Pas de donnees pour {group_name}/{orbit}")
                continue
            
            images = data_dict['images']
            masks = data_dict['masks']
            timestamps = data_dict['timestamps']
            
            # Convertir timestamps
            dates = np.array([datetime.strptime(ts.decode('utf-8'), '%Y%m%d') 
                             for ts in timestamps])
            
            # Filtrer période PAZ (>= 2020)
            year_2020 = datetime(2020, 1, 1)
            mask_paz = dates >= year_2020
            
            if not np.any(mask_paz):
                continue
            
            dates_paz = dates[mask_paz]
            images_paz = images[:, :, mask_paz]
            masks_paz = masks[:, :, mask_paz]
            
            # Calculer la moyenne pour chaque timestamp
            mean_series = []
            
            for t in range(images_paz.shape[2]):
                img_t = images_paz[:, :, t]
                mask_t = masks_paz[:, :, t]
                
                # Filtrer: mask == 0 strictement
                valid_values = filter_data_by_mask(img_t, mask_t, mask_value=0)
                
                if len(valid_values) > 0:
                    mean_series.append(np.mean(valid_values))
                else:
                    mean_series.append(np.nan)
            
            mean_series = np.array(mean_series)
            
            # Convertir en dB
            mean_db = 10 * np.log10(mean_series + 1e-10)
            
            # Plot
            ax[cl].plot(dates_paz, mean_db, '+-', 
                       color=colors[orbit], linewidth=1, 
                       markersize=6, markeredgewidth=1.5,
                       label=orbit)
            n_valid = np.sum(~np.isnan(mean_db))
            print(f"    {orbit}: {n_valid} timestamps valides (mask==0)")
        
        # Configuration
        ax[cl].set_title(group_name, fontsize=12, fontweight="bold")
        ax[cl].grid(True, alpha=0.3, linestyle='--')
        # ax[cl].legend(fontsize=10)  # Removed individual legends
        if polarization == 'HH':
            ax[cl].set_ylim(-20, 2.5)
        else:
            ax[cl].set_ylim(-25,-2.5)
        
        if cl == 0:
            ax[cl].set_ylabel("Amplitude (dB)", fontsize=12)
            ax[cl].tick_params(axis='y', rotation=0)
        
        # Format dates - afficher uniquement janvier
        if polarization == 'HH':
            ax[cl].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6]))  # Juin
            ax[cl].xaxis.set_major_formatter(mdates.DateFormatter('%Y-01'))
            ax[cl].tick_params(axis='x', rotation=0, labelsize=9)
        else:
            ax[cl].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 8 ,12]))  # Mars, Août, Décembre
            ax[cl].xaxis.set_major_formatter(mdates.DateFormatter('%Y-01'))
            ax[cl].tick_params(axis='x', rotation=0, labelsize=9)
    
    # Légende globale en dessous des subplots (2 colonnes)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='tab:blue', marker='+', linestyle='-', linewidth=1, markersize=6, markeredgewidth=1.5, label='DSC'),
        Line2D([0], [0], color='tab:orange', marker='+', linestyle='-', linewidth=1, markersize=6, markeredgewidth=1.5, label='ASC')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=12,
               frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.015))

    # Ajuster espacement
    plt.subplots_adjust(hspace=0.05, wspace=0.1, bottom=0.2, top=0.9, left=0.06, right=0.98)
    
    if save:
        os.makedirs("figure/updated", exist_ok=True)
        filename = f"figure/updated/temporal_dual_orbit_{polarization}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight", backend="pgf")
        print(f"  Figure sauvegardee: {filename}")
        plt.close(fig)
    else:
        plt.show()


def find_largest_valid_rectangle(image, mask_layer=None, target_size=800, max_mask_pct=10.0):
    """
    Trouve un rectangle sans valeurs NaN/invalides dans une image.
    Accepte jusqu'à max_mask_pct% de pixels avec mask!=0.
    Teste différentes rotations (0°, 90°, 180°, 270°).
    
    Args:
        image: Image SPAN en dB
        mask_layer: Masque binaire (True = mask!=0, False = mask==0), optionnel
        target_size: Taille cible pour le redimensionnement
        max_mask_pct: Pourcentage maximum accepté de pixels avec mask!=0
    
    Returns:
        best_rect: Le rectangle extrait, redimensionné à target_size
        best_mask_rect: Le masque correspondant au rectangle (ou None)
        best_angle: L'angle de rotation utilisé
        best_size: Taille originale du carré extrait
        mask_pct_in_rect: Pourcentage de pixels mask!=0 dans le rectangle
    """
    best_rect = None
    best_mask_rect = None
    best_size = 0
    best_angle = 0
    best_valid_ratio = 0
    best_mask_pct = 100.0
    best_coords = None
    
    # Tester différentes rotations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = image.copy()
            mask_rotated = mask_layer.copy() if mask_layer is not None else None
        else:
            rotated = np.rot90(image, k=angle // 90).copy()
            mask_rotated = np.rot90(mask_layer, k=angle // 90).copy() if mask_layer is not None else None
        
        h, w = rotated.shape
        
        # Créer un masque binaire (1 = valide, 0 = NaN)
        valid_mask = np.isfinite(rotated)
        
        # Essayer des tailles décroissantes
        min_size = 50
        
        for rect_size in range(min(h, w), min_size - 1, -5):
            # Si trop petit pour améliorer, arrêter
            if rect_size <= best_size:
                break
            
            found_good = False
            
            # Essayer différentes positions
            step = max(rect_size // 5, 5)
            
            for i in range(0, h - rect_size + 1, step):
                for j in range(0, w - rect_size + 1, step):
                    # Extraire le rectangle
                    rect_valid = valid_mask[i:i+rect_size, j:j+rect_size]
                    
                    # Calculer le ratio de pixels valides (non-NaN)
                    valid_count = np.sum(rect_valid)
                    total_count = rect_size * rect_size
                    valid_ratio = valid_count / total_count
                    
                    # Il faut 100% de pixels non-NaN
                    if valid_ratio < 1.0:
                        continue
                    
                    # Si on a un masque, calculer le pourcentage de pixels mask!=0
                    if mask_rotated is not None:
                        rect_mask = mask_rotated[i:i+rect_size, j:j+rect_size]
                        mask_count = np.sum(rect_mask)
                        mask_pct = (mask_count / total_count) * 100.0
                    else:
                        mask_pct = 0.0
                    
                    # Accepter si moins de max_mask_pct% de mask!=0 et rect_size > best_size
                    if mask_pct <= max_mask_pct and rect_size > best_size:
                        rect_data = rotated[i:i+rect_size, j:j+rect_size].copy()
                        rect_mask_data = rect_mask.copy() if mask_rotated is not None else None
                        
                        best_rect = rect_data
                        best_mask_rect = rect_mask_data
                        best_size = rect_size
                        best_angle = angle
                        best_mask_pct = mask_pct
                        best_coords = (i, j)
                        found_good = True
                        break
                    
                    # Fallback: garder le meilleur même s'il dépasse le seuil
                    if valid_ratio > best_valid_ratio and best_size == 0:
                        rect_data = rotated[i:i+rect_size, j:j+rect_size].copy()
                        rect_mask_data = rect_mask.copy() if mask_rotated is not None else None
                        
                        best_rect = rect_data
                        best_mask_rect = rect_mask_data
                        best_size = rect_size
                        best_angle = angle
                        best_mask_pct = mask_pct if mask_rotated is not None else 0.0
                        best_valid_ratio = valid_ratio
                
                if found_good:
                    break
            
            # Si on a trouvé un bon rectangle avec cette taille
            if found_good:
                break
    
    # Redimensionner le meilleur rectangle trouvé
    if best_rect is not None and best_size > 0:
        # Redimensionner
        zoom_factor = target_size / best_size
        resized = zoom(best_rect, zoom_factor, order=1)
        resized = resized[:target_size, :target_size]
        
        # Redimensionner aussi le masque si présent
        if best_mask_rect is not None:
            # Pour le masque, utiliser nearest neighbor (order=0)
            resized_mask = zoom(best_mask_rect.astype(float), zoom_factor, order=0) > 0.5
            resized_mask = resized_mask[:target_size, :target_size]
        else:
            resized_mask = None
        
        return resized, resized_mask, best_angle, best_size, best_mask_pct
    
    # Dernier recours
    h, w = image.shape
    center_h, center_w = h // 2, w // 2
    half_size = min(h, w) // 4
    rect = image[center_h-half_size:center_h+half_size, center_w-half_size:center_w+half_size].copy()
    rect_size = min(rect.shape[0], rect.shape[1])
    rect_square = rect[:rect_size, :rect_size]
    
    zoom_factor = target_size / rect_size if rect_size > 0 else 1
    resized = zoom(rect_square, zoom_factor, order=1)
    
    return resized[:target_size, :target_size], None, 0, rect_size, 0.0


def plot_sar_images_updated(h5_file, save=True):
    """
    Images SAR moyennées sur période estivale 2020.
    SPAN = sqrt(HH^2 + HV^2) en dB
    
    Stratégie:
    1. Sélectionne le groupe avec le plus de données valides
    2. Calcule SPAN en acceptant jusqu'à 10% de pixels avec mask != 0
    3. Remplace les pixels mask != 0 par NaN après sélection
    4. Extrait le plus grand rectangle sans NaN (avec rotation si nécessaire)
    
    Structure: 2x4 (7 classes + colorbar)
    """
    
    print(f"\nGeneration images SAR moyennes estivales 2020...")
    
    # Classes de base (FOR en 2ème ligne à la place d'ICA)
    base_classes = ["ABL", "ACC", "HAG", "ICA", "PLA", "FOR", "LAC", "ROC"]
    
    # Période estivale
    date_start = datetime(2020, 7, 1)
    date_end = datetime(2020, 9, 15)
    
    orbit = 'DSC'
    
    # Sélectionner le groupe avec dimensions les plus carrées et taille homogène
    print("  Selection des groupes (dimensions carrees et taille homogene)...")
    
    groups_by_class = get_available_groups_by_class(h5_file)
    selected_groups = {}
    group_shapes = {}
    
    # D'abord, collecter les dimensions de tous les groupes
    with h5py.File(h5_file, 'r') as hf:
        for base_class in base_classes:
            if base_class not in groups_by_class:
                print(f"    Pas de groupe pour {base_class}")
                continue
            
            class_groups = groups_by_class[base_class]
            best_group = None
            best_score = float('inf')
            best_shape = None
            
            for group_name in class_groups:
                path_hv = f"data/{group_name}/{orbit}/HV"
                
                if path_hv not in hf:
                    continue
                
                # Obtenir les dimensions
                images = hf[f"{path_hv}/images"]
                h, w, t = images.shape
                
                # Score: écart à un carré (on veut h ≈ w)
                aspect_ratio = max(h, w) / min(h, w)
                size = min(h, w)
                
                # Favoriser les groupes carrés de taille moyenne (100-300 pixels)
                square_score = abs(aspect_ratio - 1.0)  # 0 = parfait carré
                size_score = abs(size - 150) / 150.0  # Pénalité si loin de 150
                
                total_score = square_score + size_score * 0.5
                
                if total_score < best_score:
                    best_score = total_score
                    best_group = group_name
                    best_shape = (h, w)
            
            if best_group:
                selected_groups[base_class] = best_group
                group_shapes[base_class] = best_shape
                aspect = max(best_shape) / min(best_shape)
                print(f"    {base_class}: {best_group} (shape={best_shape[0]}x{best_shape[1]}, aspect={aspect:.2f})")
            else:
                print(f"    Pas de donnees pour {base_class}")
    
    # Calculer les moyennes SPAN
    print("  Calcul des moyennes SPAN...")
    
    mean_spans_db = {}
    
    with h5py.File(h5_file, 'r') as hf:
        for base_class, group_name in selected_groups.items():
            path_hh = f"data/{group_name}/{orbit}/HH"
            path_hv = f"data/{group_name}/{orbit}/HV"
            
            if path_hh not in hf or path_hv not in hf:
                print(f"    {group_name}: donnees manquantes")
                continue
            
            # Charger données
            images_hh = hf[f"{path_hh}/images"][:]
            masks_hh = hf[f"{path_hh}/masks"][:]
            timestamps_hh = hf[f"{path_hh}/timestamps"][:]
            
            images_hv = hf[f"{path_hv}/images"][:]
            masks_hv = hf[f"{path_hv}/masks"][:]
            timestamps_hv = hf[f"{path_hv}/timestamps"][:]
            
            # Convertir timestamps HH et HV
            dates_hh = [datetime.strptime(ts.decode('utf-8'), '%Y%m%d') 
                       for ts in timestamps_hh]
            dates_hv = [datetime.strptime(ts.decode('utf-8'), '%Y%m%d') 
                       for ts in timestamps_hv]
            
            # Trouver indices dans période estivale pour HH et HV séparément
            indices_hh = [i for i, d in enumerate(dates_hh) 
                         if date_start <= d <= date_end]
            indices_hv = [i for i, d in enumerate(dates_hv) 
                         if date_start <= d <= date_end]
            
            if not indices_hh or not indices_hv:
                print(f"    {group_name}: pas de donnees en periode estivale")
                continue
            
            # Extraire période (en utilisant les indices correspondants)
            period_hh = images_hh[:, :, indices_hh]
            period_hv = images_hv[:, :, indices_hv]
            masks_period_hh = masks_hh[:, :, indices_hh]
            masks_period_hv = masks_hv[:, :, indices_hv]
            
            # Calculer SPAN pour chaque timestamp (prendre le minimum des deux longueurs)
            span_images = []
            n_timestamps = min(len(indices_hh), len(indices_hv))
            
            # Obtenir les dimensions minimales pour recadrer
            h_hh, w_hh = period_hh.shape[:2]
            h_hv, w_hv = period_hv.shape[:2]
            h_min = min(h_hh, h_hv)
            w_min = min(w_hh, w_hv)
            
            for t in range(n_timestamps):
                hh_t = period_hh[:h_min, :w_min, t]
                hv_t = period_hv[:h_min, :w_min, t]
                mask_hh_t = masks_period_hh[:h_min, :w_min, t]
                mask_hv_t = masks_period_hv[:h_min, :w_min, t]
                
                # SPAN = sqrt(HH^2 + HV^2)
                span_t = np.sqrt(hh_t**2 + hv_t**2)
                
                # Filtrer les données invalides (-999)
                span_t = np.where(span_t == -999.0, np.nan, span_t)
                span_t = np.where((hh_t == -999.0) | (hv_t == -999.0), np.nan, span_t)
                
                # Accepter jusqu'à 10% de pixels avec mask != 0
                # On les garde pour l'instant, on les remplacera après sélection
                
                span_images.append(span_t)
            
            # Moyenne sur tous les timestamps (ignorer NaN)
            mean_span = np.nanmean(span_images, axis=0)
            
            # Calculer le pourcentage de pixels avec mask != 0
            # Utiliser la moyenne des masques sur la période (recadrer aux dimensions minimales)
            mean_mask_hh = np.mean(masks_period_hh[:h_min, :w_min, :n_timestamps], axis=2)
            mean_mask_hv = np.mean(masks_period_hv[:h_min, :w_min, :n_timestamps], axis=2)
            
            # Pixels où mask != 0 (si au moins 50% des timestamps ont mask != 0)
            mask_non_zero = ((mean_mask_hh > 0.5) | (mean_mask_hv > 0.5))
            
            total_valid = np.sum(np.isfinite(mean_span))
            non_zero_count = np.sum(mask_non_zero & np.isfinite(mean_span))
            non_zero_pct = (non_zero_count / total_valid * 100) if total_valid > 0 else 0
            
            print(f"    {group_name}: {non_zero_pct:.1f}% pixels avec mask!=0")
            
            # Convertir en dB AVANT de remplacer les pixels mask!=0
            # Cela permet d'extraire un rectangle plus grand qui pourra contenir jusqu'à 10% de mask!=0
            mean_span_db = 10 * np.log10(mean_span + 1e-10)
            
            # Remplacer -inf par NaN (mais garder les pixels mask!=0 pour l'instant)
            mean_span_db = np.where(np.isfinite(mean_span_db), mean_span_db, np.nan)
            
            # Stocker aussi le masque pour traitement ultérieur
            mean_spans_db[base_class] = {
                'span_db': mean_span_db,
                'mask_non_zero': mask_non_zero,
                'non_zero_pct': non_zero_pct
            }
            
            valid_pixels = np.sum(np.isfinite(mean_span_db))
            total_pixels = mean_span_db.size
            print(f"      Avant extraction: {valid_pixels}/{total_pixels} pixels valides ({100*valid_pixels/total_pixels:.1f}%)")
    
    # Afficher la zone complète pour chaque classe (pas de rectangle extrait)
    print("  Preparation des images completes avec masque distorsion...")
    
    extracted_images = {}
    extracted_masks = {}
    target_size = 800
    
    for base_class, data_dict in mean_spans_db.items():
        mean_span_db = data_dict['span_db']
        mask_non_zero = data_dict['mask_non_zero']
        non_zero_pct = data_dict['non_zero_pct']
        
        # Obtenir la forme originale
        h, w = mean_span_db.shape
        orig_shape = group_shapes.get(base_class, (h, w))
        
        print(f"    {base_class}: shape={h}x{w}, {non_zero_pct:.1f}% mask!=0")
        
        # Calculer les statistiques
        valid_data = mean_span_db[np.isfinite(mean_span_db)]
        
        if len(valid_data) == 0:
            print(f"      ATTENTION: Aucune donnee valide")
            extracted_images[base_class] = np.full((target_size, target_size), np.nan)
            extracted_masks[base_class] = np.zeros((target_size, target_size), dtype=bool)
            continue
        
        # Prendre la partie carrée (centré)
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        square_span = mean_span_db[start_h:start_h+size, start_w:start_w+size].copy()
        square_mask = mask_non_zero[start_h:start_h+size, start_w:start_w+size].copy()
        
        # Redimensionner à target_size
        zoom_factor = target_size / size
        resized_span = zoom(square_span, zoom_factor, order=1)[:target_size, :target_size]
        resized_mask = zoom(square_mask.astype(float), zoom_factor, order=0)[:target_size, :target_size] > 0.5
        
        extracted_images[base_class] = resized_span
        extracted_masks[base_class] = resized_mask
        
        # Statistiques
        valid_pixels = np.sum(np.isfinite(resized_span))
        distorted_pixels = np.sum(resized_mask & np.isfinite(resized_span))
        total_pixels = target_size * target_size
        
        print(f"      Redimensionne: {size}x{size} -> {target_size}x{target_size}")
        print(f"      Pixels valides: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        print(f"      Pixels distorsion: {distorted_pixels}/{total_pixels} ({100*distorted_pixels/total_pixels:.1f}%)")
        
        if len(valid_data) > 0:
            print(f"      Stats SPAN: mean={np.mean(valid_data):.2f} dB, "
                  f"range=[{np.min(valid_data):.2f}, {np.max(valid_data):.2f}] dB")
    
    # Créer la figure
    print("  Generation de la figure...")
    
    l = 2.54
    fig, axes = plt.subplots(2, 4, figsize=(17 / l, 10 / l))
    
    # Calculer limites globales pour SPAN uniquement (sans les distorsions)
    all_valid_data = []
    for base_class in extracted_images.keys():
        img = extracted_images[base_class]
        mask = extracted_masks[base_class]
        # Prendre uniquement les pixels valides ET sans distorsion
        valid = img[np.isfinite(img) & ~mask]
        if len(valid) > 0:
            all_valid_data.extend(valid)
    
    if len(all_valid_data) > 0:
        vmin = np.percentile(all_valid_data, 10)
        vmax = np.percentile(all_valid_data, 90)
    else:
        vmin, vmax = -30, 0
    
    # Créer une colormap gray standard
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    cmap_gray = cm.get_cmap('gray')
    colors = cmap_gray(np.linspace(0, 1, 256))
    cmap_custom = ListedColormap(colors)
    cmap_custom.set_bad(alpha=0.0)  # NaN transparents
    
    # Afficher les 7 classes
    for i, base_class in enumerate(base_classes):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if base_class in extracted_images:
            img = extracted_images[base_class]
            mask_distortion = extracted_masks[base_class]
            
            # Créer une image RGB pour combiner gray + rouge
            # D'abord, afficher le SPAN en niveaux de gris
            im = ax.imshow(img, cmap=cmap_custom, vmin=vmin, vmax=vmax, 
                          interpolation='bilinear', aspect='equal')
            
            # Ensuite, superposer les distorsions en rouge (semi-transparent)
            # Créer un masque RGBA pour les distorsions
            distortion_overlay = np.zeros((target_size, target_size, 4))
            distortion_overlay[mask_distortion & np.isfinite(img)] = [1, 0, 0, 0.7]  # Rouge semi-transparent
            
            ax.imshow(distortion_overlay, interpolation='nearest', aspect='equal')
            
            group_name = selected_groups[base_class]
            ax.set_title(group_name, fontsize=10, fontweight="bold")
        else:
            im = ax.imshow(np.full((target_size, target_size), np.nan), 
                          cmap=cmap_custom, vmin=vmin, vmax=vmax, aspect='equal')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(base_class, fontsize=10, fontweight="bold")
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
    
    # Déplacer la colorbar et la légende sous les subplots
    from matplotlib.patches import Patch, Rectangle

    # Colorbar pour les valeurs SPAN
    cbar_ax = fig.add_axes([0.25, 0.045, 0.25, 0.03])  # [left, bottom, width, height]
    im = axes[0, 0].images[0]  # Utiliser une image existante pour la colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    # Supprimer le label automatique et ajouter un label à gauche
    cbar.set_label('')  # Remove default label
    fig.text(0.22, 0.06, 'SPAN (dB)', fontsize=10, ha='right', va='center')

    # Légende pour distorsions SAR (pixels rouges) à côté de la colorbar
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='SAR distorsion', alpha=0.7)
    ]
    fig.legend(handles=legend_elements, loc='center left', ncol=1, fontsize=10, frameon=False, bbox_to_anchor=(0.52, 0.055), bbox_transform=fig.transFigure)

    
    # Ajuster espacement
    plt.subplots_adjust(hspace=0.02, wspace=0.075, top=0.95, left=0.05, 
                       right=0.95, bottom=0.05)
    
    if save:
        os.makedirs("figure/updated", exist_ok=True)
        filename = "figure/updated/sar_images_summer2020.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight", backend="pgf")
        print(f"  Figure sauvegardee: {filename}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Nouveau dataset HDF5
    h5_file = "../../DATASET/PAZTSX_CRYO_ML.hdf5"
    
    print("="*60)
    print("GENERATION DES FIGURES AVEC NOUVEAU DATASET")
    print("="*60)
    print(f"Dataset: {h5_file}")
    print(f"Filtrage: mask == 0 strictement")
    print("Output: figure/updated/")
    print("="*60)
    
    # Vérifier que le fichier existe
    if not os.path.exists(h5_file):
        print(f"\nERREUR: Fichier {h5_file} introuvable!")
        print("Veuillez attendre que la creation du dataset soit terminee.")
        sys.exit(1)
    
    # Générer toutes les figures
    
    # Figure 1: Histogrammes (3 périodes, DSC uniquement)
    # print("\n" + "="*60)
    # print("FIGURE 1: HISTOGRAMMES")
    # print("="*60)
    # plot_hist_updated(h5_file, polarization='HH', save=True)
    # plot_hist_updated(h5_file, polarization='HV', save=True)
    
    # # Figure 2: Plots temporels simples (DSC uniquement)
    # print("\n" + "="*60)
    # print("FIGURE 2: PLOTS TEMPORELS DSC")
    # print("="*60)
    # create_temporal_plot_updated(h5_file, polarization='HH', save=True)
    # create_temporal_plot_updated(h5_file, polarization='HV', save=True)
    
    # Figure 3: Plots temporels dual-orbit (DSC + ASC)
    print("\n" + "="*60)
    print("FIGURE 3: PLOTS TEMPORELS DUAL-ORBIT")
    print("="*60)
    create_temporal_plot_dual_orbit_updated(h5_file, polarization='HH', save=True)
    create_temporal_plot_dual_orbit_updated(h5_file, polarization='HV', save=True)
    
    # # Figure 4: Images SAR moyennes estivales
    # print("\n" + "="*60)
    # print("FIGURE 4: IMAGES SAR MOYENNES")
    # print("="*60)
    # plot_sar_images_updated(h5_file, save=True)
    
    print("\n" + "="*60)
    print("TOUTES LES FIGURES GENEREES!")
    print("="*60)
    print("Emplacement: figure/updated/")
