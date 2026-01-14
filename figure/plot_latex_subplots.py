import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import os
import datetime

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
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 13,
})

# Path to the data folders
lstm_dir = 'paper1_final/S2_formated_tres_lstm_20251222_083822_final'
convlstm_dir = 'paper1_final/S2_formated_tres_convlstm2d_20251222_102508_final'

# List of folds
folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# Classes to plot
classes_to_plot = [2, 4, 5]  # ROC, ABL, ICA
class_names = {
    2: 'ROC',
    4: 'ABL',
    5: 'ICA'
}

# Dates from log
start_past = datetime.date(2020, 1, 1)
end_past = datetime.date(2021, 6, 30)
start_future = datetime.date(2021, 7, 1)
end_future = datetime.date(2021, 12, 31)

# Generate full time axes
n_past = 44
n_future = 12
time_past = [start_past + (end_past - start_past) * i / (n_past - 1) for i in range(n_past)]
time_future = [start_future + (end_future - start_future) * i / (n_future - 1) for i in range(n_future)]

# Find start index for past from 2020-07
start_date = datetime.date(2020, 7, 1)
start_index = next(i for i, d in enumerate(time_past) if d >= start_date)

# Slice time_past
time_past_plot = time_past[start_index:]
l = 1.45
fig, axs = plt.subplots(1, 3, figsize=(17 / l, 5 / l), sharex=True, sharey=True)


# Labels for legend
legend_labels = ['X_past mean', 'X_future true', 'X_future pred LSTM', 'X_future pred CONVLSTM']

for idx, selected_class in enumerate(classes_to_plot):
    class_name = class_names[selected_class]
    print(f"Processing class {selected_class}: {class_name}")
    
    ax = axs[idx]
    
    # Collect data for the selected class across all folds
    X_past_list = []
    X_future_true_list = []
    X_future_pred_lstm_list = []
    X_future_pred_convlstm_list = []
    
    for fold in folds:
        fold_dir_lstm = os.path.join(lstm_dir, fold)
        fold_dir_convlstm = os.path.join(convlstm_dir, fold)
        y = np.load(os.path.join(fold_dir_lstm, 'y.npy'))
        mask = y == selected_class
        
        if np.sum(mask) == 0:
            continue
        
        X_past = np.load(os.path.join(fold_dir_lstm, 'X_past.npy'))[mask]  # shape (n, 44, 1, 8, 8)
        X_future_true = np.load(os.path.join(fold_dir_lstm, 'X_future.npy'))[mask]  # (n, 12, 1, 8, 8)
        X_future_pred_lstm = np.load(os.path.join(fold_dir_lstm, 'predictions.npy'))[mask]  # (n, 12, 1, 8, 8)
        X_future_pred_convlstm = np.load(os.path.join(fold_dir_convlstm, 'predictions.npy'))[mask]  # (n, 12, 1, 8, 8)
        
        # Average over spatial dims and channel: (n, 44) for past, (n,12) for future
        X_past_avg = np.mean(X_past, axis=(2,3,4))  # (n, 44)
        X_future_true_avg = np.mean(X_future_true, axis=(2,3,4))  # (n, 12)
        X_future_pred_lstm_avg = np.mean(X_future_pred_lstm, axis=(2,3,4))  # (n, 12)
        X_future_pred_convlstm_avg = np.mean(X_future_pred_convlstm, axis=(2,3,4))  # (n, 12)
        
        X_past_list.append(X_past_avg)
        X_future_true_list.append(X_future_true_avg)
        X_future_pred_lstm_list.append(X_future_pred_lstm_avg)
        X_future_pred_convlstm_list.append(X_future_pred_convlstm_avg)
    
    # Concatenate all
    X_past_all = np.concatenate(X_past_list, axis=0)  # (total_n, 44)
    X_future_true_all = np.concatenate(X_future_true_list, axis=0)  # (total_n, 12)
    X_future_pred_lstm_all = np.concatenate(X_future_pred_lstm_list, axis=0)  # (total_n, 12)
    X_future_pred_convlstm_all = np.concatenate(X_future_pred_convlstm_list, axis=0)  # (total_n, 12)
    
    # Compute mean and std over groups (samples)
    X_past_mean = np.mean(X_past_all, axis=0)
    X_past_std = np.std(X_past_all, axis=0)
    
    X_future_true_mean = np.mean(X_future_true_all, axis=0)
    X_future_true_std = np.std(X_future_true_all, axis=0)
    X_future_pred_lstm_mean = np.mean(X_future_pred_lstm_all, axis=0)
    X_future_pred_lstm_std = np.std(X_future_pred_lstm_all, axis=0)
    X_future_pred_convlstm_mean = np.mean(X_future_pred_convlstm_all, axis=0)
    X_future_pred_convlstm_std = np.std(X_future_pred_convlstm_all, axis=0)
    
    # Slice past data
    X_past_mean_plot = X_past_mean[start_index:]
    X_past_std_plot = X_past_std[start_index:]
    
    # Plot on the subplot
    # Plot X_past
    ax.fill_between(time_past_plot, X_past_mean_plot - X_past_std_plot, X_past_mean_plot + X_past_std_plot, alpha=0.3)
    ax.plot(time_past_plot, X_past_mean_plot, label='Past')
    
    # Plot X_future true and pred
    ax.fill_between(time_future, X_future_true_mean - X_future_true_std, X_future_true_mean + X_future_true_std, alpha=0.3, color='green')
    ax.plot(time_future, X_future_true_mean, label='Ground Truth', color='green')
    
    ax.fill_between(time_future, X_future_pred_lstm_mean - X_future_pred_lstm_std, X_future_pred_lstm_mean + X_future_pred_lstm_std, alpha=0.3, color='red')
    ax.plot(time_future, X_future_pred_lstm_mean, label='Pred. LSTM', color='red', linestyle='--')
    
    ax.fill_between(time_future, X_future_pred_convlstm_mean - X_future_pred_convlstm_std, X_future_pred_convlstm_mean + X_future_pred_convlstm_std, alpha=0.3, color='blue')
    ax.plot(time_future, X_future_pred_convlstm_mean, label='Pred. CONV2DLSTM', color='blue', linestyle='-.')
    
    if idx == 0:
        ax.set_ylabel('Mean value (dB)', fontsize=14)
    if idx == 1:
        ax.set_xlabel('Date')
    ax.set_title(f'\\textbf{{{class_name}}}')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Place legend below the plots
handles = axs[0].lines  # The 4 line plots
labels = ['Past', 'Ground Truth', 'Pred. LSTM', 'Pred. CONV2DLSTM']
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.075))

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig('plot_subplots.pdf', bbox_inches='tight')
plt.close()