import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

source_dir = '../Data/demo_pem/'

tracker_only_err = pd.read_csv('pose_error_tracker_only.csv')
sam6d_err = pd.read_csv('pose_error.csv')
ours_err = pd.read_csv('pose_error_tracker.csv')

# window = (580, 650)
# tracker_only_err = tracker_only_err.iloc[window[0]:window[1]]
# sam6d_err = sam6d_err.iloc[window[0]:window[1]]
# ours_err = ours_err.iloc[window[0]:window[1]]


print(sam6d_err)
print(tracker_only_err)

reset = 623


fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(range(window[0], window[1]), sam6d_err['Translation']*1000, label="Sous-module d'estimation de pose", color='#fddb6e', alpha=1.0)
axs[0].plot(range(window[0], window[1]), ours_err['Translation']*1000, label="Module de suivi d'objet avec récupération du suivi", color='#85b8ff', alpha=1.0)
axs[0].plot(range(window[0], window[1]), tracker_only_err['Translation']*1000, label="Sous-module de suivi d'objet", color='#ff8c83', alpha=1.0)


axs[1].plot(range(window[0], window[1]), sam6d_err['Rotation'], label='SAM6D', color='#fddb6e', alpha=1.0)
axs[1].plot(range(window[0], window[1]), ours_err['Rotation'], label='Ours', color='#85b8ff', alpha=1.0)
axs[1].plot(range(window[0], window[1]), tracker_only_err['Rotation'], label='Tracker Only', color='#ff8c83', alpha=1.0)


axs[0].set_ylabel('Erreur de translation (mm)')
axs[1].set_ylabel('Erreur de rotation (°)')
axs[1].set_xlabel("Index de l'image")

axs[0].grid(True, linestyle='--', alpha=0.5)
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[0].set_facecolor("#f4f4f4")
axs[1].set_facecolor("#f4f4f4")
# axs[0].set_title()
# axs[1].set_title()

axs[0].set_ylim(-10, 1000)
axs[0].set_xlim(window[0], window[1])
ymin, ymax = axs[0].get_ylim()
axs[0].vlines(x=reset, color='red', linestyle='--', ymin=ymin, ymax=ymax)
axs[1].set_ylim(-1, 180)
axs[1].set_xlim(window[0], window[1])
ymin, ymax = axs[1].get_ylim()
axs[1].vlines(x=reset, color='red', linestyle='--', ymin=ymin, ymax=ymax)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # reserve space at bottom

plt.savefig('zoomed_in_error_comparison.png', dpi=300)
plt.show()