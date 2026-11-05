"""
File: rev2-compare-MEG-EEG-areas.py
Author: Chuncheng Zhang
Date: 2026-11-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compare area results for EEG and MEG.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-11-04 ------------------------
# Requirements and constants
from util.easy_import import *

meg_df_path = Path('./data/MVPA.FBCSP.megAreas.vote.results/df_vote.csv')
eeg_df_path = Path('./data/MVPA.FBCSP.eegAreas.vote.results/df_vote.csv')

OUTPUT_DIR = Path('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-11-04 ------------------------
# Function and class


# %% ---- 2026-11-04 ------------------------
# Play ground
df_meg = pd.read_csv(meg_df_path, index_col=0)
df_meg['modality'] = 'MEG'
df_eeg = pd.read_csv(eeg_df_path, index_col=0)
df_eeg['modality'] = 'EEG'

table = pd.concat([df_meg, df_eeg], axis=0)
table.index = range(len(table))
table.to_csv(OUTPUT_DIR.joinpath('df_compare.csv'))
table


# %% ---- 2026-11-04 ------------------------
# Pending
df = table.copy()
sns.barplot(df, x='chMark', y='acc', hue='chHemi', fill='modality')
plt.show()


# %% ---- 2026-11-04 ------------------------
# Pending
# --- Option 1: hue by modality, facets by chMark × chHemi ---
g = sns.catplot(
    data=df,
    x="chMark",
    y="acc",
    hue="chHemi",         # separate panels per hemisphere
    col="modality",       # distinguish MEG vs EEG by color
    kind="bar",
    errorbar="se",        # show standard error (use "sd" for std dev, or None)
    capsize=0.1,
    height=4,
    aspect=1.1,
)
g.set_axis_labels("Channel Mark", "Accuracy")
g.set_titles(col_template="Hemi = {col_name}")
g.figure.suptitle("Accuracy by chMark, Hemisphere, and Modality", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR.joinpath('Acc by chMark, Hemisphere, and Modality.png'))
plt.show()


# %%
