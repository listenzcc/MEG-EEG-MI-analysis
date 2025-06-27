"""
File: 5.read.fsaverage.py
Author: Chuncheng Zhang
Date: 2025-06-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read fsaverage files for fwd.

    By looking at Display sensitivity maps for EEG and MEG sensors plot the sensitivity maps for EEG and compare it with the MEG, can you justify the claims that:
    <https://mne.tools/stable/auto_examples/forward/forward_sensitivity_maps.html#ex-sensitivity-maps>
    <https://mne.tools/stable/generated/mne.sensitivity_map.html#mne.sensitivity_map>

    - MEG is not sensitive to radial sources
    - EEG is more sensitive to deep sources

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-27 ------------------------
# Requirements and constants
from util.easy_import import *

SUBJECT = 'fsaverage'
SUBJECT_DIR = mne.datasets.fetch_fsaverage()
SUBJECTS_DIR = SUBJECT_DIR.parent
print(SUBJECTS_DIR, SUBJECT)

# %% ---- 2025-06-27 ------------------------
# Function and class


# %% ---- 2025-06-27 ------------------------
# Play ground

# parc, str, The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
parc = 'aparc_sub'

labels_parc = mne.read_labels_from_annot(
    SUBJECT, parc=parc, subjects_dir=SUBJECTS_DIR)
print(labels_parc)

labels_parc_dict = {e.name: e for e in labels_parc}
print(labels_parc_dict)


def _color(vec):
    return '#' + ''.join(hex(int(255 * e)).replace('x', '')[-2:] for e in vec[:3])


df = pd.DataFrame([(v.name, len(v.values), v.color, v.pos, v.vertices)
                   for v in tqdm(labels_parc, 'Generate DataFrame')],
                  columns=['name', 'num', 'rgba', 'pos', 'vertices'])

df['xyz'] = df['pos'].map(lambda e: np.mean(e, axis=0))
df['cx'] = df['xyz'].map(lambda e: e[0])
df['cy'] = df['xyz'].map(lambda e: e[1])
df['cz'] = df['xyz'].map(lambda e: e[2])
df['color'] = df['rgba'].map(lambda e: _color(e))

labels_parc_df = df[['name', 'num', 'color',
                     'cx', 'cy', 'cz', 'pos', 'vertices']]
print(labels_parc_df)

# %% ---- 2025-06-27 ------------------------
# Pending


# %% ---- 2025-06-27 ------------------------
# Pending
