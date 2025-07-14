"""
File: 8.1.plot.snr.py
Author: Chuncheng Zhang
Date: 2025-07-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot SNR

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-10 ------------------------
# Requirements and constants
import joblib
from util.easy_import import *

# data_directory = Path('./data/tfr-stc-alpha')
# data_directory = Path('./data/tfr-stc-beta')
data_directory = Path('./data/fsaverage')
compile = re.compile(r'^(?P<me>[a-z]+)-evt(?P<evt>\d+).stc-lh.stc')


# %% ---- 2025-07-10 ------------------------
# Function and class

class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


def find_stc_files(pattern: str):
    found = list(data_directory.rglob(pattern))
    return found


def mk_stc_file_table(stc_files: list):
    data = []
    for p in stc_files:
        name = p.name
        sub = p.parent.name
        dct = compile.search(name).groupdict()
        dct.update({
            'subject': sub,
            'path': p,
            'virtualPath': p.parent.joinpath(p.name.replace('.stc-lh', ''))
        })
        data.append(dct)
    return pd.DataFrame(data)


if False:
    # %% ---- 2025-07-10 ------------------------
    # Play ground
    subject = SubjectFsaverage()
    stc_files = find_stc_files('*.stc-lh.stc')
    df = mk_stc_file_table(stc_files)

    print('Reading stc')
    df['stc'] = df['virtualPath'].map(mne.read_source_estimate)

    # %%
    df

    # %% ---- 2025-07-10 ------------------------
    # Pending
    stuff_directory = Path('./data/fsaverage')
    scalings = {'meg': 1e15, 'eeg': 1e6}
    mode = 'meg'
    mat = None
    n = 0

    for i, row in df.query(f'me=="{mode}"').iterrows():
        subject = row['subject']
        stc = row['stc']
        stc.data /= scalings[mode]
        _mode = row['me']
        stuff = joblib.load(stuff_directory.joinpath(
            subject, 'stuff-estimate-snr.dump'))
        cov = stuff[f'cov_{_mode}']
        fwd = stuff[f'fwd_{_mode}']
        info = stuff[f'info_{_mode}']
        snr = stc.estimate_snr(info, fwd, cov)
        if mat is None:
            mat = snr.data
            n = 1
        else:
            mat += snr.data
            n += 1

    snr.data = mat / n

    print(stc, mode, subject)
    (snr, stc, cov, fwd, info)

    # %%
    stc.save(f'data/stc-snr-{mode}.stc')


# %%
snr = mne.read_source_estimate('data/stc-snr-meg.stc')
snr.subject = SubjectFsaverage.subject
kwargs = dict(
    initial_time=0,
    hemi="split",
    views=["lat", "med"],
    subjects_dir=SubjectFsaverage.subjects_dir,
    # size=(600, 600),
    # clim=dict(kind="value", lims=(-100, -70, -40)),
    transparent=True,
    colormap="viridis",
)
brain = snr.plot(**kwargs)
# brain = snr.plot(hemi='both')
input()

# %% ---- 2025-07-10 ------------------------
# Pending
# %%

# %%
