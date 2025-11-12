"""
File: 7.1.plot.tfr-in-source-level.py
Author: Chuncheng Zhang
Date: 2025-11-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot tfr in source level. (Single subject)

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-12 ------------------------
# Requirements and constants
from util.easy_import import *

# %%
OUTPUT_DIR = Path(f'./data/TFR-Source/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-11-12 ------------------------
# Function and class
class FsaverageSubject:
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


def find_stc(subject, band, mode, evt):
    folder = Path(f'./data/tfr-stc-{band}/')
    subjects = [e.name for e in folder.iterdir()]
    subject = [e for e in subjects if e.startswith(subject)][0]
    print(subjects, subject)
    fpath = folder.joinpath(f'{subject}/{mode}-evt{evt}.stc')
    return fpath

def screenshot(subject, band, mode, evt):
    fpath = find_stc(subject, band, mode, evt)
    print(f'{fpath=}')

    stc = mne.read_source_estimate(fpath)
    stc.crop(tmin=tmin, tmax=tmax)
    stc.subject = FsaverageSubject.subject
    print(f'{stc=}')

    # Ratio by the baseline (-1, 0)
    data = stc.data
    times = stc.times
    base_data = np.mean(data[:, times<0], axis=1)
    extended_base_data = np.stack([base_data for e in times]).T
    print(f'{data.shape=}, {times.shape=}, {base_data.shape=}, {extended_base_data.shape=}')

    # Convert into dB
    data = 10 * np.log10(data/extended_base_data)
    stc.data = data


    for t in tqdm([0, 0.2, 0.5, 1, 2, 3]):
        title=f'{subject}-{mode}-{evt}-{band}-{t}'

        brain = stc.plot(
            initial_time=t,
            hemi="split",
            views=["lat"],
            subjects_dir=FsaverageSubject.subjects_dir,
            transparent=True,
            colormap='RdBu',
            clim={'kind': 'value', 'pos_lims':(2, 3, 5)},
            # clim=dict(kind="value", pos_lims=(10, 20, 30)),
            size=(1600, 800),
            colorbar=False,
            show_traces=True,
            title=title,
            brain_kwargs=brain_kwargs,
            )

        brain.add_text(0.5, 0.9, f'{title}', 'title', font_size=16)
        brain.clear_glyphs()
        brain.save_image(OUTPUT_DIR.joinpath(f'{title}-{t=}.png'))
        brain.close()

def display(subject, band, mode, evt):
    fpath = find_stc(subject, band, mode, evt)
    print(f'{fpath=}')

    stc = mne.read_source_estimate(fpath)
    stc.crop(tmin=tmin, tmax=tmax)
    stc.subject = FsaverageSubject.subject
    print(f'{stc=}')

    # Ratio by the baseline (-1, 0)
    data = stc.data
    times = stc.times
    base_data = np.mean(data[:, times<0], axis=1)
    extended_base_data = np.stack([base_data for e in times]).T
    print(f'{data.shape=}, {times.shape=}, {base_data.shape=}, {extended_base_data.shape=}')

    # Convert into dB
    data = 10 * np.log10(data/extended_base_data)
    stc.data = data

    title=f'{subject}-{mode}-{evt}-{band}'

    brain = stc.plot(
        initial_time=0,
        hemi="split",
        views=["lat"],
        subjects_dir=FsaverageSubject.subjects_dir,
        transparent=True,
        colormap='RdBu',
        clim={'kind': 'value', 'pos_lims':(2, 3, 5)},
        # clim=dict(kind="value", pos_lims=(10, 20, 30)),
        size=(1600, 800),
        colorbar=False,
        title=title,
        brain_kwargs=brain_kwargs,
        )

    brain.add_text(0.5, 0.9, f'{title}', 'title', font_size=16)
    brain.clear_glyphs()

    brain.show()

# %% ---- 2025-11-12 ------------------------
# Play ground
brain_kwargs = {'alpha':1.0,
                'background':"white",
                'foreground':'black',
                'cortex':"low_contrast"}

subject = 'S07'
band='alpha'
mode='meg'
evt='1'
tmin, tmax = -1, 5

while True:
    prompt = 'Example: $subject $band $mode $evt (Ctrl+c to escape) >>'
    try:
        subject, band, mode, evt = input(prompt).split(' ')
    except KeyboardInterrupt:
        break
    except:
        continue
    print(subject, band, mode, evt)

    display(subject, band, mode, evt)

print('Byebye.')
sys.exit(0)

# %% ---- 2025-11-12 ------------------------
# Pending


# %% ---- 2025-11-12 ------------------------
# Pending


