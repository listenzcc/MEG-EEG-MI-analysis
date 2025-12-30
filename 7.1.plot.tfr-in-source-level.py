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


# %%
class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent
    src_path = subject_dir.joinpath('bem', 'fsaverage-ico-5-src.fif')
    src = mne.read_source_spaces(src_path)


subject = SubjectFsaverage()
parc = 'aparc'
parc = 'aparc_sub'
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
labels_parc_df = pd.DataFrame([(e.name, e)
                              for e in labels_parc], columns=['name', 'label'])
display(labels_parc_df)


ROI_labels = [
    # meg-alpha
    ('alpha', 'precentral_5-lh'),
    ('alpha', 'precentral_8-lh'),
    ('alpha', 'postcentral_7-lh'),
    ('alpha', 'postcentral_9-lh'),
    ('alpha', 'postcentral_4-lh'),
    ('alpha', 'supramarginal_6-lh'),
    ('alpha', 'supramarginal_9-lh'),
    ('alpha', 'superiorparietal_5-lh'),
    ('alpha', 'superiorparietal_3-lh'),
    # meg-beta
    ('beta', 'postcentral_5-lh'),
    ('beta', 'postcentral_4-lh'),
    ('beta', 'superiorparietal_5-lh'),
    ('beta', 'precentral_4-lh'),
    ('beta', 'precentral_6-lh'),
    ('beta', 'precentral_8-lh'),
    ('beta', 'precentral_5-lh'),
    ('beta', 'precentral_3-lh'),
    ('beta', 'superiorfrontal_18-lh'),
    ('beta', 'superiorfrontal_17-lh'),
    ('beta', 'superiorfrontal_12-lh'),
    ('beta', 'caudalmiddlefrontal_6-lh'),
    ('beta', 'postcentral_7-lh'),
    ('beta', 'postcentral_3-lh'),
]

label_names = [e[1] for e in ROI_labels if 'central' in e[1]]
display(label_names)

# %% ---- 2025-11-12 ------------------------
# Function and class


class FsaverageSubject:
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


def find_stc(subject, band, mode, evt):
    folder = Path(f'./data/tfr-stc-{band}/')
    subjects = [e.name for e in folder.iterdir()]
    subject = [e for e in subjects if e.lower().startswith(subject.lower())][0]
    print(subjects, subject)
    fpath = folder.joinpath(f'{subject}/{mode}-evt{evt}.stc')
    return fpath


def read_compute_stc(subject, band, mode, evt, tmin, tmax):
    fpath = find_stc(subject, band, mode, evt)
    print(f'{fpath=}')

    stc = mne.read_source_estimate(fpath)
    stc.crop(tmin=tmin, tmax=tmax)
    stc.subject = FsaverageSubject.subject
    print(f'{stc=}')

    # Ratio by the baseline (-1, 0)
    data = stc.data
    times = stc.times
    base_data = np.mean(data[:, times < 0], axis=1)
    extended_base_data = np.stack([base_data for e in times]).T
    print(f'{data.shape=}, {times.shape=}, {base_data.shape=}, {extended_base_data.shape=}')

    # Convert into dB
    data = 10 * np.log10(data/extended_base_data)
    stc.data = data

    return stc


def screenshot(subject, band, mode, evt):
    if subject == 'average':
        stcs = [read_compute_stc(s, band, mode, evt, tmin, tmax) for s in
                ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']]
        data = np.mean([e.data for e in stcs], axis=0)
        stc = stcs[0]
        stc.data = data
        clim = {'kind': 'value', 'pos_lims': (1, 2, 3)}
    else:
        stc = read_compute_stc(subject, band, mode, evt, tmin, tmax)
        clim = {'kind': 'value', 'pos_lims': (2, 3, 5)}

    for t in [5]:  # tqdm([0, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5]):
        title = f'{subject}-{mode}-{evt}-{band}-{t:0.1f}'
        if t == 5:
            stc.data *= 0

        brain = stc.plot(
            initial_time=t,
            hemi="split",
            views=["lat"],
            subjects_dir=FsaverageSubject.subjects_dir,
            transparent=True,
            colormap='RdBu',
            clim=clim,
            size=(1600, 800),
            colorbar=False,
            show_traces=False,
            title=title,
            brain_kwargs=brain_kwargs,
        )
        brain.add_text(0.5, 0.9, f'{title}', 'title', font_size=16)
        brain.clear_glyphs()
        brain.save_image(OUTPUT_DIR.joinpath(f'{title}.png'))
        brain.close()
    return


def display(subject, band, mode, evt):
    if subject == 'average':
        stcs = [read_compute_stc(s, band, mode, evt, tmin, tmax) for s in
                ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']]
        data = np.mean([e.data for e in stcs], axis=0)
        stc = stcs[0]
        stc.data = data
        clim = {'kind': 'value', 'pos_lims': (1, 2, 3)}
    else:
        stc = read_compute_stc(subject, band, mode, evt, tmin, tmax)
        clim = {'kind': 'value', 'pos_lims': (2, 3, 5)}

    title = f'{subject}-{mode}-{evt}-{band}'
    brain = stc.plot(
        initial_time=0,
        hemi="split",
        views=["lat"],
        subjects_dir=FsaverageSubject.subjects_dir,
        transparent=True,
        colormap='RdBu',
        clim=clim,
        size=(1600, 800),
        colorbar=True,
        title=title,
        brain_kwargs=brain_kwargs,
    )
    brain.add_text(0.5, 0.9, f'{title}', 'title', font_size=16)

    # ['postcentral-lh', 'precentral-lh']:
    for label_name in label_names:
        label = labels_parc_df.query(f'name=="{label_name}"')[
            'label'].values[0]
        brain.add_label(label, hemi='lh',
                        color='red', borders=True)
        brain.add_label(label, hemi='lh',
                        color='red', alpha=0.5, borders=False)

    brain.show()
    return


# %% ---- 2025-11-12 ------------------------
# Play ground
brain_kwargs = {'alpha': 1.0,
                'background': "white",
                'foreground': 'black',
                'cortex': "low_contrast"}

subject = 'average'  # 'S01' ~ 'S10' or 'average'
band = 'alpha'  # 'alpha' or 'beta'
mode = 'meg'
evt = '1'  # '1' ~ '5'
tmin, tmax = -1, 5

mode = 'meg'
subject = 'S07'

# ! Screenshot all combinations
# for band in ['alpha', 'beta']:
#     for evt in ['1', '2', '3', '4', '5']:
#         screenshot(subject, band, mode, evt)

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
