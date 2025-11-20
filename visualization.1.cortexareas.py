"""
File: visualization.1.cortexareas.py
Author: Chuncheng Zhang
Date: 2025-11-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Visualization the ERDs for cortex areas.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-19 ------------------------
# Requirements and constants
import joblib
from nilearn import datasets
from util.easy_import import *


# %%
CACHE_DIR = Path(f'./data/cache/')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# %% ---- 2025-11-19 ------------------------
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
    base_data = np.mean(data[:, times < 0], axis=1)
    extended_base_data = np.stack([base_data for e in times]).T
    print(f'{data.shape=}, {times.shape=}, {base_data.shape=}, {extended_base_data.shape=}')

    # Convert into dB
    data = 10 * np.log10(data/extended_base_data)
    stc.data = data

    title = f'{subject}-{mode}-{evt}-{band}'

    brain = stc.plot(
        initial_time=0,
        hemi="split",
        views=["lat"],
        # surface='pial',
        subjects_dir=FsaverageSubject.subjects_dir,
        transparent=True,
        colormap='RdBu',
        clim={'kind': 'value', 'pos_lims': (2, 3, 5)},
        # clim=dict(kind="value", pos_lims=(10, 20, 30)),
        size=(1600, 800),
        colorbar=False,
        title=title,
        brain_kwargs=brain_kwargs,
    )

    brain.add_text(0.5, 0.9, f'{title}', 'title', font_size=16)
    # brain.clear_glyphs()

    for name, labels in roi_labels.items():
        for label in labels:
            brain.add_label(label, borders=True, alpha=0.7,
                            color=roi_colors[name])

    brain.show()


def extract_motor_areas_from_labels():
    """
    从精细分区标签中提取运动皮层区域
    """

    motor_regions = {
        'PMd': [],
        'PMv': [],
        'SMA': [],
        'SM1_hand': []
    }

    for name, area_nums in motor_region_nums.items():
        for area_idx in area_nums:
            filtered = label_table.query(f'Name == "{area_idx:03d}"')
            motor_regions[name].extend(filtered['Label'].tolist())

    # for name, area_names in motor_region_names.items():
    #     for aname in area_names:
    #         area_idx = atlas_map[aname]
    #         print(f'{aname=}, {area_idx=}')
    #         filtered = label_table.query(f'Name == "{area_idx:03d}"')
    #         motor_regions[name].extend(filtered['Label'].tolist())

    return motor_regions


# %% ---- 2025-11-19 ------------------------
# Play ground

# %%
# 下载 200-parcel, 17-network 版本
# atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)

# labels 是一个列表： index = parcel 编号
# labels = atlas['labels']

# # atlas map (nii.gz) 在 atlas['maps']
# # labels[0] 是 background，因此 label 1~200 对应 labels[1:]
# atlas_map = {}
# for i, name in enumerate(labels[1:], start=1):
#     # print(f"{i}\t{name}")
#     atlas_map[name.replace('17Networks_', '')] = i
# print(atlas_map)

# labels = joblib.load('./data/schaefer_labels.dump')
# label_table = []
# smoothed_dir = CACHE_DIR.joinpath('smoothed-schaefer-labels/')
# smoothed_dir.mkdir(parents=True, exist_ok=True)
# for label in labels:
#     label_name = f'smoothed-{label.name}-fsaverage5-{label.hemi}.label'
#     name = label.name

#     try:
#         label = mne.read_label(smoothed_dir.joinpath(label_name))
#         label.name = name
#     except:
#         label.smooth()
#         label.save(smoothed_dir.joinpath(label_name))

#     label_table.append(
#         (label.name, label.hemi, label, len(label.vertices)))

# label_table = pd.DataFrame(label_table, columns=[
#                            'Name', 'Hemi', 'Label', 'NumVertices'])
# print(label_table)

# Load the labels
subject = FsaverageSubject()
parc = "aparc_sub"
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir
)
labels_parc_df = pd.DataFrame(
    [(e.name, e) for e in labels_parc], columns=["name", "label"]
)
labels_parc_df = labels_parc_df[labels_parc_df['name'].map(
    lambda e: not e.startswith('unknown'))]
labels_parc_df.index = labels_parc_df['name']
labels_parc_df

# %%
roi_names = {
    'central': [f'postcentral_{i}' for i in [3, 4, 5, 6, 7, 8, 9]] + [f'precentral_{i}' for i in [5, 6, 7, 8]],
    'pariental': [f'superiorparietal_{i}' for i in [11, 12]] + [f'inferiorparietal_{i}' for i in [1, 4, 3, 8]],
    'occipital': [f'lateraloccipital_{i}' for i in [2, 3, 6, 7, 8, 9]]
}
roi_colors = {
    'central': 'cyan',
    'pariental': 'green',
    'occipital': 'blue'
}
roi_labels = {k: [labels_parc_df.loc[e+'-lh', 'label'] for e in v]
              for k, v in roi_names.items()}
roi_labels

# %% ---- 2025-11-19 ------------------------
# Pending
if __name__ == '__main__':
    brain_kwargs = {'alpha': 1.0,
                    'background': "white",
                    'foreground': 'black',
                    'cortex': "low_contrast"}

    subject = 'S07'
    band = 'alpha'
    mode = 'meg'
    evt = '1'
    tmin, tmax = -1, 5

    # for band in ['alpha', 'beta']:
    #     for evt in ['1', '2', '3', '4', '5']:
    #         screenshot(subject, band, mode, evt)

    while True:
        prompt = 'Example: $subject $band $mode $evt (Ctrl+c to escape) >>'
        try:
            raw_inp = input(prompt)
            if raw_inp.strip().lower() != 'default':
                subject, band, mode, evt = raw_inp.split(' ')
        except KeyboardInterrupt:
            break
        except:
            continue
        print(subject, band, mode, evt)

        display(subject, band, mode, evt)

    print('Byebye.')
    sys.exit(0)


# %% ---- 2025-11-19 ------------------------
# Pending
