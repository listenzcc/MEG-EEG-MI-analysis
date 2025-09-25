"""
File: 6.1.plot.inverse.fsaverage.py
Author: Chuncheng Zhang
Date: 2025-09-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the inverse solution in fsaverage space.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-22 ------------------------
# Requirements and constants
from scipy import stats
from util.easy_import import *
from util.subject_fsaverage import SubjectFsaverage


# %%
mode = 'meg'  # 'meg' or 'eeg'
band = 'gamma'  # 'alpha' or 'beta' or 'gamma
data_directory = Path('./data/fsaverage')
# data_directory = Path(f'./data/tfr-stc-{band}')
subject_directories = sorted(list(data_directory.glob('S*')))
print(f'{subject_directories=}')

output_directory = Path(f'./img/source-amp-{mode}')
# output_directory = Path(f'./img/source-tfr-{band}-{mode}')
output_directory.mkdir(parents=True, exist_ok=True)
print(f'{output_directory=}')

subject = SubjectFsaverage()

# %%
tmin = -1
tmax = 4

p = subject_directories[0].joinpath(f'{mode}-evt1.stc')
stc = mne.read_source_estimate(p)
stc.subject = subject.subject
stc.crop(tmin, tmax)
print(stc.data.shape)

d_freedom = len(subject_directories) - 1
n_comp_times = stc.data.shape[1]
t_thresholds = np.array([stats.t.ppf(1 - p/n_comp_times, df=d_freedom)
                         for p in [0.05, 0.01, 0.001]])
print(f'{t_thresholds=}')

# %%

# %% ---- 2025-09-22 ------------------------
# Function and class


def read_evt(evt='1', stc=stc):
    data_stack = []
    for subject_directory in tqdm(subject_directories, 'Reading subjects'):
        p = subject_directory.joinpath(f'{mode}-evt{evt}.stc')
        _stc = mne.read_source_estimate(p)
        _stc.crop(tmin, tmax)
        data_stack.append(_stc.data)
    data = np.mean(data_stack, axis=0)

    # data shape is (n_vertices, n_times)
    stc.data = data

    # Compute tvalue
    times = stc.times
    m = np.mean(stc.data[:, times < 0], axis=1, keepdims=True)
    s = np.std(stc.data[:, times < 0], axis=1, keepdims=True)
    stc.data = (stc.data - m) / s

    # Not interested in negative values
    # stc.data[stc.data < 0] = 0

    return stc


# %%
# Screen shots
if True:
    for evt in tqdm(['1', '2', '3', '4', '5']):
        read_evt(evt=evt, stc=stc)

        # Plot with stc.plot
        # <https://mne.tools/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate.plot>
        brain_kwargs = dict(alpha=1.0, background="white",
                            cortex="low_contrast")

        clim = {
            'kind': 'value',
            'lims': t_thresholds,
        }

        for t in tqdm([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 'Saving images'):
            brain = stc.plot(
                hemi='lh',
                # views=['lateral', 'medial'],
                views='lateral',
                surface='inflated',
                initial_time=t,
                time_viewer=False,
                show_traces=False,
                # time_label=None,
                clim=clim if mode == 'meg' else 'auto',
                subjects_dir=subject.subjects_dir,
                brain_kwargs=brain_kwargs)

            brain.add_text(
                0.1, 0.9, f'evt{evt} t={t:.1f}s', 'title', font_size=16)
            brain.save_image(output_directory.joinpath(
                f'fsaverage-evt{evt}-t{t:.1f}.png'))
            brain.close()

    exit(0)


# %% ---- 2025-09-22 ------------------------
# Play ground

while True:
    inp = input("Enter evt number (1-5) (or quit): ")
    if inp.lower() in ['q', 'quit', 'exit']:
        break

    evt = inp.strip()

    try:
        read_evt(evt=evt, stc=stc)
    except Exception as e:
        print(f"Error: {e}")
        continue

    # Plot with stc.plot
    # <https://mne.tools/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate.plot>
    brain_kwargs = dict(alpha=1.0, background="white", cortex="low_contrast")

    clim = {
        'kind': 'value',
        'lims': t_thresholds,
    }

    brain = stc.plot(
        hemi='lh',
        # views=['lateral', 'medial'],
        views='lateral',
        surface='inflated',
        initial_time=0,
        time_viewer=True,
        # show_traces=False,
        # time_label=None,
        # clim=clim,
        subjects_dir=subject.subjects_dir,
        brain_kwargs=brain_kwargs)

    brain.add_text(0.1, 0.9, evt, 'title', font_size=16)

print('Bye!')
exit(0)

# %% ---- 2025-09-22 ------------------------
# Pending


# %% ---- 2025-09-22 ------------------------
# Pending
