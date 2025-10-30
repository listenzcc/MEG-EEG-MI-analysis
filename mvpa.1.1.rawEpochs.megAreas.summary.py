"""
File: mvpa.1.1.rawEpochs.megAreas.summary.py
Author: Chuncheng Zhang
Date: 2025-09-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the megAreas MVPA results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-10 ------------------------
# Requirements and constants
import matplotlib as mpl
from collections import defaultdict
from util.easy_import import *
from util.io.file import load
from util.read_example_raw import md

plt.style.use('ggplot')

meg_ch_name_dct = json.load(open('./data/meg_ch_name_dct.json'))

data_directory = Path('./data/MVPA.megAreas.withCoef')

# %%
epochs_kwargs = {'tmin': -1, 'tmax': 4-1e-3, 'decim': 6*5}
md.generate_epochs(**epochs_kwargs)

md.meg_epochs.load_data()
# md.meg_epochs.pick('mag')

print(md.eeg_epochs)
print(md.meg_epochs)
print(md.meg_epochs.times)

# %% ---- 2025-09-10 ------------------------
# Function and class


# %% ---- 2025-09-10 ------------------------
# Play ground
print(meg_ch_name_dct.keys())

#
mode = 'meg'
evts = [1, 2, 3, 4, 5]

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=16, va='bottom')
    return


for ch_mark in sorted(list(meg_ch_name_dct.keys())) + ['ALL']:
    files = list(data_directory.rglob(f'decoding-meg-chmark-{ch_mark}.dump'))
    objs = [load(file) for file in files]
    times = objs[0]['times']
    scores_mean = np.mean([obj['scores'] for obj in objs], axis=0)
    scores_std = np.std([obj['scores'] for obj in objs], axis=0)

    if len(ch_mark) == 1:
        ax = axes[0]
    elif ch_mark[0] == 'L':
        ax = axes[1]
    elif ch_mark[0] == 'R':
        ax = axes[2]
    elif ch_mark == 'ALL':
        ax = axes[0]
    else:
        raise ValueError()

    ax.plot(times, np.diag(scores_mean), label=f"score({ch_mark})")

for i, ax in enumerate(axes):
    add_top_left_notion(ax, 'abc'[i])

    ax.axhline(1/len(evts), color="gray", linestyle="--")  # , label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AccScore")
    ax.legend(loc='lower right')  # , bbox_to_anchor=(1, 1))
    ax.axvline(0.0, color="gray", linestyle="-")
    ax.set_title(f"Acc over time ({['Both', 'Left', 'Right'][i]} hemi)")
    if i == 0:
        ax.set_ylim([0.15, 0.4])
    else:
        ax.set_ylim([0.15, 0.35])

fig.tight_layout()
fig.savefig(data_directory.joinpath('mvpa_megAreas_summary.png'))

plt.show()

# %%

mpl.use('pdf')
for ch_mark in sorted(list(meg_ch_name_dct.keys())) + ['ALL']:
    with PdfPages(data_directory.joinpath(f'mvpa_megAreas_topomap_{ch_mark}.pdf')) as pdf:
        ch_mark = ch_mark.upper()

        files = list(data_directory.rglob(
            f'decoding-meg-chmark-{ch_mark}.dump'))
        objs = [load(file) for file in files]

        values = defaultdict(list)
        for obj in tqdm(objs):
            # shape is (n_ch, n_events, n_times)
            coef = obj['coef']
            # print(obj['coef'].shape)
            for ch_name, value in zip(obj['ch_names'], coef):
                values[ch_name].append(np.abs(value))

        for k, v in values.items():
            values[k] = np.mean(v, axis=0)[np.newaxis, :, :]

        for k, v in values.items():
            # v.shape is (1, n_events, n_times)
            print(k, v.shape)

        # Make data as shape (n_epochs, n_channels, n_times)
        data = np.concatenate(list(values.values()), axis=0).transpose(1, 0, 2)
        print(f'{data.shape=}')
        events = [[10000*(i+1), 0, i+1] for i in range(data.shape[0])]
        print(events)
        epochs = md.meg_epochs.copy()
        epochs.pick(list(values.keys()))

        ea = mne.EpochsArray(data, epochs.info,
                             tmin=epochs.tmin,
                             events=events,
                             event_id=epochs.event_id)
        ea.crop(tmin=0, tmax=4)
        print(ea)

        # Plot
        _times = [0, 0.5, 1, 2, 3]
        topomap_args = {
            'extrapolate': 'local'  # 'box', 'local', 'head'
        }
        for evt in ['1', '2', '3', '4', '5']:
            evoked = ea[evt].average()
            fig = evoked.plot_joint(
                title=f'MEG @evt: {evt}', topomap_args=topomap_args)
            pdf.savefig(fig)


# %% ---- 2025-09-10 ------------------------
# Pending

# %%
