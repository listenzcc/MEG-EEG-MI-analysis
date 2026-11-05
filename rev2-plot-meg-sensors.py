"""
File: rev2-plot-meg-sensors.py
Author: Chuncheng Zhang
Date: 2026-11-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read meg info and plot the sensors.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-11-04 ------------------------
# Requirements and constants

from matplotlib.lines import Line2D
from mne.channels.layout import _find_topomap_coords
from cycler import cycler
from matplotlib import patches
from matplotlib import pyplot as plt
import warnings
import json
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'meg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# %% ---- 2026-11-04 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    # Raw freq is 1200 Hz
    # epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 12}

    use_latest_ds_directories = 1  # 8

    # Read from file
    found = find_ds_directories(subject_directory)
    mds = [read_ds_directory(p) for p in found[-use_latest_ds_directories:]]

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)

        if mode in ['eeg', 'all']:
            md.eeg_epochs.load_data()
            # md.eeg_epochs.filter(**filter_kwargs)
            # md.eeg_epochs.crop(tmin=-1, tmax=4)
            # md.eeg_epochs.apply_baseline((-1, 0))

        if mode in ['meg', 'all']:
            md.meg_epochs.load_data()
            # md.meg_epochs.filter(**filter_kwargs)
            # md.meg_epochs.crop(tmin=-1, tmax=4)
            # md.meg_epochs.apply_baseline((-1, 0))

    return mds, event_id


# %% ---- 2026-11-04 ------------------------
# Play ground
subject_name = subject_directory.name

mds, event_id = read_data()


# %% ---- 2026-11-04 ------------------------
# Pending
raw = mds[0].raw
raw.pick_types(meg=True, eeg=False, stim=False, eog=False, exclude='bads')
info = raw.info
info

ch_map = json.load(open('./data/meg_ch_name_dct.json'))
print(info['ch_names'], ch_map)

# %%
mne.viz.plot_sensors(info)

# %% ---- 2026-11-04 ------------------------
# Pending

# %% ---- MEG sensor classification and visualization
# Figure S1: MEG sensor coverage across five cortical regions


# --------------------------------------------------
# 1. Load MEG information
# --------------------------------------------------

info = raw.info
ch_names = info['ch_names']

# Load channel-to-region dictionary
with open('./data/meg_ch_name_dct.json', 'r') as f:
    ch_map = json.load(f)


# --------------------------------------------------
# 2. Define five cortical regions
# --------------------------------------------------

region_names = {
    'F': 'Frontal',
    'C': 'Central',
    'P': 'Parietal',
    'T': 'Temporal',
    'O': 'Occipital',
}

# Distinct colors for the five regions
region_colors = {
    'F': '#E76F51',  # Coral
    'C': '#2A9D8F',  # Teal
    'O': '#257B9D',  # Blue
    'T': '#9B72CF',  # Violet
    'P': '#E9A23B',  # Orange
}

region_order = ['F', 'C', 'P', 'T', 'O']


# --------------------------------------------------
# 3. Match actual MEG channels to region definitions
# --------------------------------------------------

def normalize_channel_name(name):
    """
    Remove the CTF dataset suffix.

    Example:
        MLC11-4504 -> MLC11
    """
    return name.split('-')[0]


# Normalize channel names in the region dictionary
region_channel_sets = {
    region: {
        normalize_channel_name(name)
        for name in ch_map[region]
    }
    for region in region_order
}

# Build channel indices for each region
region_indices = {}

for region in region_order:

    indices = []

    for idx, name in enumerate(ch_names):

        normalized_name = normalize_channel_name(name)

        if normalized_name in region_channel_sets[region]:
            indices.append(idx)

    region_indices[region] = np.asarray(
        indices, dtype=int
    )


# --------------------------------------------------
# 4. Check channel assignment
# --------------------------------------------------

assigned_channels = []

print('\n' + '=' * 70)
print('MEG SENSOR DISTRIBUTION')
print('=' * 70)

for region in region_order:

    indices = region_indices[region]

    names = [
        normalize_channel_name(ch_names[i])
        for i in indices
    ]

    # Left, right and bilateral
    left = [n for n in names if n.startswith('ML')]
    right = [n for n in names if n.startswith('MR')]
    midline = [n for n in names if n.startswith('MZ')]

    print(
        f"{region} ({region_names[region]:10s}): "
        f"Total = {len(names):3d}, "
        f"Left = {len(left):3d}, "
        f"Right = {len(right):3d}, "
        f"Midline = {len(midline):3d}"
    )

    assigned_channels.extend(indices.tolist())


# Check duplicate assignments
if len(assigned_channels) != len(set(assigned_channels)):
    raise ValueError(
        'Some MEG channels are assigned to multiple regions.'
    )

# Check missing channels
unassigned_indices = sorted(
    set(range(len(ch_names))) - set(assigned_channels)
)

if unassigned_indices:

    unassigned_names = [
        ch_names[i] for i in unassigned_indices
    ]

    print('\nWARNING: Unassigned MEG channels:')
    print(unassigned_names)

else:
    print('\nAll MEG channels have been assigned.')


print(f'\nTotal MEG channels: {len(ch_names)}')
print(f'Assigned channels: {len(assigned_channels)}')
print('=' * 70)

# --------------------------------------------------
# 5. Obtain MNE 2D topomap sensor coordinates
# --------------------------------------------------


# Select MEG magnetometers
picks = mne.pick_types(
    info,
    meg='mag',
    ref_meg=False,
    exclude='bads'
)

# MNE native 2D topomap coordinates
pos = _find_topomap_coords(info, picks)

# Map original channel indices to 2D coordinates
channel_positions = {
    ch_idx: pos[i]
    for i, ch_idx in enumerate(picks)
}


# --------------------------------------------------
# 6. Create MNE native 2D head topomap
# --------------------------------------------------

fig = mne.viz.plot_sensors(
    info,
    kind='topomap',
    ch_type='mag',
    show_names=False,
    pointsize=0,       # Small background points
    linewidth=0.0,
    title='MEG sensor distribution',
    show=False,
)

ax = fig.axes[0]

# Overlay sensors with explicitly assigned region colors
for region in region_order:

    indices = [
        idx for idx in region_indices[region]
        if idx in channel_positions
    ]

    if len(indices) == 0:
        print(f'Warning: no sensors found for {region}')
        continue

    xy = np.array([
        channel_positions[idx]
        for idx in indices
    ])

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=35,
        color=region_colors[region],
        edgecolors='black',
        linewidths=0.4,
        zorder=10,
    )


# --------------------------------------------------
# 7. Create an explicitly matched legend
# --------------------------------------------------

legend_handles = []

for region in region_order:

    n_channels = sum(
        idx in channel_positions
        for idx in region_indices[region]
    )

    legend_handles.append(
        Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markerfacecolor=region_colors[region],
            markeredgecolor='black',
            markeredgewidth=0.4,
            markersize=7,
            label=(
                f'{region_names[region]} '
                f'(n = {n_channels})'
            ),
        )
    )

ax.legend(
    handles=legend_handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.02),
    frameon=False,
    fontsize=9,
    handletextpad=0.6,
    labelspacing=1.0,
)

# --------------------------------------------------
# 8. Figure formatting and export
# --------------------------------------------------

fig.set_size_inches(8.0, 6.5)

fig.tight_layout()

output_dir = Path('./data')
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / 'meg-sensor-distribution.png',
    dpi=600,
    bbox_inches='tight',
    facecolor='white',
)

fig.savefig(
    output_dir / 'meg-sensor-distribution.pdf',
    bbox_inches='tight',
    facecolor='white',
)

fig.savefig(
    output_dir / 'meg-sensor-distribution.svg',
    bbox_inches='tight',
    facecolor='white',
)

plt.show()


# %%
