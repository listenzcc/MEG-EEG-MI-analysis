"""
File: 1.1.plot.tfr.py
Author: Chuncheng Zhang
Date: 2025-09-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the tfr.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-15 ------------------------
# Requirements and constants
import seaborn as sns
from util.easy_import import *

re_pattern = re.compile(
    r'^(?P<mode>[a-z]+)-(?P<method>[a-z]+)-(?P<evt>[0-9])-average-tfr.h5')

data_directory = Path('./data/TFR')

# %% ---- 2025-09-15 ------------------------
# Function and class


def find_h5_files():
    h5_files = list(data_directory.rglob('*tfr.h5'))
    details = [{'name': p.name, 'full': p} for p in h5_files]
    [e.update(re_pattern.search(e['name']).groupdict()) for e in details]
    h5_files = pd.DataFrame(details)
    return h5_files


def mark_channel(channel):
    hemi = channel[1]
    area = channel[2]
    return hemi, area


def mark_time(time):
    if time < 1:
        return 'a'
    if time < 2:
        return 'b'
    if time < 3:
        return 'c'
    return 'd'


# %% ---- 2025-09-15 ------------------------
# Play ground
# Read h5 files
kw_query = {
    'mode': 'meg',
    'method': 'logratio'
}
query = ' & '.join([f'{key}=="{value}"' for key, value in kw_query.items()])
h5_files = find_h5_files().query(query)
h5_files

# %%
'''
Load the data into dataFrame.
It takes long time and costs memory, so try to run it once.
'''
dfs = []
for path, evt in tqdm(zip(h5_files['full'], h5_files['evt']), 'Reading h5 files', total=len(h5_files)):
    subject = path.parent.name
    tfr = mne.time_frequency.read_tfrs(path)
    tfr.crop(tmin=0, tmax=4)
    tfr_df = tfr.to_data_frame(long_format=True)
    tfr_df['hemi'] = tfr_df['channel'].map(lambda c: mark_channel(c)[0])
    tfr_df['area'] = tfr_df['channel'].map(lambda c: mark_channel(c)[1])
    tfr_df['subject'] = subject
    tfr_df['evt'] = evt
    dfs.append(tfr_df)
tfr_df = pd.concat(dfs)
tfr_df['phase'] = tfr_df['time'].map(mark_time)
tfr_df

# %% ---- 2025-09-15 ------------------------
# Pending
group = tfr_df.groupby(['area', 'freq', 'evt', 'phase'], as_index=False)
averaged = group.mean('value')
averaged

# %%
plt.style.use('ggplot')

evts = averaged['evt'].unique()
phases = averaged['phase'].unique()

fig, axes = plt.subplots(len(evts), len(phases),
                         figsize=(6*len(phases), 6*len(evts)))
for i, evt in enumerate(evts):
    for j, phase in enumerate(phases):
        ax = axes[i, j]
        sns.lineplot(averaged.query(f'evt=="{evt}" & phase=="{phase}"'),
                     x='freq', y='value', hue='area', ax=ax)
        ax.set_title(f'{evt=}, {phase=}')
        ax.set_ylim((-0.4, -0.1))
        ax.legend(loc='lower right')

fig.savefig(data_directory.joinpath('erd-megAreas.png'))
plt.show()


# %% ---- 2025-09-15 ------------------------
# Pending
# %%

# %%
