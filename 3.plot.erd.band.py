"""
File: 3.plot.erd.band.py
Author: Chuncheng Zhang
Date: 2025-06-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot ERD in band in single and averaged subject level.
    Use all channels

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-24 ------------------------
# Requirements and constants
from util.easy_import import *

data_directory = Path('./data/TFR')
compile = re.compile(
    r'^(?P<me>[a-z]+)-(?P<mode>[a-z]+)-(?P<evt>\d+)-average-tfr.h5')

# %% ---- 2025-06-24 ------------------------
# Function and class


def find_tfr_files(pattern: str):
    found = list(data_directory.rglob(pattern))
    return found


def read_tfr(path: Path):
    tfr = mne.time_frequency.read_tfrs(path)
    return tfr


def to_df(path: Path, channels: list = []):
    '''
    Read TFR from path and pick channels.
    If channels list is empty, use all channels.
    Return df with columns: freq, time, channel, ch_type, value, name, evt
    '''
    name = path.name
    dct = compile.search(name).groupdict()
    subject_name = path.parent.name
    tfr = read_tfr(path)

    if len(channels) > 0:
        tfr.pick(channels)

    df = tfr.to_data_frame(long_format=True)
    df['name'] = subject_name
    df['evt'] = dct['evt']
    return df


def append_averaged_subject(df: pd.DataFrame):
    '''
    Append averaged subject to the df
    '''
    # Average value across name
    columns = [c for c in df.columns if c not in ['name', 'value']]
    _df = df.groupby(columns, observed=True)['value'].mean().reset_index()
    _df['name'] = 'Averaged'
    df = pd.concat([df, _df])
    return df


# %% ---- 2025-06-24 ------------------------
# Play ground

class EEG_Opt:
    vmin = -1
    vmax = 0.5
    vcenter = 0
    cmap = 'RdBu'
    pattern = 'eeg-logratio-*-average-tfr.h5'
    output_fname = 'data/img/ERD-band/eeg-{}Band.png'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)


class MEG_Opt:
    vmin = -1
    vmax = 0.5
    vcenter = 0
    cmap = 'RdBu'
    pattern = 'meg-logratio-*-average-tfr.h5'
    output_fname = 'data/img/ERD-band/meg-{}Band.png'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)


for opt in [EEG_Opt, MEG_Opt]:
    Path(opt.output_fname.format('')).parent.mkdir(parents=True, exist_ok=True)
    found = find_tfr_files(opt.pattern)
    dfs = [to_df(p) for p in tqdm(found, 'Read TFR')]
    raw_df = pd.concat(dfs)

    for band_name, band in zip(['alpha', 'beta'], [(8, 13), (15, 25)]):
        query = ' & '.join(
            ['time <= 5.0', f'freq<={band[1]}', f'freq>={band[0]}'])
        df = raw_df.copy().query(query)

        # Average across freqs
        columns = [c for c in df.columns if c not in ['freq', 'value']]
        df = df.groupby(columns, observed=True)['value'].mean().reset_index()

        df = append_averaged_subject(df)
        print(df)

        evts = sorted(df['evt'].unique())
        names = sorted(df['name'].unique())
        rows = len(names)
        cols = len(evts)
        print(names, evts)

        fig_width = 4 * cols  # inch
        fig_height = 4 * rows  # inch
        fig, axes = plt.subplots(
            rows, cols+1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [10] * cols + [1]})

        for name, evt in tqdm(itertools.product(names, evts), 'Plotting'):
            i = names.index(name)
            j = evts.index(evt)
            ax = axes[i, j]

            query = ' & '.join(
                [f'name=="{name}"', f'evt=="{evt}"'])
            _df = df.query(query)

            ax.scatter(_df['time'], _df['channel'],
                       c=_df['value'], **opt.scatter_kwargs)
            ax.set_title(f'ERD @evt: {evt}, @sub: {name}')
            ax.set_xlabel('Time (s)')
            if j == 0:
                ax.set_ylabel(f'Channel')
            else:
                ax.set_yticks([])

        for i in range(rows):
            fig.colorbar(axes[i, 0].collections[0], cax=axes[i, cols],
                         orientation='vertical').ax.set_yscale('linear')

        fig.tight_layout()
        f = opt.output_fname.format(band_name)
        fig.savefig(f)
        logger.info(f'Wrote: {f}')

# %% ---- 2025-06-24 ------------------------
# Pending


# %% ---- 2025-06-24 ------------------------
# Pending
