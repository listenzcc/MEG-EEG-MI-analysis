"""
File: 2.plot.erd.exampleChannels.py
Author: Chuncheng Zhang
Date: 2025-06-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot ERD in single and averaged subject level.
    Use the example channels.

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
class BasicOpt:
    vmin = -0.5
    vmax = 0
    vcenter = -0.25
    cmap = 'RdBu'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)


class EEG_Opt(BasicOpt):
    channels = ['C3', 'Cz', 'C4']
    pattern = 'eeg-logratio-*-average-tfr.h5'
    output_fname = 'data/img/ERD-example-channels/eeg-evt-{}.png'


class MEG_Opt(BasicOpt):
    channels = ['MLC42', 'MZC03', 'MRC42']
    pattern = 'meg-logratio-*-average-tfr.h5'
    output_fname = 'data/img/ERD-example-channels/meg-evt-{}.png'


for opt in [EEG_Opt, MEG_Opt]:
    Path(opt.output_fname.format(0)).parent.mkdir(parents=True, exist_ok=True)
    found = find_tfr_files(opt.pattern)
    dfs = [to_df(p, opt.channels) for p in tqdm(found, 'Read TFR')]
    df = pd.concat(dfs).query('time <= 5.0')
    df = append_averaged_subject(df)
    print(df)

    evts = sorted(df['evt'].unique())
    names = sorted(df['name'].unique())
    rows = len(names)
    cols = len(opt.channels)
    print(names, evts)

    for evt in evts:
        fig_width = 4 * cols  # inch
        fig_height = 4 * rows  # inch
        fig, axes = plt.subplots(
            rows, cols+1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [10] * cols + [1]})

        for name, chn in tqdm(itertools.product(names, opt.channels), 'Plotting'):
            i = names.index(name)
            j = opt.channels.index(chn)
            ax = axes[i, j]

            query = ' & '.join(
                [f'name=="{name}"', f'evt=="{evt}"', f'channel=="{chn}"'])
            _df = df.query(query)

            ax.scatter(_df['time'], _df['freq'],
                       c=_df['value'], **opt.scatter_kwargs)
            ax.set_title(f'ERD @chn: {chn}, @sub: {name}')
            ax.set_xlabel('Time (s)')
            if j == 0:
                ax.set_ylabel(f'Freq (Hz)')
            else:
                ax.set_yticks([])

        for i in range(rows):
            fig.colorbar(axes[i, 0].collections[0], cax=axes[i, cols],
                         orientation='vertical').ax.set_yscale('linear')

        fig.tight_layout()
        f = opt.output_fname.format(evt)
        fig.savefig(f)
        logger.info(f'Wrote: {f}')

# %% ---- 2025-06-24 ------------------------
# Pending


# %% ---- 2025-06-24 ------------------------
# Pending
