"""
File: 6.compute.mrcp.py
Author: Chuncheng Zhang
Date: 2025-07-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-03 ------------------------
# Requirements and constants
from util.easy_import import *

evoked_directory = Path('./data/evoked')
data_directory = Path('./data/MRCP')
data_directory.mkdir(exist_ok=True, parents=True)
compile = re.compile(
    r'^(?P<mode>[a-z]+)-evt(?P<evt>\d+)-n(?P<nave>\d+)-ave.fif')

# %%
pattern = '*-ave.fif'
files = list(evoked_directory.rglob(pattern))
table = []
for p in files:
    dct = compile.search(p.name).groupdict()
    try:
        evoked = mne.read_evokeds(p)[0]
        dct.update({'path': p, 'evoked': evoked})
    except Exception:
        dct.update({'path': p, 'evoked': None})
        pass
    finally:
        table.append(dct)
table = pd.DataFrame(table)
print(table)

# %% ---- 2025-07-03 ------------------------
# Function and class


def get_merged_evoked(mode: str, evt: str, baseline: tuple = None):
    selected = table.query(f'mode=="{mode}" & evt=="{evt}"')

    pick_channels = dict(
        meg=['MLC42', 'MZC03', 'MRC42'],
        eeg=['C3', 'Cz', 'C4']
    )

    if mode == 'meg':
        evokeds = [e for e in selected['evoked']]
        chs = set(evokeds[0].ch_names)
        for e in evokeds:
            chs = chs.intersection(set(e.ch_names))
        chs = list(chs)
        evokeds = [e.pick(chs).pick('mag') for e in selected['evoked']]
    else:
        evokeds = [e for e in selected['evoked']]

    [e.pick(pick_channels[mode]) for e in evokeds]

    [e.filter(l_freq=0.1, h_freq=40, n_jobs=32) for e in evokeds]

    if baseline:
        [e.apply_baseline(baseline) for e in evokeds]

    evoked = evokeds[0].copy()

    evoked.data = np.mean([e.data for e in evokeds], axis=0)
    return evoked


# %% ---- 2025-07-03 ------------------------
# Play ground
evts = sorted(list(table['evt'].unique()))
modes = ['meg', 'eeg']

mpl.use('pdf')
p = data_directory.joinpath('mrcp.pdf')
with PdfPages(p) as pdf:
    for mode, evt in itertools.product(modes, evts):
        with redirect_stdout(io.StringIO()):
            with redirect_stderr(io.StringIO()):
                evoked = get_merged_evoked(mode, evt)
        fig = evoked.plot_joint(
            title=f'{mode} @evt{evt}', times=[0, 0.15, 0.2])
        pdf.savefig(fig)

        with redirect_stdout(io.StringIO()):
            with redirect_stderr(io.StringIO()):
                evoked = get_merged_evoked(mode, evt, (-0.2, 0))
                evoked.crop(tmin=-0.3, tmax=0.7)
        fig = evoked.plot_joint(
            title=f'{mode} @evt{evt}', times=[0, 0.15, 0.2])
        pdf.savefig(fig)

logger.info(f'Saved to {p}')

# %% ---- 2025-07-03 ------------------------
# Pending

# %% ---- 2025-07-03 ------------------------
# Pending

# %%
