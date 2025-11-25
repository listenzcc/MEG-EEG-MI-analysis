"""
File: summary.decoding.results.py
Author: Chuncheng Zhang
Date: 2025-11-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the decoding results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-25 ------------------------
# Requirements and constants
import seaborn as sns
from util.easy_import import *


# %% ---- 2025-11-25 ------------------------
# Function and class
def load_accumulating_df():
    dump_files = list(Path('./data/MVPA.accumulate').rglob('*.joblib'))
    dfs = [joblib.load(f) for f in dump_files]
    for df, file in zip(dfs, dump_files):
        mode = file.stem.split('.')[-1]
        df['mode'] = mode
    df = pd.concat(dfs)
    df['t'] = df['tmax']
    return df


def load_sliding_df():
    dump_files = list(Path('./data/MVPA').rglob('decoding-*-band-all.dump'))
    # compile = re.compile(r'^decoding-(?P<mode>[a-z]+)-band-all.dump')
    data = []
    for p in dump_files:
        # dct = compile.search(p.name).groupdict()
        # mode = dct['mode']
        d = joblib.load(p)
        times = d['times']
        mode = d['mode']
        subject = d['subject_name']
        for t, s in zip(times, np.diag(d['scores'])):
            data.append((t, s, mode, subject))
    df = pd.DataFrame(data, columns=['t', 'accuracy', 'mode', 'subject'])
    return df


def load_accumulating_voting_df():
    data_directories = [
        Path('./data/MVPA.FBCSP.vote-accumulate.eeg'),
        Path('./data/MVPA.FBCSP.vote-accumulate.meg'),
    ]

    tmax_array = [0.1, 0.2, 0.3, 0.4, 0.5,
                  0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    data = []

    for data_directory in data_directories:
        name = data_directory.name
        mode = name.split('.')[-1]
        print(f'Working with {name=}, {mode=}')

        data_files = sorted(list(data_directory.rglob('*.dump')))
        print(data_files)
        for f in tqdm(data_files, 'Reading files'):
            subject = f.parent.name
            obj = joblib.load(f)
            y_true = obj['y_true']
            _res = pd.DataFrame(obj['results'])
            for tmax in tqdm(tmax_array, 'Accumulating'):
                _selected = _res.query(f'tmax=={tmax}')
                joint_proba = np.prod(_selected['y_proba'])
                y_pred = np.argmax(joint_proba, axis=1) + 1
                acc = np.mean(y_true == y_pred)
                data.append({
                    'accuracy': acc,
                    't': tmax,
                    'mode': mode,
                    'subject': subject,
                })

    df = pd.DataFrame(data)
    return df


# %% ---- 2025-11-25 ------------------------
# Play ground
df1 = load_accumulating_df()
df1['method'] = 'accumulating'
display(df1)

df2 = load_sliding_df()
df2['method'] = 'sliding'
display(df2)

df3 = load_accumulating_voting_df()
df3['method'] = 'voting'
display(df3)

# %%

# %%

# %%
dfc = pd.concat([df1, df2, df3])
dfc['mode'] = dfc['mode'].map(lambda e: e.lower())
display(dfc)
sns.set_theme(context='paper', style='ticks', font_scale=1)
sns.lineplot(dfc, x='t', y='accuracy', hue='mode', style='method')
plt.show()


# %% ---- 2025-11-25 ------------------------
# Pending


# %% ---- 2025-11-25 ------------------------
# Pending
