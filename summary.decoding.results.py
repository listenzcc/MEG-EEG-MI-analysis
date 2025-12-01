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
import joblib
import seaborn as sns
from sklearn import metrics
from util.easy_import import *

# %%
OUTPUT_DIR = Path('./data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def combine_eeg_meg_FBCSP_results():
    data_directories = {
        'eeg': Path('./data/MVPA.FBCSP.vote.eeg.fine'),
        'meg': Path('./data/MVPA.FBCSP.vote.meg.fine'),
    }

    def vote(preds):
        candidates = {k: 0 for k in [1, 2, 3, 4, 5]}
        for i, e in enumerate(preds):
            candidates[e] += 1
        return sorted(candidates.items(), key=lambda e: e[1])

    df2s = []
    for eeg_fname, meg_fname in zip(
        sorted(list(data_directories['eeg'].rglob('*.dump'))),
        sorted(list(data_directories['meg'].rglob('*.dump'))),
    ):
        print(eeg_fname, meg_fname)

        data = []
        yy_true = []
        yy_pred = []
        yy_pred_2 = []
        y_probas_stack = []
        y_true_stack = []
        d = joblib.load(eeg_fname)
        d_m = joblib.load(meg_fname)

        # d['freqs'] = d['freqs'][:-1]
        # d.pop(6)

        subject = d['subject_name']

        # y_true shape: samples
        y_true = d['y_true']
        y_true_stack.append(y_true)

        # y_preds shape: bands x samples
        y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]

        # y_probas shape: bands x samples x classes
        y_probas = [v['y_proba']
                    for k, v in d.items() if isinstance(k, int)]

        y_probas_m = [v['y_proba']
                      for k, v in d_m.items() if isinstance(k, int)]
        for y1, y2 in zip(y_probas, y_probas_m):
            y1 *= y2

        y_probas_stack.append(y_probas)

        for i, y_pred in enumerate(y_preds):
            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
            data.append({'subject': subject, 'acc': acc, 'freqIdx': i})

        # Hard vote
        y_pred = [vote(e)[-1][0] for e in np.array(y_preds).T]

        # Soft vote
        y_pred_2 = np.argmax(
            np.prod(np.array(y_probas), axis=0), axis=1) + 1

        conf_mat = metrics.confusion_matrix(
            y_true=y_true, y_pred=y_pred_2, normalize='true')

        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        acc_2 = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_2)
        print(acc, acc_2)
        data.append({'subject': subject, 'acc': acc,
                    'acc2': acc_2, 'freqIdx': 'vote',
                     'confusionMatrix': conf_mat})

        yy_true.extend(y_true)
        yy_pred.extend(y_pred)
        yy_pred_2.extend(y_pred_2)

        data = pd.DataFrame(data)

        df1 = data.query('freqIdx != "vote"')
        freqs = [np.mean(f) for f in d['freqs']]
        df1['freq'] = df1['freqIdx'].map(lambda i: freqs[i])

        df2 = data.query('freqIdx == "vote"')
        df2['acc'] = df2['acc2']

        df2s.append(df2)

    # Results of filter bands
    df2 = pd.concat(df2s)
    df2['accuracy'] = df2['acc']
    df2['method'] = 'FBCSP'
    df2['mode'] = 'COMBINE'
    df21 = df2[['mode', 'subject', 'accuracy', 'method']]
    df22 = df2[['mode', 'subject', 'confusionMatrix', 'method']]

    return df21, df22


def load_FBCSP_df():
    data_directories = [
        Path('./data/MVPA.FBCSP.vote.eeg.fine'),
        Path('./data/MVPA.FBCSP.vote.meg.fine'),
        # Path('./data/MVPA.FBCSP.all.vote'),
    ]

    df1s = []
    df2s = []

    for data_directory in data_directories:
        name = data_directory.name
        mode = name.split('.')[-2].upper()
        print(f'Working with {name=}, {mode=}')

        data_files = list(data_directory.rglob('*.dump'))
        data_files.sort()
        print(data_files)

        def vote(preds):
            candidates = {k: 0 for k in [1, 2, 3, 4, 5]}
            for i, e in enumerate(preds):
                candidates[e] += 1
            return sorted(candidates.items(), key=lambda e: e[1])

        data = []
        yy_true = []
        yy_pred = []
        yy_pred_2 = []
        y_probas_stack = []
        y_true_stack = []
        for f in data_files:
            d = joblib.load(f)

            # d['freqs'] = d['freqs'][:-1]
            # d.pop(6)

            subject = d['subject_name']

            # y_true shape: samples
            y_true = d['y_true']
            y_true_stack.append(y_true)

            # y_preds shape: bands x samples
            y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]

            # y_probas shape: bands x samples x classes
            y_probas = [v['y_proba']
                        for k, v in d.items() if isinstance(k, int)]
            y_probas_stack.append(y_probas)

            for i, y_pred in enumerate(y_preds):
                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                data.append({'subject': subject, 'acc': acc, 'freqIdx': i})

            # Hard vote
            y_pred = [vote(e)[-1][0] for e in np.array(y_preds).T]

            # Soft vote
            y_pred_2 = np.argmax(
                np.prod(np.array(y_probas), axis=0), axis=1) + 1

            conf_mat = metrics.confusion_matrix(
                y_true=y_true, y_pred=y_pred_2, normalize='true')

            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
            acc_2 = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_2)
            print(acc, acc_2)
            data.append({'subject': subject, 'acc': acc,
                        'acc2': acc_2, 'freqIdx': 'vote',
                         'confusionMatrix': conf_mat})

            yy_true.extend(y_true)
            yy_pred.extend(y_pred)
            yy_pred_2.extend(y_pred_2)

        data = pd.DataFrame(data)
        data['mode'] = mode

        df1 = data.query('freqIdx != "vote"')
        freqs = [np.mean(f) for f in d['freqs']]
        df1['freq'] = df1['freqIdx'].map(lambda i: freqs[i])

        df2 = data.query('freqIdx == "vote"')
        df2['acc'] = df2['acc2']

        df1s.append(df1)
        df2s.append(df2)

    # Results of single filter band
    df1 = pd.concat(df1s)
    df1['accuracy'] = df1['acc']
    df1['method'] = 'CSP'
    df1 = df1[['mode', 'subject', 'accuracy', 'method', 'freq']]

    # Results of filter bands
    df2 = pd.concat(df2s)
    df2['accuracy'] = df2['acc']
    df2['method'] = 'FBCSP'
    df21 = df2[['mode', 'subject', 'accuracy', 'method']]
    df22 = df2[['mode', 'subject', 'confusionMatrix', 'method']]

    return df1, df21, df22


# %%
df21, df22 = combine_eeg_meg_FBCSP_results()
display(df21)
display(df22)
print(df21['accuracy'].mean())

# %%
dfs, dfm_a, dfm_c = load_FBCSP_df()

dfm_a = pd.concat([dfm_a, df21])
dfm_c = pd.concat([dfm_c, df22])

dfs.to_csv(OUTPUT_DIR / 'decoding-on-freq.csv')
dfm_a.to_csv(OUTPUT_DIR / 'decoding-fbcsp.csv')
dfm_c.to_csv(OUTPUT_DIR / 'decoding-fbcsp-confusion-matrix.csv')
display(dfs)
display(dfm_a)
display(dfm_c)

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
dfc = pd.concat([df1, df2])
dfc = dfc[['subject', 'mode', 'method', 't', 'accuracy']]
dfc['mode'] = dfc['mode'].map(lambda e: e.lower())
dfc.to_csv(OUTPUT_DIR / 'decoding-on-time.csv')
display(dfc)

# %%
sns.set_theme(context='paper', style='ticks', font_scale=1)
sns.lineplot(dfc, x='t', y='accuracy', hue='mode', style='method')
plt.show()

# %% ---- 2025-11-25 ------------------------
# Pending


# %% ---- 2025-11-25 ------------------------
# Pending
