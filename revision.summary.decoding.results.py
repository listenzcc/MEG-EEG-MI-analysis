# %%
import joblib
import seaborn as sns
from sklearn import metrics
from util.easy_import import *

# %%


def combine_eeg_meg_FBCSP_results(data_directories):
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


def load_FBCSP_df(data_directories):

    df1s = []
    df2s = []

    for data_directory in data_directories.values():
        name = data_directory.name
        # name.split('.')[-2].upper()
        mode = 'EEG' if '.eeg.' in name else 'MEG'
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

# %%

array = []

for tasks in tqdm(['all', '1234', '12', '13', '14'], 'Tasks'):
    if tasks == 'all':
        data_directories = {
            'eeg': Path('./data/MVPA.FBCSP.vote.eeg.fine'),
            'meg': Path('./data/MVPA.FBCSP.vote.meg.fine'),
        }
    else:
        data_directories = {
            'eeg': Path(f'./data/MVPA.FBCSP.vote.eeg.fine.{tasks}'),
            'meg': Path(f'./data/MVPA.FBCSP.vote.meg.fine.{tasks}'),
        }

    try:
        df21, df22 = combine_eeg_meg_FBCSP_results(data_directories)
        dfs, dfm_a, dfm_c = load_FBCSP_df(data_directories)

        dfm_a = pd.concat([dfm_a, df21])
        dfm_c = pd.concat([dfm_c, df22])
    except:
        continue

    df = dfm_a
    df['tasks'] = tasks
    array.append(df)

table = pd.concat(array)
table

# %%
sns.barplot(table, x='mode', y='accuracy', hue='tasks')
plt.show()

# %%

# %%
