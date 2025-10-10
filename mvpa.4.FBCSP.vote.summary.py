# %%
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from util.easy_import import *
from util.io.file import load

plt.style.use('ggplot')

# %%
# Summary the CSP

output_directory = Path('./data/MVPA.FBCSP.vote.results')
output_directory.mkdir(parents=True, exist_ok=True)

data_directory = Path('./data/MVPA.FBCSP.vote.eeg.fine')

data_directories = [
    Path('./data/MVPA.FBCSP.vote.eeg.fine'),
    Path('./data/MVPA.FBCSP.vote.eeg'),
    Path('./data/MVPA.FBCSP.vote.meg.fine'),
    Path('./data/MVPA.FBCSP.vote.meg')
]


for data_directory in data_directories:
    name = data_directory.name
    print(f'Working with {name=}')

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
        d = load(f)

        # d['freqs'] = d['freqs'][:-1]
        # d.pop(6)

        subject = d['subject_name']

        # y_true shape: samples
        y_true = d['y_true']
        y_true_stack.append(y_true)

        # y_preds shape: bands x samples
        y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]

        # y_probas shape: bands x samples x classes
        y_probas = [v['y_proba'] for k, v in d.items() if isinstance(k, int)]
        y_probas_stack.append(y_probas)

        for i, y_pred in enumerate(y_preds):
            acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            data.append({'subject': subject, 'acc': acc, 'freqIdx': i})

        # Hard vote
        y_pred = [vote(e)[-1][0] for e in np.array(y_preds).T]
        # Soft vote
        y_pred_2 = np.argmax(np.prod(np.array(y_probas), axis=0), axis=1) + 1

        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        acc_2 = accuracy_score(y_true=y_true, y_pred=y_pred_2)
        print(acc, acc_2)
        data.append({'subject': subject, 'acc': acc,
                    'acc2': acc_2, 'freqIdx': 'vote'})
        yy_true.extend(y_true)
        yy_pred.extend(y_pred)
        yy_pred_2.extend(y_pred_2)

    data = pd.DataFrame(data)
    print(data)

    group = data.groupby(by='freqIdx')
    print(group['acc'].mean())
    print(group['acc2'].mean())

    yy_true = np.vstack(yy_true).ravel()
    yy_pred = np.vstack(yy_pred).ravel()
    yy_pred_2 = np.vstack(yy_pred_2).ravel()

    print(metrics.classification_report(y_true=yy_true, y_pred=yy_pred))
    print(metrics.classification_report(y_true=yy_true, y_pred=yy_pred_2))
    confusion_matrix = metrics.confusion_matrix(
        y_true=yy_true, y_pred=yy_pred_2, normalize='true')
    print(f'{confusion_matrix=}')

    # --------------------------------------------------------------------------------
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    acc = data.query('freqIdx=="vote"')['acc2'].mean()
    x = [np.mean(f) for f in d['freqs']]
    ax = axes[0]
    ax.plot(x, group['acc'].mean().to_list()[:-1])
    ax.hlines(y=0.2, xmin=np.min(x), xmax=np.max(x),
              colors='gray', linestyles='-.')
    ax.hlines(y=acc, xmin=np.min(x), xmax=np.max(x),
              colors='blue')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('Acc')
    ax.set_title(f'{name} | vote {acc=:0.3f}')
    ax.set_ylim([0.15, 0.5])

    import seaborn as sns
    ax = axes[1]
    ticklabels = ['1', '2', '3', '4', '5']
    sns.heatmap(confusion_matrix, vmin=0.1, vmax=0.6, cbar=False, annot=True,
                xticklabels=ticklabels, yticklabels=ticklabels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    ax.set_title('Confusion matrix')

    fig.tight_layout()
    plt.savefig(output_directory.joinpath(f'{name}.png'))
    plt.show()

# %%
