# %%
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from util.easy_import import *
from util.io.file import load

plt.style.use('ggplot')

# %%
# Summary the CSP

OUTPUT_DIR = Path('./data/MVPA.FBCSP.vote.results.better')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_directories = [
    Path('./data/MVPA.FBCSP.vote.eeg.fine'),
    # Path('./data/MVPA.FBCSP.vote.eeg'),
    Path('./data/MVPA.FBCSP.vote.meg.fine'),
    # Path('./data/MVPA.FBCSP.vote.meg')
]

df1s = []
df2s = []
confusion_matrixes = {}

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
    data['mode'] = mode
    print(data)

    df1 = data.query('freqIdx != "vote"')
    freqs = [np.mean(f) for f in d['freqs']]
    df1['freq'] = df1['freqIdx'].map(lambda i: freqs[i])

    df2 = data.query('freqIdx == "vote"')
    df2['acc'] = df2['acc2']

    df1s.append(df1)
    df2s.append(df2)

    conf_mat = metrics.confusion_matrix(
        y_true=yy_true, y_pred=yy_pred_2, normalize='true')
    print(f'{conf_mat=}')
    confusion_matrixes[mode] = conf_mat

df1 = pd.concat(df1s)
df2 = pd.concat(df2s)
df2['method'] = 'FBCSP'
df2 = df2[['mode', 'subject', 'acc', 'method']]

# %%
df2a = pd.read_csv('../MEG-EEG-MI-fbcnet/data/fbcnet-fine/mean-accuracy.csv')
df2a = df2a[['mode', 'subject', 'accuracy']]
df2a['mode'] = df2a['mode'].map(lambda e: e.upper())
df2a.columns = ['mode', 'subject', 'acc']
df2a['method'] = 'FBCNet'
df2 = pd.concat([df2, df2a])

# %%
print('-' * 80)
print(df1)
print(df2)


# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 10), gridspec_kw={
                         "width_ratios": [15]*2 + [1]})


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=16, va='bottom')
    return


ax = axes[0, 0]
# Simple text in axis coordinates
add_top_left_notion(ax, 'a')
ax.text(-0.1, 1.05, 'a)', transform=ax.transAxes,
        fontsize=16, va='bottom')
sns.lineplot(df1, x='freq', y='acc', hue='mode', ax=ax)
ax.axhline(0.2, color='gray', linestyle='-.')
ax.set_ylim([0.15, 0.5])
ax.set_title('Accuracy on frequencies')

ax = axes[0, 1]
add_top_left_notion(ax, 'b')
sns.barplot(df2, x='method', y='acc', hue='mode', ax=ax)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.set_ylim([0.2, 0.6])
ax.set_title('Accuracy')

ticklabels = ['1', '2', '3', '4', '5']
ax = axes[1, 0]
add_top_left_notion(ax, 'c')
mode = 'MEG'
sns.heatmap(confusion_matrixes[mode], vmin=0.1, vmax=0.6, annot=True,
            xticklabels=ticklabels, yticklabels=ticklabels, ax=ax,
            cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('Truth')
ax.set_title(f'Confusion matrix ({mode.upper()})')

ax = axes[1, 1]
add_top_left_notion(ax, 'd')
mode = 'EEG'
sns.heatmap(confusion_matrixes[mode], vmin=0.1, vmax=0.6, annot=True,
            xticklabels=ticklabels, yticklabels=ticklabels, ax=ax,
            cbar=True, cbar_ax=axes[1, 2])
ax.set_xlabel('Predicted')
ax.set_ylabel('Truth')
ax.set_title(f'Confusion matrix ({mode.upper()})')

fig.delaxes(axes[0, 2])  # Remove the third subplot

fig.tight_layout()

plt.show()

# %%
# %%
# %%
# %%
# %%

if False:
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

    acc_vote = group['acc2'].mean()['vote']

    # --------------------------------------------------------------------------------
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    x = [np.mean(f) for f in d['freqs']]
    ax = axes[0]
    ax.plot(x, group['acc'].mean().to_list()[:-1])
    ax.hlines(y=0.2, xmin=np.min(x), xmax=np.max(x),
              colors='gray', linestyles='-.')
    # ax.hlines(y=acc_2, xmin=np.min(x), xmax=np.max(x),
    #           colors='blue')
    ax.axhline(acc_vote, color='blue')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('Acc')
    ax.set_title(f'{name} | vote acc = {acc_vote=:0.3f}')
    ax.set_ylim([0.15, 0.5])

    ax = axes[1]
    ticklabels = ['1', '2', '3', '4', '5']
    sns.heatmap(confusion_matrix, vmin=0.1, vmax=0.6, cbar=True, annot=True,
                xticklabels=ticklabels, yticklabels=ticklabels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    ax.set_title('Confusion matrix')

    fig.tight_layout()
    plt.savefig(OUTPUT_DIR.joinpath(f'{name}.png'))
    plt.show()

# %%
df1 = data.query('freqIdx != "vote"')
freqs = [np.mean(f) for f in d['freqs']]
df1['freq'] = df1['freqIdx'].map(lambda i: freqs[i])

df2 = data.query('freqIdx == "vote"')
df2['acc'] = df2['acc2']
print(df1, df2)

# %%
sns.lineplot(df1, x='freq', y='acc')
plt.show()

sns.barplot(df2, y='acc')
plt.show()
# %%

# %%
