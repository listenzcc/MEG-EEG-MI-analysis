"""
File: mvpa.4.FBCSP.vote.megAreas.summary.py
Author: Chuncheng Zhang
Date: 2025-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Perform the voting for the FBCSP proba. of megAreas.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-11 ------------------------
# Requirements and constants
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from util.easy_import import *
from util.io.file import load

plt.style.use('ggplot')

# %%

data_directory = Path(f'./data/MVPA.FBCSP.megAreas.vote')
output_directory = Path(f'./data/MVPA.FBCSP.megAreas.vote.results')
output_directory.mkdir(exist_ok=True, parents=True)


# Read necessary files
meg_ch_name_dct = json.load(open('./data/meg_ch_name_dct.json'))
dump_files = sorted(list(data_directory.rglob('freq-CSP-results.dump')))
objs = [load(p) for p in dump_files]


# %% ---- 2025-09-11 ------------------------
# Function and class
def explain_ch_mark(ch_mark):
    if len(ch_mark) == 1:
        return 'B'  # Both hemisphere

    elif ch_mark.startswith('L'):
        return 'L'  # Left hemisphere

    elif ch_mark.startswith('R'):
        return 'R'  # Right hemisphere

    return ValueError()


def analysis_obj(obj):
    freqs = np.mean(obj['freqs'], axis=1)
    y_true = obj['y_true']
    results = obj['results']
    subject = obj['subject_name']

    dfv = []
    dfs = []

    for ch_mark in sorted(list(meg_ch_name_dct.keys())):
        data = [res for res in results if res['ch_mark'] == ch_mark]
        accs = [accuracy_score(y_true, e['y_pred']) for e in data]

        # y_preds shape: bands x samples x classes
        y_probas = np.array([e['y_proba'] for e in data])
        y_pred = np.argmax(np.prod(np.array(y_probas), axis=0), axis=1) + 1
        accv = accuracy_score(y_true, y_pred)

        # Record vote acc
        dfv.append({
            'acc': accv,
            'chMark': ch_mark,
            'subject': subject
        })

        # Record single freq acc
        df = pd.DataFrame()
        df['freq'] = freqs
        df['acc'] = accs
        df['subject'] = subject
        df['chMark'] = ch_mark
        dfs.append(df)

    df_freqs = pd.concat(dfs)
    df_vote = pd.DataFrame(dfv)

    return df_freqs, df_vote


# %% ---- 2025-09-11 ------------------------
# Play ground
obj = objs[0]
df_freqs = []
df_vote = []
for obj in objs:
    dfs, dfv = analysis_obj(obj)
    df_freqs.append(dfs)
    df_vote.append(dfv)

df_freqs = pd.concat(df_freqs)
df_vote = pd.concat(df_vote)

for df in [df_freqs, df_vote]:
    df['chHemi'] = df['chMark'].map(explain_ch_mark)
    df['chMark'] = df['chMark'].map(lambda e: e[-1])

print(df_freqs)
print(df_vote)

# %%
# Summary
group = df_vote.groupby(['chHemi', 'chMark'])
sum_vote = group.mean(True)
print(sum_vote)

group = df_freqs.groupby(['chHemi', 'chMark', 'freq'])
sum_freq = group.mean(True)
print(sum_freq)

# %% ---- 2025-09-11 ------------------------
# Pending
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=16, va='bottom')
    return


ax = axes[0]
add_top_left_notion(ax, 'a')
sns.barplot(df_vote, x='chMark', y='acc', hue='chHemi', ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_title('Voted acc using different MEG channels')

ax = axes[1]
add_top_left_notion(ax, 'b')
sns.lineplot(sum_freq, x='freq', y='acc', hue='chMark', style='chHemi', ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_title('Acc of single freqs using different MEG channels')

plt.tight_layout()
plt.savefig(output_directory.joinpath('Acc with different MEG channels.png'))
plt.show()


# %% ---- 2025-09-11 ------------------------
# Pending
