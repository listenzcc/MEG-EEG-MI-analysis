# %%
from sklearn.metrics import accuracy_score, classification_report
from util.easy_import import *
from util.io.file import load


# %%
# Summary the CSP
data_directory = Path('./data/MVPA.FBCSP.vote')

data_files = list(data_directory.rglob('*.dump'))
data_files.sort()
print(data_files)


def vote(preds):
    candidates = {k: 0 for k in [1, 2, 3, 4, 5]}
    for i, e in enumerate(preds):
        candidates[e] += 1
    return sorted(candidates.items(), key=lambda e: e[1])


data = []
for f in data_files:
    d = load(f)
    subject = d['subject_name']

    y_true = d['y_true']
    y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]
    for i, y_pred in enumerate(y_preds):
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        data.append({'subject': subject, 'acc': acc, 'freqIdx': i})

    y_pred = [vote(e)[-1][0] for e in np.array(y_preds).T]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    data.append({'subject': subject, 'acc': acc, 'freqIdx': 'vote'})

data = pd.DataFrame(data)
print(data)

group = data.groupby(by='freqIdx')
print(group['acc'].mean())


# %%

# %%
