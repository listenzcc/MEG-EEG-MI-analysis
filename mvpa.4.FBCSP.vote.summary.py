# %%
from sklearn import metrics
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
yy_true = []
yy_pred = []
yy_pred_2 = []
for f in data_files:
    d = load(f)
    subject = d['subject_name']

    y_true = d['y_true']
    y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]
    y_probas = [v['y_proba'] for k, v in d.items() if isinstance(k, int)]
    for i, y_pred in enumerate(y_preds):
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        data.append({'subject': subject, 'acc': acc, 'freqIdx': i})

    y_pred = [vote(e)[-1][0] for e in np.array(y_preds).T]
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
print(metrics.confusion_matrix(y_true=yy_true, y_pred=yy_pred))

print(metrics.classification_report(y_true=yy_true, y_pred=yy_pred_2))
print(metrics.confusion_matrix(y_true=yy_true, y_pred=yy_pred_2))

# %%

# %%
