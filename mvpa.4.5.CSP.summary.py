# %%
from sklearn.metrics import accuracy_score, classification_report
import joblib
from util.easy_import import *

# %%
# Summary the FBCSP
data_directory = Path('./data/MVPA.FBCSP')

data_files = list(data_directory.rglob('cv_scores.npy'))
data_files.sort()
print(data_files)

data = []
for f in data_files:
    d = np.load(f)
    data.append(d)

data = np.array(data)
print(data)
print(np.mean(data))

plt.imshow(data)
plt.colorbar()
plt.show()

# %%
# Summary the CSP
data_directory = Path('./data/MVPA.CSP')

data_files = list(data_directory.rglob('*.dump'))
data_files.sort()
print(data_files)


def vote(preds):
    candidates = {k: 0 for k in [1, 2, 3, 4, 5]}
    for e in preds:
        candidates[e] += 1
    return sorted(candidates.items(), key=lambda e: e[1])


accs = []
for f in data_files:
    d = joblib.load(f)
    y_true = d['y_true']
    y_preds = np.array([v['y_pred']
                       for k, v in d.items() if isinstance(k, int)])
    print(f'{y_true.shape=}, {y_preds.shape=}')
    y_pred = [vote(e)[-1][0] for e in y_preds.T]
    print(d['subject_name'])
    print(classification_report(y_true=y_true, y_pred=y_pred))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    accs.append(acc)

print(np.mean(accs))

# %%


# %%
