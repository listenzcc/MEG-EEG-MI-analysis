# %%
import seaborn as sns
from util.easy_import import *
from util.io.file import load


# %%
DATA_DIR = Path(f'./data/MVPA-accumulate')
MODE = 'eeg'

# %%
files = sorted(list(DATA_DIR.rglob(f'{MODE}-results.dump')))
print(files)

# %%
data = []
for f in files:
    results = load(f)
    subject = results['subject']
    y_true = results['y_true']
    print(subject)

    for obj in results['decoded']:
        tmax = obj['tmax']
        y_pred = obj['y_pred']
        acc = np.sum(y_true == y_pred) / len(y_true)
        data.append((subject, tmax, acc))

print(data)
df = pd.DataFrame(data, columns=['subject', 'tmax', 'acc'])
print(df)

# %%
sns.lineplot(df, x='tmax', y='acc')
plt.show()

# %%
