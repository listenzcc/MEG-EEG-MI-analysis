# %%
# Imports
from util.easy_import import *
from util.io.file import load

# Constants
data_directory = Path(f'./data/fsaverage')

# %%
# Load data
trained_files = sorted(list(data_directory.rglob('trained.dump')))
raw_files = sorted(list(data_directory.rglob('X-y-times-groups.dump')))

times = load(raw_files[0])['times']

scores = []
coef = []
for p in trained_files:
    subj = p.parent.parent.name
    obj = load(p)
    _scores = obj['scores']
    _coef = obj['coef']
    scores.append(_scores)
    coef.append(_coef)

scores = np.mean(np.array(scores), axis=0)
coef = np.mean(np.array(coef), axis=0)
print(f'{scores.shape=}, {coef.shape=}')

# %%
plt.plot(times, np.mean(scores, axis=0))
plt.show()

# %%
