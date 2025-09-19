"""
File: mvpa.source.1.dev.summary.py
Author: Chuncheng Zhang
Date: 2025-09-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the results of mvpa.source.1.dev.py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-19 ------------------------
# Requirements and constants
from util.easy_import import *
from util.io.file import load

# %%
data_directory = Path(f'./data/fsaverage')

# %% ---- 2025-09-19 ------------------------
# Function and class


# %% ---- 2025-09-19 ------------------------
# Play ground
dump_files = sorted(list(data_directory.rglob('trained-scores.dump')))
print(dump_files)

scores_stack = []
current_max = {
    'val': 0,
    'scores': None,
    'subject': None,
}
data = []

for p in dump_files:
    scores = load(p)
    m = np.mean(scores)
    data.append((p.parent.name, scores, m))
    if m > current_max['val']:
        current_max['val'] = m
        current_max['scores'] = scores
        current_max['subject'] = p.parent.name
    print(p.parent.name, scores.shape, np.mean(scores), np.std(scores))
    scores_stack.append(scores)
data = pd.DataFrame(data, columns=['subject', 'scores', 'maxAverage'])
data.sort_values('maxAverage', ascending=False, inplace=True)
print(data)

# %%

scores_stack = np.array(scores_stack)
print(scores_stack.shape)

plt.style.use('ggplot')

times = np.linspace(-1, 4, scores_stack.shape[-1])
avg_scores = np.mean(np.mean(scores_stack, axis=0), axis=0)
plt.plot(times, avg_scores, label='Average')

for i in range(2):
    scores = data.iloc[i]['scores']
    label = data.iloc[i]['subject']
    plt.plot(times, np.mean(scores, axis=0), label=label, alpha=0.7)
plt.legend()
plt.show()


# %% ---- 2025-09-19 ------------------------
# Pending


# %% ---- 2025-09-19 ------------------------
# Pending
