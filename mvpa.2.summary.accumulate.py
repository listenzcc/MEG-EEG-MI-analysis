"""
File: mvpa.2.summary.accumulate.py
Author: Chuncheng Zhang
Date: 2025-07-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary of MVPA with raw epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-23 ------------------------
# Requirements and constants
import seaborn as sns
import joblib
from util.easy_import import *


# %% ---- 2025-07-23 ------------------------
# Function and class
def find_dump_files():
    data_directory = Path('./data/MVPA')
    pattern = 'decoding-*-band-all.dump'
    found = list(data_directory.rglob(pattern))
    return found


def find_accumulate_dump_files():
    data_directory = Path('./data/MVPA-accumulate')
    pattern = '*-results.dump'
    found = list(data_directory.rglob(pattern))
    return found


# %% ---- 2025-07-23 ------------------------
# Play ground
data = []

compile = re.compile(r'^decoding-(?P<mode>[a-z]+)-band-all.dump')
dump_files = find_dump_files()
print(dump_files)
for p in dump_files:
    dct = compile.search(p.name).groupdict()
    mode = dct['mode']
    print(f'Loading {p}...')
    decoding = joblib.load(p)
    times = decoding.pop('times')
    data.append(decoding)
    print(f'Decoding {mode} loaded.')
    print(decoding)


data = pd.DataFrame(data)
print(times)
print(data)

# %%
data_accumulate = []
dump_files = find_accumulate_dump_files()
print(dump_files)
for p in dump_files:
    mode = p.name.split('-')[0]
    obj = joblib.load(p)
    y_true = obj['y_true']
    for _decoded in obj['decoded']:
        y_pred = _decoded['y_pred']
        t = _decoded['tmax']
        acc = np.mean(y_true == y_pred)
        data_accumulate.append((t, acc, mode))
data_accumulate = pd.DataFrame(data_accumulate, columns=['t', 'acc', 'mode'])
print(data_accumulate)
sns.lineplot(data_accumulate, x='t', y='acc', hue='mode')
plt.show()

# %%

# %% ---- 2025-07-23 ------------------------
# Pending
evts = ['1', '2', '3', '4', '5']
mode = 'meg'

plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={
                         "width_ratios": [4, 6]})


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=16, va='bottom')
    return


for i, mode in enumerate(['meg', 'eeg']):
    v = list(data[data['mode'] == mode]['scores'].values)
    scores_mean = np.mean(v, axis=0)
    scores_std = np.std(v, axis=0)

    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    ax = axes[i, 0]
    add_top_left_notion(ax, 'a' if i == 0 else 'c')
    ax.plot(times, np.diag(scores_mean), label="score(avg)")
    # Draw shadow for standard deviation
    ax.fill_between(times, np.diag(scores_mean - scores_std*0.5),
                    # , label='std')
                    np.diag(scores_mean+scores_std*0.5), alpha=0.2)
    ax.axhline(1/len(evts), color="gray", linestyle="--")  # , label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AccScore")
    ax.legend(loc='lower right')
    ax.axvline(0.0, color="gray", linestyle="-")
    ax.set_title(f"Scores over time ({mode.upper()})")
    ax.set_ylim([0.15, 0.4])

    # Plot the temporal generalization matrix

    ax = axes[i, 1]
    add_top_left_notion(ax, 'b' if i == 0 else 'd')
    im = ax.imshow(
        scores_mean,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        vmin=0.0,
        vmax=0.5,
        # norm=norm,
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"Temporal generalization ({mode.upper()})")
    ax.axvline(0, color="gray")
    ax.axhline(0, color="gray")
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)  # , fraction=0.6, aspect=20)

fig.tight_layout()

plt.show()

# %% ---- 2025-07-23 ------------------------
# Pending

# %%
