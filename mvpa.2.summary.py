"""
File: mvpa.2.summary.py
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
import joblib
from util.easy_import import *

data_directory = Path('./data/MVPA')

compile = re.compile(r'^decoding-(?P<mode>[a-z]+)-band-all.dump')
pattern = 'decoding-*-band-all.dump'


# %% ---- 2025-07-23 ------------------------
# Function and class
def find_dump_files(pattern: str = pattern):
    found = list(data_directory.rglob(pattern))
    return found


# %% ---- 2025-07-23 ------------------------
# Play ground
dump_files = find_dump_files()
print(dump_files)

data = []
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

# %% ---- 2025-07-23 ------------------------
# Pending
evts = ['1', '2', '3', '4', '5']
mode = 'meg'

plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for i, mode in enumerate(['meg', 'eeg']):
    v = list(data[data['mode'] == mode]['scores'].values)
    scores_mean = np.mean(v, axis=0)
    scores_median = np.median(v, axis=0)
    scores_std = np.std(v, axis=0)

    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    ax = axes[i, 0]
    ax.plot(times, np.diag(scores_mean), label="score")
    # Draw shadow for standard deviation
    ax.fill_between(times, np.diag(scores_mean - scores_std*0.5),
                    np.diag(scores_mean+scores_std*0.5), alpha=0.2, label='std')
    ax.plot(times, np.diag(scores_median), label="score(median)")
    ax.axhline(1/len(evts), color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AccScore")
    ax.legend(loc='upper right')
    ax.axvline(0.0, color="gray", linestyle="-")
    ax.set_title(f"Decoding over time (mode: {mode})")

    # Plot the temporal generalization matrix

    # Make cmap center with 0.2
    # 创建标准化对象，vmin=0.0, vcenter=0.2, vmax=0.5
    # from matplotlib.colors import TwoSlopeNorm
    # norm = TwoSlopeNorm(vmin=0.0, vcenter=0.2, vmax=0.5)

    ax = axes[i, 1]
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
    ax.set_title(f"Temporal generalization (mode: {mode})")
    ax.axvline(0, color="gray")
    ax.axhline(0, color="gray")
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Accuracy Score")

fig.tight_layout()

fig.savefig(data_directory / 'mvpa_summary.png', dpi=300)

plt.show()

# %% ---- 2025-07-23 ------------------------
# Pending

# %%
