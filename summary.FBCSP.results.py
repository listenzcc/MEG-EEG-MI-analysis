
"""
File: summary.FBCSP.results.py
Author: Chuncheng Zhang
Date: 2025-07-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    1. Summary of MVPA with raw epochs.
    2. Compare with FBCSP accumulate voting results.

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

data_directory = Path('./data/MVPA')

compile = re.compile(r'^decoding-(?P<mode>[a-z]+)-band-all.dump')
pattern = 'decoding-*-band-all.dump'


# %% ---- 2025-07-23 ------------------------
# Function and class
def find_dump_files(pattern: str = pattern):
    found = list(data_directory.rglob(pattern))
    return found


# %% ---- 2025-07-23 ------------------------
# Read time-by-time decoding results
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
df1 = pd.DataFrame(data)
print(times)
print(df1)


# %% Read accumulate voting results

data_directories = [
    Path('./data/MVPA.FBCSP.vote-accumulate.eeg'),
    Path('./data/MVPA.FBCSP.vote-accumulate.meg'),
]

tmax_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

data = []

for data_directory in data_directories:
    name = data_directory.name
    mode = name.split('.')[-1]
    print(f'Working with {name=}, {mode=}')

    data_files = sorted(list(data_directory.rglob('*.dump')))
    print(data_files)
    for f in tqdm(data_files, 'Reading files'):
        subject = f.parent.name
        obj = joblib.load(f)
        y_true = obj['y_true']
        _res = pd.DataFrame(obj['results'])
        for tmax in tqdm(tmax_array, 'Accumulating'):
            _selected = _res.query(f'tmax=={tmax}')
            joint_proba = np.prod(_selected['y_proba'])
            y_pred = np.argmax(joint_proba, axis=1) + 1
            acc = np.mean(y_true == y_pred)
            data.append({
                'acc': acc,
                'tmax': tmax,
                'mode': mode,
                'subject': subject,
                'y_pred': y_pred,
            })

df2 = pd.DataFrame(data)
print(df2)

# %%
display(df1)
display(df2)

# %% ---- 2025-07-23 ------------------------
# Pending
evts = ['1', '2', '3', '4', '5']


plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={
                         "width_ratios": [4, 6]})


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=16, va='bottom')
    return


for i, mode in enumerate(['meg', 'eeg']):
    v = list(df1[df1['mode'] == mode]['scores'].values)
    scores_mean = np.mean(v, axis=0)
    scores_median = np.median(v, axis=0)
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
    ax.set_ylim([0.15, 0.5])

    sns.lineplot(df2[df2['mode'] == mode], x='tmax', y='acc', ax=ax)

    # Plot the temporal generalization matrix

    # Make cmap center with 0.2
    # 创建标准化对象，vmin=0.0, vcenter=0.2, vmax=0.5
    # from matplotlib.colors import TwoSlopeNorm
    # norm = TwoSlopeNorm(vmin=0.0, vcenter=0.2, vmax=0.5)

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
    # cbar.ax.set_position([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar.set_label("Accuracy Score")

fig.tight_layout()

# fig.savefig(data_directory / 'mvpa_summary.png', dpi=300)

plt.show()

# %% ---- 2025-07-23 ------------------------
# Pending

# %%
