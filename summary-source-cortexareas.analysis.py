"""
File: summary-source-cortexareas.analysis.py
Author: Chuncheng Zhang
Date: 2025-11-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Working on the source analysis results.
    Analysis the tfr details.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-21 ------------------------
# Requirements and constants
import seaborn as sns
import statsmodels.api as sm

from scipy import stats
from itertools import product
from util.easy_import import *

sns.set_theme(context='paper', style='ticks', font_scale=1)

# %%
CACHE_DIR = Path(f'./data/cache/')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path('./data')

# MODE = 'meg'  # 'meg' or 'eeg'

# %%
TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}


class ROI:
    names = {
        'central': [f'postcentral_{i}' for i in [3, 4, 5, 6, 7, 8, 9]] + [f'precentral_{i}' for i in [5, 6, 7, 8]],
        # 'parietal': [f'superiorparietal_{i}' for i in [11, 12]] + [f'inferiorparietal_{i}' for i in [1, 4, 3, 8]],
        # 'occipital': [f'lateraloccipital_{i}' for i in [2, 3, 6, 7, 8, 9]]
    }
    colors = {
        'central': 'cyan',
        'parietal': 'green',
        'occipital': 'blue'
    }

# %% ---- 2025-11-21 ------------------------
# Function and class


def load_labels(parc='aparc_sub'):
    labels_parc = mne.read_labels_from_annot(
        'fsaverage', parc=parc, subjects_dir=mne.datasets.fetch_fsaverage().parent
    )
    df = pd.DataFrame(
        [(e.name, e) for e in labels_parc], columns=["name", "label"]
    )

    assert len(df['name'].unique()) == len(df), "Duplicate label names found!"

    df.index = df['name']
    return df


def find_stc(subject, band, mode, evt):
    folder = Path(f'./data/tfr-stc-{band}/')
    subjects = [e.name for e in folder.iterdir()]
    subject = [e for e in subjects if e.startswith(subject)][0]
    fpath = folder.joinpath(f'{subject}/{mode}-evt{evt}.stc')
    return fpath


def read_compute_stc(subject, band, mode, evt, tmin=-2, tmax=5):
    fpath = find_stc(subject, band, mode, evt)

    stc = mne.read_source_estimate(fpath)
    stc = stc.in_label(label)
    stc.crop(tmin=tmin, tmax=tmax)

    # Define frequency bands
    bands = {
        'alpha': (8, 15),  # (8, 13),
        'beta': (13, 30),
    }

    if band not in bands:
        raise ValueError(f"Band must be one of {list(bands.keys())}")
    fmin, fmax = bands[band]

    # 频率采样
    freqs = np.linspace(fmin, fmax, num=10)
    n_cycles = freqs / 2.0  # 常见经验设置

    # stc.data: (n_vertices, n_times)
    data = stc.data

    # 升维为 (n_epochs=1, n_channels=n_vertices, n_times)
    data = data[np.newaxis, :, :]

    sfreq = 1.0 / stc.tstep

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        zero_mean=True,
    )

    # power shape: (1, n_vertices, n_freqs, n_times)
    power = power[0]

    return {
        "power": power,          # (n_vertices, n_freqs, n_times)
        "freqs": freqs,
        "times": stc.times,
        # "vertices": stc.vertices
    }


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=12, va='bottom')
    return


def combine_labels(selected_labels):
    # 获取所有选中标签的顶点和位置信息
    all_vertices = []
    all_pos = []
    all_values = []
    all_hemi = None

    for label in selected_labels:
        all_vertices.append(label.vertices)
        all_pos.append(label.pos)
        all_values.append(label.values)
        if all_hemi is None:
            all_hemi = label.hemi
        elif all_hemi != label.hemi:
            print("警告：标签来自不同半球")

    # 合并顶点（注意去重）
    combined_vertices = np.unique(np.concatenate(all_vertices))
    # combined_pos = np.mean(np.concatenate(all_pos, axis=0), axis=0)  # 取平均位置
    # combined_values = np.mean(np.concatenate(all_values), axis=0)  # 取平均值
    combined_pos = np.concatenate(all_pos, axis=0)  # 取平均位置
    combined_values = np.concatenate(all_values)  # 取平均值

    # 创建新标签
    combined_label = mne.Label(
        vertices=combined_vertices,
        pos=combined_pos,
        values=combined_values,
        hemi=all_hemi,
        name='combined_label',  # 自定义名称
        subject=selected_labels[0].subject,  # 假设所有标签来自同一被试
        color='red'  # 可选：设置颜色
    )

    return combined_label


# %% ---- 2025-11-21 ------------------------
# Play ground
label_df = load_labels(parc='aparc_sub')
label_df = label_df[label_df['name'].map(
    lambda e: e.endswith('-lh') and e[:-3] in ROI.names['central'])]
display(label_df)

# %%
label = combine_labels(label_df['label'].to_list())
display(label)

# %%
dfs = []
subjects = [f'S{sub+1:02d}' for sub in range(10)]
events = list(TASK_TABLE.keys())
modes = ['meg', 'eeg']
bands = ['alpha', 'beta']

# %%
try:
    table = joblib.load(OUTPUT_DIR / 'source-central-analysis.dump')
except:
    dfs = []
    for sub, evt, mode, band in tqdm(product(subjects, events, modes, bands),
                                     total=np.prod([len(subjects), len(events), len(modes), len(bands)])):
        tfr = read_compute_stc(sub, band, mode=mode, evt=evt)
        dfs.append([sub, evt, mode, band, tfr])

    table = pd.DataFrame(dfs, columns=['sub', 'evt', 'mode', 'band', 'tfr'])
    joblib.dump(table, OUTPUT_DIR / 'source-central-analysis.dump')

display(table)

# %%
mode = 'meg'
band = 'alpha'
evt = '1'

fig, axes = plt.subplots(4, 5, figsize=(12, 10))

for mode, band, evt in tqdm(product(modes, bands, events)):
    j = events.index(evt)
    i = modes.index(mode) * 2 + bands.index(band)
    ax = axes[i][j]
    query = '&'.join([f'mode=="{mode}"', f'band=="{band}"', f'evt=="{evt}"'])
    df = table.query(query)
    tfr = df.iloc[0]['tfr']
    times = [e for e in tfr['times'] if e > -1 and e < 4]
    freqs = tfr['freqs']
    # Currently, mat shape is (n_samples, n_vertices, n_freqs, n_times)
    mat = np.array(df['tfr'].map(lambda e: np.log10(e['power'])).to_list())
    mat = mat[:, :, :, [e in times for e in tfr['times']]]
    # Mean the 1st, 2nd dim of the mat
    mat = np.mean(np.mean(mat, axis=0), axis=0)

    # Make x as times, y as freqs
    im = ax.imshow(
        mat,
        aspect='auto',
        origin='lower',
        extent=[
            times[0], times[-1],      # x: time (s)
            freqs[0], freqs[-1]       # y: frequency (Hz)
        ]
    )
    ax.set_title(f'{mode}-{band}-{TASK_TABLE[evt]}')
    ax.set_aspect('auto')  # 保持自动纵横比
    plt.colorbar(im)

fig.tight_layout()
plt.show()

# %%

# %%
