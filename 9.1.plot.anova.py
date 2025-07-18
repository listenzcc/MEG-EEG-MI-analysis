# %%
import sys
import joblib

from mne import spatial_src_adjacency
from mne.stats import cluster_level
from mne.stats import fdr_correction
from mne.stats.cluster_level import _find_clusters

from scipy import stats
from scipy.ndimage import label

from util.easy_import import *

compile = re.compile(r'^(?P<me>[a-z]+)-evt(?P<evt>\d+).stc-lh.stc')


# %%


class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent
    src_path = subject_dir.joinpath('bem', 'fsaverage-ico-5-src.fif')
    src = mne.read_source_spaces(src_path)
    adjacency = spatial_src_adjacency(src)


def find_stc_files(pattern: str, data_directory: Path):
    found = list(data_directory.rglob(pattern))
    return found


def mk_stc_file_table(stc_files: list):
    data = []
    for p in stc_files:
        name = p.name
        dct = compile.search(name).groupdict()
        dct.update({
            'path': p,
            'virtualPath': p.parent.joinpath(p.name.replace('.stc-lh', ''))
        })
        data.append(dct)
    return pd.DataFrame(data)


# %%
subject = SubjectFsaverage()

parc = 'aparc_sub'

labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
labels_parc_df = pd.DataFrame([(e.name, e)
                              for e in labels_parc], columns=['name', 'label'])
labels_parc_df

# %%
data_directory = Path('./data/tfr-stc-alpha')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df = mk_stc_file_table(stc_files)
stc = mne.read_source_estimate(df.iloc[0]['virtualPath'])
stc.crop(tmin=-2, tmax=4.5)
stc.subject = subject.subject
stc

# %%
print(stc.data.shape)
print(subject.adjacency.data.shape)
# %%
anova_data_directory = Path('./data/anova/')
n = 10
k = 4

df_cond = k - 1
df_error = (n - 1) * (k - 1)

# %%
# 1. 定义显著性阈值（单点水平）
p_thresh = 0.05  # 初始p值阈值
# F临界值（dfn, dfd 是F检验自由度）
f_thresh = stats.f.ppf(1 - p_thresh, df_cond, df_error)
alpha = 0.05
print(f'Using p*={p_thresh} (f={f_thresh})')

anova = joblib.load(anova_data_directory.joinpath('meg-alpha.dump'))
p_obs = anova['p']
F_obs = anova['f']
p_fdr = np.zeros_like(p_obs) + 1

for t in tqdm(range(F_obs.shape[1]), 'Find clusters'):
    clusters = _find_clusters(
        F_obs[:, t],
        threshold=f_thresh,
        adjacency=subject.adjacency,
        tail=1,
    )
    for idx in clusters[0]:
        _p_obs = p_obs[idx, t]
        reject, _p_fdr = fdr_correction(_p_obs, alpha=alpha, method='indep')
        p_fdr[idx, t] = _p_fdr

print(clusters)
print(p_fdr.shape)
print(p_fdr)

# %%
_p_obs = p_obs[clusters[0][0], 0]
print(_p_obs)
reject, p_fdr = fdr_correction(_p_obs, alpha=0.05, method='indep')
print(reject, p_fdr)

# %%

F = np.linspace(0, 10, 1000)
p_uncorrected = stats.f.sf(F, df_cond, df_error)
plt.plot(F, p_uncorrected)
plt.hlines([0.05, 0.01], F[0], F[-1], colors='black', linestyles='--')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.show()

# %%
while True:
    inp = input('Input like meg-alpha | eeg-beta >>')
    if inp == 'q':
        break

    try:
        anova = joblib.load(anova_data_directory.joinpath(f'{inp}.dump'))
        stc.data = anova['f']

        # Plot in 3D view
        brain = stc.plot(
            initial_time=1,
            hemi="split",
            views=["lat", "med"],
            subjects_dir=SubjectFsaverage.subjects_dir,
            transparent=True,
            clim=dict(kind="value", pos_lims=(10, 20, 30)),
        )

    except Exception as err:
        logger.exception(err)
        pass

sys.exit(0)


# %%
