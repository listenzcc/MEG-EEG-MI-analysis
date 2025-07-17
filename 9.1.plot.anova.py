# %%
from mne.stats.cluster_level import _find_clusters
from scipy.ndimage import label
import joblib
from mne.stats import cluster_level
from mne import spatial_src_adjacency
from scipy import stats
import sys
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
data_directory = Path('./data/anova/')
n = 10
k = 4

df_cond = k - 1
df_error = (n - 1) * (k - 1)

# %%
anova = joblib.load(data_directory.joinpath('meg-alpha.dump'))
p_obs = anova['p']
F_obs = anova['f']

# %%

# 1. 定义显著性阈值（单点水平）
p_thresh = 0.05  # 初始p值阈值
# F临界值（dfn, dfd 是F检验自由度）
f_thresh = stats.f.ppf(1 - p_thresh, df_cond, df_error)

clusters = _find_clusters(
    F_obs[:, 0],
    threshold=f_thresh,
    adjacency=subject.adjacency,
    tail=1,
)

print(clusters)


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
        anova = joblib.load(data_directory.joinpath(f'{inp}.dump'))
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
