# %%
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne import spatial_src_adjacency
from scipy import stats as stats
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
subject.src

# %%
data_directory = Path('./data/tfr-stc-alpha')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df1 = mk_stc_file_table(stc_files)
df1['band'] = 'alpha'

data_directory = Path('./data/tfr-stc-beta')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df2 = mk_stc_file_table(stc_files)
df2['band'] = 'beta'

df = pd.concat([df1, df2])
df.index = range(len(df))

print('Reading stc')
with redirect_stdout(io.StringIO()):
    df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
    df['stc'].map(lambda stc: stc.crop(tmin=-2, tmax=4.5))
    df['stc'].map(lambda stc: stc.apply_baseline((-1, 0)))
df

# %%
stc = df.iloc[0]['stc']

# X shape is (n_subjects, n_times, n_vertices_fsave)
conditions = ['me=="meg"', 'evt=="1"', 'band=="alpha"']
X1 = np.array([stc.data.T for stc in df.query('&'.join(conditions))['stc']])
n_subjects1 = X1.shape[0]
print('X1', X1.shape)

conditions = ['me=="meg"', 'evt=="2"', 'band=="alpha"']
X2 = np.array([stc.data.T for stc in df.query('&'.join(conditions))['stc']])
n_subjects2 = X2.shape[0]
print('X2', X2.shape)

X = [X1, X2]

# Now let's actually do the clustering. This can take a long time...
# Here we set the threshold quite high to reduce computation,
# and use a very low number of permutations for the same reason.
n_permutations = 50
p_threshold = 0.05  # 0.001
f_threshold = stats.distributions.f.ppf(
    1.0 - p_threshold / 2.0, n_subjects1 - 1, n_subjects2 - 1
)

print("Clustering.")
F_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_test(
    X,
    adjacency=subject.adjacency,
    n_jobs=n_jobs,
    n_permutations=n_permutations,
    threshold=f_threshold,
    buffer_size=None,
)
# Now select the clusters that are sig. at p < 0.05 (note that this value
# is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
print(good_cluster_inds)

# %%
fsave_vertices = [np.arange(10242), np.arange(10242)]
stc_all_cluster_vis = summarize_clusters_stc(
    clu, tstep=stc.tstep*1000, vertices=fsave_vertices, subject=subject.subject
)
stc_all_cluster_vis

# %%
# %%
# %%
