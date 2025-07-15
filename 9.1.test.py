# %%
from util.easy_import import *

compile = re.compile(r'^(?P<me>[a-z]+)-evt(?P<evt>\d+).stc-lh.stc')

# %%


class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


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

# %%
