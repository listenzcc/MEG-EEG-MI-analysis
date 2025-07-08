# %% ---- 2025-06-27 ------------------------
from util.easy_import import *

data_directory = Path('./data/fsaverage-alpha')
compile = re.compile(r'^(?P<me>[a-z]+)-evt(?P<evt>\d+).stc-lh.stc')


# %%
class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


def find_stc_files(pattern: str):
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
stc_files = find_stc_files('*.stc-lh.stc')
df = mk_stc_file_table(stc_files)

print('Reading stc')
df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
print(df)

# %%
selected = df.query('me=="eeg"').query('evt=="1"')
print(selected)
n_jobs = 32

stc = selected.iloc[0]['stc']
mat = np.zeros_like(stc.data)
for s in selected['stc'].values:
    stc.data = stc.data.astype(np.float64)
    stc.filter(l_freq=8, h_freq=12, n_jobs=n_jobs)
    stc.apply_hilbert(envelope=True, n_jobs=n_jobs)
    mat += stc.data
mat /= len(selected)
stc.data = mat

# %% ---- 2025-06-27 ------------------------
# Play ground
# parc, str, The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
# parc = 'aparc_sub'
# labels_parc = mne.read_labels_from_annot(
#     subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
# labels_parc_dict = {e.name: e for e in labels_parc}
# print(labels_parc_dict)

# %%
stc.subject = subject.subject
# stc.crop(tmin=-0.3, tmax=0.7)
stc.crop(tmin=-1, tmax=2)
print(stc)

# %%
# Plot in 3D view
brain = stc.plot(hemi='both')
input('Press enter to quit.')

# %%
# Block to prevent brain being closed automatically
# s = stc.in_label(labels_parc_dict['postcentral_8-lh'])
# inspect(s)
# fig, ax = plt.subplots(1, 1, figsize=(8, 3))
# y = np.mean(s.data, axis=0)
# ax.plot(s.times, y)
# plt.show()
