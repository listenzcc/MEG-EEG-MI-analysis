# %% ---- 2025-06-27 ------------------------
from util.easy_import import *

data_directory = Path('./data/tfr-stc-alpha')
data_directory = Path('./data/tfr-stc-beta')
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

# %% ---- 2025-06-27 ------------------------
# Play ground
while True:
    while True:
        inp = input('Input like meg-1 | eeg-1 | q >> ')
        if inp == 'q':
            import sys
            sys.exit(0)
        try:
            a, b = inp.split('-')
        except:
            continue
        if a in ['meg', 'eeg'] and b in ['1', '2', '3', '4', '5']:
            break

    selected = df.query(f'me=="{a}"').query(f'evt=="{b}"')
    print(selected)

    # Average stcs
    stc = selected.iloc[0]['stc']
    mat = np.zeros_like(stc.data)
    for s in selected['stc'].values:
        mat += stc.data
    mat /= len(selected)
    stc.data = mat

    # Prepare the stc
    stc.subject = subject.subject
    stc.crop(tmin=-2, tmax=4.5)
    stc.apply_baseline((None, 0))
    print(stc)

    # Plot in 3D view
    brain = stc.plot(hemi='both')

# %%
