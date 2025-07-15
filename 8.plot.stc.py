# %% ---- 2025-06-27 ------------------------
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
subject = SubjectFsaverage()

data_directory = Path('./data/tfr-stc-alpha')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df1 = mk_stc_file_table(stc_files)
df1['band'] = 'alpha'
data_directory = Path('./data/tfr-stc-beta')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df2 = mk_stc_file_table(stc_files)
df2['band'] = 'beta'
df = pd.concat([df1, df2])

print('Reading stc')
df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
print(df)

# %% ---- 2025-06-27 ------------------------
# Play ground


def plot_brain(mode, evt, band):
    conditions = [
        f'me=="{mode}"',
        f'evt=="{evt}"',
        f'band=="{band}"',
    ]

    selected = df.query('&'.join(conditions))
    print(selected)

    # Average stcs
    stc = selected.iloc[0]['stc']
    mat = np.zeros_like(stc.data)
    for s in selected['stc'].values:
        mat += s.data
    mat /= len(selected)
    stc.data = mat

    # Prepare the stc
    stc.subject = subject.subject
    stc.crop(tmin=-2, tmax=4.5)
    stc.apply_baseline((-1, 0))
    print(stc)

    # Plot in 3D view
    brain = stc.plot(
        initial_time=1,
        hemi="split",
        views=["lat", "med"],
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
    )
    return brain


def plot_brain_sub_evts(mode, evt1, evt2, band):
    # evt1
    conditions1 = [
        f'me=="{mode}"',
        f'evt=="{evt1}"',
        f'band=="{band}"',
    ]
    selected1 = df.query('&'.join(conditions1))

    # evt2
    conditions2 = [
        f'me=="{mode}"',
        f'evt=="{evt2}"',
        f'band=="{band}"',
    ]
    selected2 = df.query('&'.join(conditions2))

    stc = selected1.iloc[0]['stc']

    # Average stcs
    mat1 = np.zeros_like(stc.data)
    for s in selected1['stc'].values:
        mat1 += s.data
    mat1 /= len(selected1)

    mat2 = np.zeros_like(stc.data)
    for s in selected2['stc'].values:
        mat2 += s.data
    mat2 /= len(selected2)

    stc.data = mat1 - mat2

    # Prepare the stc
    stc.subject = subject.subject
    stc.crop(tmin=-2, tmax=4.5)
    # stc.apply_baseline((-1, 0))

    # Plot in 3D view
    brain = stc.plot(
        initial_time=1,
        hemi="split",
        views=["lat", "med"],
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
        clim=dict(kind="value", pos_lims=(50, 75, 100)),
    )
    return brain


# %%
# ! CLI
while True:
    while True:
        inp = input('Input like meg-1-alpha | sub-meg-1-2-alpha | q >> ')
        if inp == 'q':
            break

        if inp.startswith('sub'):
            try:
                _, mode, evt1, evt2, band = inp.split('-')
                plot_brain_sub_evts(mode, evt1, evt2, band)
            except:
                pass

        try:
            mode, evt, band = inp.split('-')
            plot_brain(mode, evt, band)
        except:
            pass

    print('ByeBye')
    import sys
    sys.exit(0)


# %%
