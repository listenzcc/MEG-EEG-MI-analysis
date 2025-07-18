
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


def get_stc():
    stc_data_directory = Path('./data/tfr-stc-alpha')
    stc_files = find_stc_files('*.stc-lh.stc', stc_data_directory)
    df = mk_stc_file_table(stc_files)
    stc = mne.read_source_estimate(df.iloc[0]['virtualPath'])
    stc.crop(tmin=-2, tmax=4.5)
    stc.subject = subject.subject
    return stc


# %%
subject = SubjectFsaverage()
parc = 'aparc_sub'
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
labels_parc_df = pd.DataFrame([(e.name, e)
                              for e in labels_parc], columns=['name', 'label'])
labels_parc_df

# %%
while True:
    inp = input('Input like meg-alpha | eeg-beta >>').strip()

    if inp == 'q':
        break

    if inp == '':
        continue

    try:
        #
        anova_data_directory = Path('./data/anova/')
        anova = joblib.load(anova_data_directory.joinpath(f'{inp}.dump'))
        p_obs = anova['p']
        F_obs = anova['f']

        print(F_obs.shape, p_obs.shape)

        #
        stc = get_stc()
        t_map = [e < 4 and e > 1 for e in stc.times]
        for i, s in enumerate(np.mean(F_obs[:, t_map], axis=1)):
            stc.data[i] = s
        stc

        #
        brain = stc.plot(
            initial_time=1,
            hemi="split",
            views=["lat"],
            subjects_dir=SubjectFsaverage.subjects_dir,
            transparent=True,
            clim=dict(kind="value", lims=(10, 20, 30)),
            title=f'F {inp}',
            size=(1600, 800)
        )
    except Exception as err:
        logger.exception(err)
        pass

sys.exit(0)

# %%
