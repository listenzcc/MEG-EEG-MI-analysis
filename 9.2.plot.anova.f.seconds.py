
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
parc = 'PALS_B12_Brodmann'
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

        for t_start in [0, 1, 2, 3]:
            t_map = [e < t_start+1 and e > t_start for e in stc.times]
            for i, s in enumerate(np.mean(F_obs[:, t_map], axis=1)):
                stc.data[i, t_map] = s

        # Put all 0~4 into <0 stc.data
        t_map_1 = [e > 0 and e < 4 for e in stc.times]
        t_map_2 = [e < 0 for e in stc.times]
        for i, s in enumerate(np.mean(F_obs[:, t_map_1], axis=1)):
            stc.data[i, t_map_2] = s
        stc

        #
        for t_start in tqdm([-1, 0, 1, 2, 3]):
            t = t_start+0.5
            brain_kwargs = dict(alpha=1.0, background="white",
                                cortex="low_contrast")
            brain = stc.plot(
                initial_time=t,
                hemi="split",
                views=["lat"],
                subjects_dir=SubjectFsaverage.subjects_dir,
                transparent=True,
                background='white',
                colorbar=False,
                clim=dict(kind="value", lims=(10, 20, 30)),
                title=f'F {inp}',
                size=(1600, 800),
                show_traces=False,
                brain_kwargs=brain_kwargs,
            )
            brain.add_text(0.5, 0.9, f'{t}', 'title', font_size=16)

            bas = [1, 4, 7, 40, 39, 19]
            colors = ['blue', 'green', 'black', 'cyan', 'magenta', 'yellow']
            for ba, color in zip(bas, colors):
                for hemi in ['lh', 'rh']:
                    label_name = f'Brodmann.{ba}-{hemi}'
                    label = labels_parc_df.query(f'name=="{label_name}"')[
                        'label'].values[0]
                    brain.add_label(label, hemi=hemi,
                                    color=color, borders=True)

            # for label_name, color in [
            #     (labels_parc_df.query('name=="Brodmann.1-lh"')
            #      ['label'].values[0], 'black'),
            #     ('BA4a', 'blue'),
            #     ('BA4p', 'green'),
            #     ('V1', 'white'),
            #     ('V2', 'yellow'),
            # ]:
            #     for hemi in ['lh', 'rh']:
            #         brain.add_label(label_name, hemi=hemi,
            #                         color=color, borders=True)
            #         # brain.add_label("BA4a", hemi=hemi,
            #         #                 color="green", borders=True)
            #         # brain.add_label("BA4p", hemi=hemi,
            #         #                 color="blue", borders=True)
            # break
            brain.save_image(f'./figures/anova-f-{inp}-{t=}.png')
            brain.close()

    except Exception as err:
        logger.exception(err)
        pass

sys.exit(0)

# %%
