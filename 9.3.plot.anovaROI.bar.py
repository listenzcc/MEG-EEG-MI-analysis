# %%
import seaborn as sns
from util.easy_import import *

compile = re.compile(r'^(?P<me>[a-z]+)-evt(?P<evt>\d+).stc-lh.stc')

ROI_labels = [
    # meg-alpha
    ('alpha', 'precentral_5-lh'),
    ('alpha', 'precentral_8-lh'),
    ('alpha', 'postcentral_7-lh'),
    ('alpha', 'postcentral_9-lh'),
    ('alpha', 'postcentral_4-lh'),
    ('alpha', 'supramarginal_6-lh'),
    ('alpha', 'supramarginal_9-lh'),
    ('alpha', 'superiorparietal_5-lh'),
    ('alpha', 'superiorparietal_3-lh'),
    # meg-beta
    ('beta', 'postcentral_5-lh'),
    ('beta', 'postcentral_4-lh'),
    ('beta', 'superiorparietal_5-lh'),
    ('beta', 'precentral_4-lh'),
    ('beta', 'precentral_6-lh'),
    ('beta', 'precentral_8-lh'),
    ('beta', 'precentral_5-lh'),
    ('beta', 'precentral_3-lh'),
    ('beta', 'superiorfrontal_18-lh'),
    ('beta', 'superiorfrontal_17-lh'),
    ('beta', 'superiorfrontal_12-lh'),
    ('beta', 'caudalmiddlefrontal_6-lh'),
    ('beta', 'postcentral_7-lh'),
    ('beta', 'postcentral_3-lh'),
]

ROI_labels_df = pd.DataFrame(ROI_labels, columns=['band', 'label'])
ROI_labels_df

# %%
ROI_labels_df_pivot = ROI_labels_df.pivot(
    index='label', columns='band', values='band')
ROI_labels_df_pivot


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
parc = 'aparc_sub'
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
labels_parc_df = pd.DataFrame([(e.name, e)
                              for e in labels_parc], columns=['name', 'label'])
labels_parc_df

# %%

data_directory = Path('./data/tfr-stc-alpha')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df1 = mk_stc_file_table(stc_files)
df1['band'] = 'alpha'

data_directory = Path('./data/tfr-stc-beta')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df2 = mk_stc_file_table(stc_files)
df2['band'] = 'beta'

df = pd.concat([df1, df2]).query('me=="meg"')
df.index = range(len(df))

print('Reading stc')
with redirect_stdout(io.StringIO()):
    df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
    df['stc'].map(lambda stc: stc.crop(tmin=-2, tmax=4.5))
    df['stc'].map(lambda stc: stc.apply_baseline((-1, 0)))
    df['stc'].map(lambda stc: stc.crop(tmin=1, tmax=4))
data_df = df
data_df

# %%
data = []
for _, se in tqdm(df.iterrows(), total=len(df)):
    band = se['band']
    stc = se['stc']
    evt = se['evt']
    for label_name in set(e[1] for e in ROI_labels):
        label = labels_parc_df.query(f'name=="{label_name}"').iloc[0]['label']
        d = np.mean(stc.in_label(label).data)
        data.append({'band': band, 'evt': evt,
                    'label': label_name, 'erd': d})
data = pd.DataFrame(data)
data

# %%
data = data.sort_values(by=['evt', 'erd'])
data

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 30))

sns.set_theme(style="whitegrid")
sns.despine(bottom=True, left=True)

for ax, band in zip(axes, ['alpha', 'beta']):
    roi_label_names = ROI_labels_df_pivot.query(
        f'{band}=="{band}"').index.values
    df = data.query(f'band=="{band}"')
    df = df[df['label'].map(lambda l: l in roi_label_names)]
    sns.boxplot(df, y='label', x='erd', hue='evt', ax=ax,
                legend=True, showfliers=False)
    ax.set_title(band)

fig.tight_layout()
fig.savefig(Path('./img/compare-erd-alpha-beta-ROI.png'))

plt.show()

# %%

# %%
