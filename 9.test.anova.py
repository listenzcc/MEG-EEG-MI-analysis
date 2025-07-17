# %%
from scipy import stats
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


def fast_rm_anova(data):
    """
    快速手撕重复测量ANOVA
    参数:
        data - 2D numpy数组 (n_subjects × n_conditions)
    返回:
        F值, p值(未校正), p值(GG校正), epsilon(GG校正因子)
    """
    n, k = data.shape[:2]  # n=受试者数, k=条件数

    # 计算各项均值
    grand_mean = np.mean(np.mean(data, axis=0), axis=0)
    subj_means = np.mean(data, axis=1)
    cond_means = np.mean(data, axis=0)

    # 计算平方和 (SS)
    SS_total = np.sum(np.sum((data - grand_mean)**2, axis=0), axis=0)
    SS_subj = np.sum((subj_means - grand_mean)**2, axis=0) * k
    SS_cond = np.sum((cond_means - grand_mean)**2, axis=0) * n
    SS_error = SS_total - SS_subj - SS_cond

    # 计算自由度
    df_cond = k - 1
    df_error = (n - 1) * (k - 1)

    # 计算均方和F值
    MS_cond = SS_cond / df_cond
    MS_error = SS_error / df_error
    F = MS_cond / MS_error

    # 计算自由度
    df_cond = k - 1
    df_error = (n - 1) * (k - 1)

    # 计算均方和F值
    MS_cond = SS_cond / df_cond
    MS_error = SS_error / df_error
    F = MS_cond / MS_error

    # 未校正p值
    p_uncorrected = stats.f.sf(F, df_cond, df_error)

    return F, p_uncorrected


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
data_directory = Path('./data/anova/')
data_directory.mkdir(exist_ok=True, parents=True)

modes = ['meg', 'eeg']
bands = ['alpha', 'beta']
evts = ['1', '2', '3', '4']

for mode, band in itertools.product(modes, bands):
    data = dict()
    for evt in evts:
        conditions = [f'me=="{mode}"', f'evt=="{evt}"', f'band=="{band}"']
        X = np.array([stc.data for stc in df.query(
            '&'.join(conditions))['stc']])
        data[evt] = X

    data = np.stack([data[e] for e in evts])
    f, p = fast_rm_anova(data)
    print(mode, band, data.shape, f.shape, p.shape)
    res = dict(
        f=f,
        p=p,
        mode=mode,
        band=band
    )
    import joblib
    path = data_directory.joinpath(f'{mode}-{band}.dump')
    joblib.dump(res, path)
    print(f'Saved to {path}')


# %%


# %%
