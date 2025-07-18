# coding: utf-8

# In[1]:

from util.read_example_raw import md
from util.easy_import import *

data_directory = Path('./data/TFR')
compile = re.compile(
    r'^(?P<me>[a-z]+)-(?P<mode>[a-z]+)-(?P<evt>\d+)-average-tfr.h5')

cache_dir = Path('./data/cache')
cache_dir.mkdir(exist_ok=True, parents=True)

pdf_directory = Path('./data/pdf/ERD-jointPlot')
pdf_directory.mkdir(exist_ok=True, parents=True)

epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6*2}
md.generate_epochs(**epochs_kwargs)
print(md.eeg_epochs)
print(md.meg_epochs)

# In[2]:


def find_tfr_files(pattern: str):
    found = list(data_directory.rglob(pattern))
    return found


def read_tfr(path: Path):
    tfr = mne.time_frequency.read_tfrs(path)
    return tfr


def to_df(path: Path, channels: list = []):
    '''
    Read TFR from path and pick channels.
    If channels list is empty, use all channels.
    Return df with columns: freq, time, channel, ch_type, value, name, evt
    '''
    name = path.name
    dct = compile.search(name).groupdict()
    subject_name = path.parent.name
    tfr = read_tfr(path)

    if len(channels) > 0:
        tfr.pick(channels)

    df = tfr.to_data_frame(long_format=True)
    df['name'] = subject_name
    df['evt'] = dct['evt']
    return df


def append_averaged_subject(df: pd.DataFrame):
    '''
    Append averaged subject to the df
    '''
    # Average value across name
    columns = [c for c in df.columns if c not in ['name', 'value']]
    _df = df.groupby(columns, observed=True)['value'].mean().reset_index()
    _df['name'] = 'Averaged'
    df = pd.concat([df, _df])
    return df


# In[3]:


# In[ ]:


class EEG_Opt:
    vmin = -0.5
    vmax = 0.5
    vcenter = 0
    cmap = 'RdBu'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)
    pattern = 'eeg-logratio-*-average-tfr.h5'
    fpath = cache_dir.joinpath('eeg-logratio-averaged-df.h5')
    evoked = md.eeg_epochs['1'][0].average()
    joint_plot_title = 'EEG @evt: {} @band: {}'
    pdf_path = pdf_directory.joinpath('EEG.pdf')


class MEG_Opt:
    vmin = -0.5
    vmax = 0.5
    vcenter = 0
    cmap = 'RdBu'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)
    pattern = 'meg-logratio-*-average-tfr.h5'
    fpath = cache_dir.joinpath('meg-logratio-averaged-df.h5')
    evoked = md.meg_epochs['1'][0].average()
    joint_plot_title = 'MEG @evt: {} @band: {}'
    pdf_path = pdf_directory.joinpath('MEG.pdf')


# In[5]:

for Opt in [EEG_Opt, MEG_Opt]:

    try:
        table = pd.read_hdf(Opt.fpath)
    except:
        found = find_tfr_files(Opt.pattern)
        dfs = [to_df(p) for p in tqdm(found, 'Read TFR')]
        table = pd.concat(dfs).query('time <= 5.0')
        table = append_averaged_subject(table).query('name=="Averaged"')
        table.to_hdf(Opt.fpath, key='df', mode='w', format='table')
    print(table)

    mpl.use('pdf')
    with PdfPages(Opt.pdf_path) as pdf:
        band = (8, 13)
        band_name = 'alpha'

        query = ' & '.join(
            ['time <= 5.0', f'freq<={band[1]}', f'freq>={band[0]}'])
        df = table.copy().query(query)

        # Average across freqs
        columns = [c for c in df.columns if c not in ['freq', 'value']]
        df = df.groupby(columns, observed=True)['value'].mean().reset_index()

        print(df)

        evts = sorted(df['evt'].unique())
        cols = len(evts)

        fig_width = 4 * cols  # inch
        fig_height = 4  # inch
        fig, axes = plt.subplots(
            1, cols+1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [10] * cols + [1]})

        for evt in evts:
            j = evts.index(evt)

            ax = axes[j]

            query = f'evt=="{evt}"'
            _df = df.query(query)

            ax.scatter(_df['time'], _df['channel'],
                       c=_df['value'], **Opt.scatter_kwargs)
            ax.set_title(f'ERD @band:{band_name} @evt: {evt}')
            ax.set_xlabel('Time (s)')
            if j == 0:
                ax.set_ylabel(f'Channel')
            else:
                ax.set_yticks([])

        fig.colorbar(axes[0].collections[0], cax=axes[cols],
                     orientation='vertical').ax.set_yscale('linear')

        fig.tight_layout()
        pdf.savefig(fig)

        scalings = {'eeg': 1, 'mag': 1}
        for evt in evts:
            mat = df.query(f'evt=="{evt}"').pivot(
                index='channel', columns='time', values='value').to_numpy()
            Opt.evoked.pick(list(df['channel'].unique()))
            Opt.evoked.data = mat
            fig = Opt.evoked.plot_joint(
                title=Opt.joint_plot_title.format(evt, band_name),
                times=[0, 0.5, 1, 2, 3, 4],
                ts_args=dict(scalings=scalings),
                topomap_args=dict(scalings=scalings, cmap=Opt.cmap, cnorm=Opt.norm))
            pdf.savefig(fig)


# In[ ]:
