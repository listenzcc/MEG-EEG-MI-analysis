# coding: utf-8

# In[1]:

from util.bands import Bands
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

bands = Bands()

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

    df_raw = tfr.to_data_frame(long_format=True)
    df_raw['name'] = subject_name
    df_raw['evt'] = dct['evt']

    df_bands = []
    for band_name in ['alpha', 'beta']:
        band = bands.get_band(band_name)

        query = ' & '.join(
            ['time < 4.0', 'time > 0.0', f'freq<={band[1]}', f'freq>={band[0]}'])
        df = df_raw.query(query)

        df1 = df.groupby([e for e in df.columns if e not in ['value', 'time']],
                         observed=True)['value'].mean().reset_index()
        print(df1['evt'].unique())

        min_indices = df1.groupby(['channel', 'ch_type', 'name', 'evt'],
                                  observed=True)['value'].idxmin()
        df3 = df1.loc[min_indices]
        df3['freq'] = band_name
        print(df3)

        df_bands.append(df3)

    df_bands = pd.concat(df_bands)

    return df_bands


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


def plot_erd_scatter(df, Opt, evts, band_name):
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
    fig.suptitle(Opt.joint_plot_title.format(
        ' & '.join(evts), band_name))
    fig.tight_layout()
    return fig


def plot_erd_topomap(df, Opt, evts, band_name):
    cols = len(evts)
    fig, axes = plt.subplots(
        1, cols+1,
        figsize=(4*cols+1, 4),
        gridspec_kw={"width_ratios": [10] * cols + [1]})

    if Opt.mode == 'MEG':
        Opt.evoked.pick('mag')

    event_names = [e.title()
                   for e in ['hand', 'wrist', 'elbow', 'shoulder',  'rest']]

    for ax, evt in zip(axes, evts):
        query = [f'evt=="{evt}"']  # , 'time<4.0', 'time>0.0']
        _df = df.query(' & '.join(query))

        # Convert into dB
        _averaged_df = 10*_df.groupby('channel', observed=True)['value'].mean()

        mask = None

        if Opt.mode.lower() == 'eeg':
            mask = np.array([e in ['C3', 'Cz', 'C4']
                            for e in Opt.evoked.info.ch_names])
            print(evt, _averaged_df['C3'], _averaged_df['FC3'])

        if Opt.mode.lower() == 'meg':
            mask = np.array([e in ['MLC42', 'MZC03', 'MRC42']
                            for e in Opt.evoked.info.ch_names])

        # print(_averaged_df)
        # print(Opt.evoked.info.ch_names)
        _array = [_averaged_df[e] for e in Opt.evoked.info.ch_names]

        im, cn = mne.viz.plot_topomap(_array, Opt.evoked.info, image_interp='cubic',
                                      contours=[-3, -1, 0],
                                      mask=mask,
                                      mask_params=dict(marker='o', markerfacecolor='r', markeredgecolor='k',
                                                       linewidth=0, markersize=4),
                                      #   sphere=(0, 0, 0, 0.1),
                                      sphere=(
                                          0, 0, 0, 0.1) if Opt.mode.lower() == 'meg' else None,
                                      extrapolate='local',
                                      cnorm=Opt.norm, cmap=Opt.cmap, size=4, axes=ax, show=False)

        ax.clabel(cn, inline=True, fontsize=10, fmt='-%1.0f dB')

        ax.set_title(f'{event_names[int(evt)-1]}')

    fig.colorbar(im, cax=axes[cols]).ax.set_yscale('linear')

    axes[cols].set_title('dB')
    fig.suptitle(f'TFR topomap {Opt.mode.upper()} ({band_name.title()})')
    fig.tight_layout()
    return fig


def make_df(table, band_name):
    return table.query(f'freq == "{band_name}"')


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=12, va='bottom')
    return

# In[ ]:


class BasicOpt:
    vmin = -5
    vmax = 1
    vcenter = -3
    cmap = 'RdBu'
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)


class EEG_Opt(BasicOpt):
    mode = 'EEG'
    pattern = 'eeg-logratio-*-average-tfr.h5'
    fpath = cache_dir.joinpath('eeg-logratio-averaged-df.h5')
    evoked = md.eeg_epochs['1'][0].average()
    joint_plot_title = 'EEG @evt: {} @band: {}'
    pdf_path = pdf_directory.joinpath('EEG.better.pdf')


class MEG_Opt(BasicOpt):
    mode = 'MEG'
    pattern = 'meg-logratio-*-average-tfr.h5'
    fpath = cache_dir.joinpath('meg-logratio-averaged-df.h5')
    evoked = md.meg_epochs['1'][0].average()
    joint_plot_title = 'MEG @evt: {} @band: {}'
    pdf_path = pdf_directory.joinpath('MEG.better.pdf')


# In[5]:

save_to_pdf = True

for Opt in [MEG_Opt, EEG_Opt]:

    try:
        # assert False, 'Skip it.'
        table = pd.read_hdf(Opt.fpath)
    except:
        found = find_tfr_files(Opt.pattern)
        dfs = [to_df(p) for p in tqdm(found, 'Read TFR')]
        table = pd.concat(dfs)
        table = append_averaged_subject(table).query('name=="Averaged"')
        table.to_hdf(Opt.fpath, key='df', mode='w', format='table')
    print(table)

    # if save_to_pdf:
    #     mpl.use('pdf')

    for band_name in ['alpha', 'beta']:
        band = bands.get_band(band_name)

        df = make_df(table, band_name)
        print(df)

        evts = sorted(df['evt'].unique())

        # ! So large and slow opening the pdf file is!
        # fig = plot_erd_scatter(df, Opt, evts, band_name)
        # if save_to_pdf:
        #     pdf.savefig(fig)

        fig = plot_erd_topomap(df, Opt, evts, band_name)
        fig.savefig(Opt.pdf_path.with_suffix(f'.{band_name}.png'))

        print(f'Plotted {Opt.mode} @band: {band_name}')

    print(f'Saved {Opt.mode} to {Opt.pdf_path}')


# In[ ]:

# %%

# %%
# %%
