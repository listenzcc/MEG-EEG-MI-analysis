# coding: utf-8

# In[1]:

from matplotlib.colors import Normalize
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


def mk_info():
    # 你的通道列表
    ch_names = ['F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FCz',
                'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']

    # 获取标准位置（不通过完整的蒙太奇）
    # 先创建一个临时 info 获取标准位置
    temp_info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
    temp_info.set_montage('standard_1020')

    # 提取位置
    ch_pos = temp_info.get_montage().get_positions()['ch_pos']

    # 向后移动所有电极 0.2cm
    for ch in ch_pos:
        ch_pos[ch] = ch_pos[ch].copy()  # 创建副本
        ch_pos[ch][1] -= 0.03  # Y轴减小 0.02m

    # 创建新的蒙太奇
    new_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, coord_frame='head')

    # 创建最终的 info
    info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
    info.set_montage(new_montage)
    return info


def find_tfr_files(pattern: str):
    found = list(data_directory.rglob(pattern))
    return found


def read_tfr(path: Path):
    tfr = mne.time_frequency.read_tfrs(path)
    return tfr


def subtract_baseline(df):
    # 方法1：使用分组和变换
    def subtract_negative_mean(group):
        # 计算 t<0 的均值
        negative_mean = group.loc[group['time'] < 0, 'value'].mean()

        # 如果 t<0 的数据不存在，返回原值
        if pd.isna(negative_mean):
            return group['value']

        # 只对 t>0 的值做减法，t<0 的值保持不变
        return group.apply(lambda row: row['value'] - negative_mean if row['time'] > 0 else row['value'], axis=1)

    # 按分组键分组
    group_keys = ['freq', 'channel', 'ch_type', 'name', 'evt']
    df['value'] = df.groupby(group_keys, group_keys=False).apply(
        subtract_negative_mean).reset_index(drop=True)
    return df


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

    # df_raw = subtract_baseline(df_raw)
    display(df_raw)

    df_bands = []
    for band_name in ['alpha', 'beta']:
        band = bands.get_band(band_name)

        query = ' & '.join(
            ['time < 4.0', 'time > 0.0', f'freq<={band[1]}', f'freq>={band[0]}'])
        df = df_raw.query(query)
        df1 = df.groupby([e for e in df.columns if e not in ['value', 'time']],
                         observed=True)['value'].mean().reset_index()

        query = ' & '.join(
            ['time < 0.0', 'time > -2.0', f'freq<={band[1]}', f'freq>={band[0]}'])
        df = df_raw.query(query)
        df2 = df.groupby([e for e in df.columns if e not in ['value', 'time']],
                         observed=True)['value'].mean().reset_index()

        # display(df1)
        # display(df2)
        df1['value'] -= df2['value']

        min_indices = df1.groupby(['channel', 'ch_type', 'name', 'evt'],
                                  observed=True)['value'].idxmin()
        df3 = df1.loc[min_indices]
        df3['freq'] = band_name
        # print(df3)

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


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=12, va='bottom')
    return


def cut_red_end_of_rdbu_r(cut_points=10):
    """
    截断 RdBu_r colormap 的红色端

    参数:
    ----------
    cut_points : int
        要截断的红色端点数（从最红的颜色开始）

    返回:
    ----------
    cmap : LinearSegmentedColormap
        截断后的 colormap
    """
    # 获取原始的 RdBu_r colormap
    rdbu_r = plt.cm.RdBu_r

    # 获取完整的 256 个颜色
    original_colors = rdbu_r(np.linspace(0, 1, 256))

    # 计算要保留的颜色数量
    n_colors = 256 - cut_points

    # 截断红色端：保留从 0 到 (1 - cut_points/256) 的部分
    # 这相当于去掉了最红的 cut_points 个颜色
    keep_fraction = n_colors / 256

    # 获取截断后的颜色
    truncated_colors = original_colors[:n_colors]

    # 创建新的 colormap
    from matplotlib.colors import LinearSegmentedColormap
    new_cmap = LinearSegmentedColormap.from_list(
        f'RdBu_r_truncated_{cut_points}',
        truncated_colors,
        N=n_colors
    )

    return new_cmap


def plot_erd_topomap(df, Opt):
    evts = sorted(df['evt'].unique())
    band_names = sorted(df['freq'].unique())

    assert len(band_names) == 2, f'Incorrect {band_names=}'

    cols = len(evts)
    fig, axes = plt.subplots(
        2, cols+1,
        figsize=(3*cols, 6),
        gridspec_kw={
            "width_ratios": [10] * cols + [1]
        })

    if Opt.mode == 'MEG':
        Opt.evoked.pick('mag')

    event_names = [e.title()
                   for e in ['hand', 'wrist', 'elbow', 'shoulder',  'rest']]

    # for ax, evt in zip(axes, evts):
    for i_band, band_name in enumerate(band_names):
        for i_evt, evt in enumerate(evts):
            # , 'time<4.0', 'time>0.0']
            query = [f'evt=="{evt}"', f'freq=="{band_name}"']
            _df = df.query(' & '.join(query))

            # Convert into dB
            _averaged_df = 20 * _df.groupby('channel', observed=True)[
                'value'].mean()

            _averaged_df -= np.max(_averaged_df)
            if Opt.mode == 'EEG' and band_name == 'alpha' and evt == '4':
                _averaged_df += 1

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

            ax = axes[i_band, i_evt]

            if band_name.lower() == 'alpha' and Opt.mode.lower() == 'meg':
                cnorm = Normalize(vmin=-3.5, vmax=0)

            if band_name.lower() == 'beta' and Opt.mode.lower() == 'meg':
                cnorm = Normalize(vmin=-2, vmax=0)

            if band_name.lower() == 'alpha' and Opt.mode.lower() == 'eeg':
                cnorm = Normalize(vmin=-3, vmax=0)

            if band_name.lower() == 'beta' and Opt.mode.lower() == 'eeg':
                cnorm = Normalize(vmin=-2, vmax=0)

            if Opt.mode.lower() == 'eeg':
                # # Your channel names
                # ch_names = ['F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FCz',
                #             'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5',
                #             'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']

                # # Create a simple info structure
                # info = mne.create_info(ch_names=ch_names,
                #                        sfreq=250., ch_types='eeg')

                # # Get a standard 10-20 montage
                # montage = mne.channels.make_standard_montage('standard_1020')
                # montage = mk_montage()
                # info.set_montage(montage)
                info = mk_info()

            im, cn = mne.viz.plot_topomap(
                _array,
                pos=info if Opt.mode.lower() == 'eeg' else Opt.evoked.info,
                image_interp='cubic',
                # sensors=False,
                extrapolate='head',  # 'head', 'box', 'local',
                # outlines=None,
                cnorm=cnorm,
                # cmap='RdBu_r',  # create_soft_rdbu(),
                cmap=cut_red_end_of_rdbu_r(50),
                size=4,
                axes=ax,
                show=False,
                sphere=[0, 0, 0, 0.2] if Opt.mode.lower() == 'meg' else None
                # contours=[-3, -1, 0],
                # mask=mask,
                # mask_params=dict(marker='o', markerfacecolor='r', markeredgecolor='k',
                #                  linewidth=0, markersize=4),
                #   sphere=(0, 0, 0, 0.1),
                # sphere=(
                #     0, 0, 0, 0.1) if Opt.mode.lower() == 'meg' else None,
            )

            # ax.clabel(cn, inline=True, fontsize=10, fmt='-%1.0f dB')

            if i_evt == 0:
                add_top_left_notion(ax, 'abcdefg'[i_band])

            ax.set_title(f'{event_names[int(evt)-1]}')

        cax = axes[i_band, -1]
        fig.colorbar(im, cax=cax).ax.set_yscale('linear')
        cax.set_title('dB')

    # # Delete the unwanted axes
    # for j in [0, 1, 3, 4]:
    #     fig.delaxes(axes[2, j])

    fig.suptitle(f'TFR topomap {Opt.mode.upper()}')
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
    vmin = -3
    vmax = 0
    vcenter = -3
    cmap = 'RdBu_r'
    # TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)
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

    df = table.copy()  # make_df(table, band_name)
    print(df)

    fig = plot_erd_topomap(df, Opt)
    fig.savefig(Opt.pdf_path.with_suffix(f'.png'))

    print(f'Saved {Opt.mode} to {Opt.pdf_path}')


# %%

# %%

# %%

# %%
