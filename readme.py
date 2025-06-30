#!/usr/bin/env python
# coding: utf-8

# ## Experiment Design

# In[1]:


import plotly.express as px
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.easy_import import *
EventsInFirstTwoSessions = {
    '1': '左手',
    '2': '右手',
    '3': '双手',
    '4': '双脚',
    '5': '静息',
}


# In[2]:


EventsInOtherSessions = {
    '1': '手',
    '2': '腕',
    '3': '肘',
    '4': '肩',
    '5': '静息',
}


# ## Load Example Data

# In[3]:


def apply_trans(trans, pts, move=True):
    """Apply a transform matrix to an array of points.

    Parameters
    ----------
    trans : array, shape = (4, 4) | instance of Transform
        Transform matrix.
    pts : array, shape = (3,) | (n, 3)
        Array with coordinates for one or n points.
    move : bool
        If True (default), apply translation.

    Returns
    -------
    transformed_pts : shape = (3,) | (n, 3)
        Transformed point(s).
    """

    if isinstance(trans, dict):
        trans = trans['trans']
    pts = np.array(pts)
    if pts.size == 0:
        return pts.copy()

    # apply rotation & scale
    out_pts = np.dot(pts, trans[:3, :3].T)
    # apply translation
    if move:
        out_pts += trans[:3, 3]

    return out_pts


def fig_raw_dig(raw):
    '''
    Parse the digital points (dig) of the system from the raw.

    The dig contains FIFFV_POINT_CARDINAL and FIFFV_POINT_EXTEA
    - The FIFFV_POINT_CARDINAL is the 3-point system of nose and ears in the both sides,
        which is tagged on the subject's head to locate the brain;
    - The FIFFV_POINT_EXTEA are the inside points of the MEG device, it tells where the hat is.
    '''

    dev_head_t = raw.info['dev_head_t']

    dig = raw.info['dig']
    df = pd.DataFrame(
        [
            (e['ident'], e['kind']._name, e['r'])
            for e in dig
        ],
        columns=['ident', 'kind', 'loc']
    )

    df['pos'] = df['loc'].map(lambda e: apply_trans(dev_head_t, e[:3]))
    df['x'] = df['pos'].map(lambda e: e[0])
    df['y'] = df['pos'].map(lambda e: e[1])
    df['z'] = df['pos'].map(lambda e: e[2])

    df['size'] = 1
    select = df.query('kind == "FIFFV_POINT_CARDINAL"')
    df.loc[select.index, 'size'] = 2

    df['title'] = 'dig'

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='kind',
                        size_max=10, size='size', title='Dig-layout')

    return fig, df


def fig_raw_sensors(raw, eeg_epochs, meg_epochs):
    '''
    Parse the sensor points of the system from the raw.

    The sensors are the meg sensors on the device.
    '''
    dev_head_t = raw.info['dev_head_t']

    df = pd.DataFrame(
        [
            (e['scanno'], e['logno'], e['ch_name'], e['kind']._name, e['loc'])
            for e in eeg_epochs.info['chs'] + meg_epochs.info['chs']
            if not np.isnan(e['loc'][0])
        ],
        columns=['scanno', 'logno', 'chname', 'kind', 'loc']
    )

    df['pos'] = df['loc'].map(lambda e: apply_trans(dev_head_t, e[:3]))

    df['x'] = df['pos'].map(lambda e: e[0])
    df['y'] = df['pos'].map(lambda e: e[1])
    df['z'] = df['pos'].map(lambda e: e[2])
    df['size'] = 1
    df['title'] = 'sensor'

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='kind',
                        size_max=10, size='size', title='Sensor-layout')

    return fig, df


def fig_raw_mix(df1, df2):
    '''
    Joint plot the dig and sensors figures.
    '''
    df = pd.concat([df1, df2], axis=0)

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='kind',
                        size_max=10, size='size', title='Mix-layout')
    return fig, df


subject_directory = Path('./rawdata/S01_20220119')
found = find_ds_directories(subject_directory)
md = read_ds_directory(found[0])

# Read info
raw = md.raw
if False:
    dps = raw.info['dig'][:3]
    dev_head_t = raw.info['dev_head_t']
    # Apply transform and inverse-transform
    trans = dev_head_t['trans'].copy()
    pt = np.array(dps[1]['r'])
    # Transform
    pt1 = np.dot(pt, trans[:3, :3].T) + trans[:3, 3]
    # Downward the LPA by 1 cm
    # pt1[2] -= 0.01
    # Inverse transform
    pt2 = np.dot(pt1 - trans[:3, 3], np.linalg.inv(trans[:3, :3].T))
    # Write back to the info
    dps[1]['r'] = pt2

md.generate_epochs()


# In[4]:


raw = md.raw
eeg_epochs = md.eeg_epochs
meg_epochs = md.meg_epochs

print(raw)
print(eeg_epochs)
print(meg_epochs)


# In[5]:


eeg_epochs.info['chs']


# In[6]:


meg_epochs.info['chs']


# In[7]:


fig, df1 = fig_raw_dig(raw)
fig.show()

fig, df2 = fig_raw_sensors(raw, eeg_epochs, meg_epochs)
fig.show()

fig, _ = fig_raw_mix(df1, df2)
fig.show()


# In[8]:


df1


# In[9]:


df2


# In[10]:


SUBJECT = 'fsaverage'
SUBJECT_DIR = mne.datasets.fetch_fsaverage()
SUBJECTS_DIR = SUBJECT_DIR.parent
print(SUBJECTS_DIR, SUBJECT)

# parc, str, The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
parc = 'aparc_sub'

labels_parc = mne.read_labels_from_annot(
    SUBJECT, parc=parc, subjects_dir=SUBJECTS_DIR)

df3 = pd.DataFrame([(v.name, len(v.values), v.color, v.pos, v.vertices)
                   for v in tqdm(labels_parc, 'Generate DataFrame')],
                   columns=['name', 'num', 'rgba', 'pos', 'vertices'])

df3['xyz'] = df3['pos'].map(lambda e: np.mean(e, axis=0))
df3['x'] = df3['xyz'].map(lambda e: e[0])
df3['y'] = df3['xyz'].map(lambda e: e[1])
df3['z'] = df3['xyz'].map(lambda e: e[2])
df3['kind'] = 'brain'
df3['size'] = 1


def fig_all(df1, df2, df3):
    '''
    Joint plot the dig and sensors figures.
    '''
    columns = ['x', 'y', 'z', 'kind', 'size']
    df = pd.concat([e[columns] for e in [df1, df2, df3]], axis=0)

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='kind',
                        size_max=10, size='size', title='All-layout')
    return fig, df


fig, _ = fig_all(df1, df2, df3)
fig.show()


# In[21]:


mne.viz.plot_alignment(
    raw.info,
    trans=SUBJECT,  # SUBJECT_DIR / 'bem/fsaverage-head.fif',
    subject=SUBJECT,
    # dig=False,
    dig='fiducials',
    meg=["helmet", "sensors"],
    # eeg=['original', 'projected'],
    eeg=[],
    coord_frame='head',
    subjects_dir=SUBJECTS_DIR,
    surfaces=dict(brain=0.2, outer_skull=0.1, head=None),
)


# In[20]:


print(mne.viz.get_3d_backend())

input()

# In[ ]:
