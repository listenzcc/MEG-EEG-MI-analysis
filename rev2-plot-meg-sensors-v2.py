# %%
import mne
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import display

# %%
# MEG
obj = json.load(open('./data/meg_info.json', 'r'))
info = mne.Info.from_json_dict(obj)
display(info)

mne.viz.plot_sensors(
    info,
    #  show_names=True,
    kind='topomap',
    ch_type='mag',
    title='MEG sensor distribution',
    sphere=[0, 0, 0, 0.195],
    to_sphere=True,
    show=False)
plt.show()

# %%
# Load channel-to-region dictionary
with open('./data/meg_ch_name_dct.json', 'r') as f:
    ch_map = json.load(f)

display(ch_map)

# %%
display(info['ch_names'])
# %%
