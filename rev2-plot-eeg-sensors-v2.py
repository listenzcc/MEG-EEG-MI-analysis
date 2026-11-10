# %%
import mne
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import display

from matplotlib.lines import Line2D
from mne.channels.layout import _find_topomap_coords

# %%
# EEG
from joblib import load
info = load('./data/eeg_info.joblib')
output_dir = Path('./data')
display(info)

# %%
# Load channel-to-region dictionary
with open('./data/eeg_ch_name_dct.json', 'r') as f:
    ch_map = json.load(f)
print(ch_map)

# %%
info['ch_names']

# %%
fig = mne.viz.plot_sensors(
    info,
    kind='topomap',
    ch_type='eeg',
    show_names=True,
    # pointsize=10,
    linewidth=0.0,
    title='EEG sensor distribution',
    sphere=[0, 0.03, 0, 0.095],
    to_sphere=True,
    show=False
    )


# # %%

# ax = fig.gca()

# from matplotlib.collections import PathCollection
# # 1. 找到传感器散点所在的 PathCollection，并取出它的 2D 坐标
# sensor_pts = None
# for coll in ax.collections:
#     if isinstance(coll, PathCollection):
#         offs = coll.get_offsets()
#         if len(offs) > 0:
#             sensor_pts = np.asarray(offs)
#             coll.remove()          # 移除默认着色的散点
#             break

# # 2. 取出对应通道名（只取 eeg，顺序与 plot_sensors 内部一致）
# ch_names = [ch for ch, t in zip(info['ch_names'], info.get_channel_types()) if t == 'eeg']

# # 3. 按名字第三个字符（索引 2）分组
# color_map = {
#     # 'F': 'tab:red',
#     # 'O': 'tab:blue',
#     # 'P': 'tab:green',
#     # 'T': 'tab:orange',
#     # 'C': 'tab:purple',

#     'F': '#E76F51',  # Coral
#     'C': '#2A9D8F',  # Teal
#     'O': '#257B9D',  # Blue
#     'T': '#9B72CF',  # Violet
#     'P': '#E9A23B',  # Orange
# }
# groups = {}
# for name, xy in zip(ch_names, sensor_pts):
#     print(name)
#     letter = name[2]
#     groups.setdefault(letter, []).append(xy)

# # 4. 分组重画散点，大小与原来 pointsize=10 接近（s ≈ pointsize^2 的量级，按需微调）
# for letter, pts in groups.items():
#     pts = np.asarray(pts)
#     ax.scatter(pts[:, 0], pts[:, 1],
#                s=20,
#                color=color_map.get(letter, 'gray'),
#                zorder=3,
#                label=letter)

# ax.legend(title='EEG Sensors', loc='upper right', fontsize=8)

fig.savefig(output_dir / 'eeg-sensors-colored.png', dpi=300, bbox_inches='tight')
plt.show()

# %%