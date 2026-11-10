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
# MEG
output_dir = Path('./data')
obj = json.load(open('./data/meg_info.json', 'r'))
info = mne.Info.from_json_dict(obj)
display(info)

# %%
mag_picks = mne.pick_types(info, meg='mag')
mag_names = [info['ch_names'][i] for i in mag_picks]
keep_picks = [p for p, name in zip(mag_picks, mag_names)]

# 3. 只保留不重叠的 mag 通道
keep_picks = [p for p, name in zip(mag_picks, mag_names) if len(name)==10]

# 4. 用剔除后的 picks 重新生成一个临时 info（或者直接传 picks 给后续绘图）
info = mne.pick_info(info, keep_picks)
print(info['ch_names'])

# %%
# Load channel-to-region dictionary
with open('./data/meg_ch_name_dct.json', 'r') as f:
    ch_map = json.load(f)
print(ch_map)

# %%
fig = mne.viz.plot_sensors(
    info,
    kind='topomap',
    ch_type='mag',
    show_names=False,
    pointsize=10,
    linewidth=0.0,
    title='MEG sensor distribution',
    sphere=[0, 0, 0, 0.195],
    to_sphere=True,
    show=False)

ax = fig.gca()

from matplotlib.collections import PathCollection
# 1. 找到传感器散点所在的 PathCollection，并取出它的 2D 坐标
sensor_pts = None
for coll in ax.collections:
    if isinstance(coll, PathCollection):
        offs = coll.get_offsets()
        if len(offs) > 0:
            sensor_pts = np.asarray(offs)
            coll.remove()          # 移除默认着色的散点
            break

# 2. 取出对应通道名（只取 mag，顺序与 plot_sensors 内部一致）
ch_names = [ch for ch, t in zip(info['ch_names'], info.get_channel_types()) if t == 'mag']

# 3. 按名字第三个字符（索引 2）分组
color_map = {
    # 'F': 'tab:red',
    # 'O': 'tab:blue',
    # 'P': 'tab:green',
    # 'T': 'tab:orange',
    # 'C': 'tab:purple',

    'F': '#E76F51',  # Coral
    'C': '#2A9D8F',  # Teal
    'O': '#257B9D',  # Blue
    'T': '#9B72CF',  # Violet
    'P': '#E9A23B',  # Orange
}
groups = {}
for name, xy in zip(ch_names, sensor_pts):
    letter = name[2]
    groups.setdefault(letter, []).append(xy)

# 4. 分组重画散点，大小与原来 pointsize=10 接近（s ≈ pointsize^2 的量级，按需微调）
for letter, pts in groups.items():
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1],
               s=20,
               color=color_map.get(letter, 'gray'),
               zorder=3,
               label=letter)

ax.legend(title='MEG Sensors', loc='upper right', fontsize=8)

fig.savefig(output_dir / 'meg-sensors-colored.png', dpi=300, bbox_inches='tight')
plt.show()

# %%