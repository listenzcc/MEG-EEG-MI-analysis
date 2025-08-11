# %%
from util.easy_import import *
from util.io.file import load

# %%
data_directory = Path(f'./data/MVPA.6/S01_20220119')

files = list(data_directory.rglob('*.dump'))

for f in files:
    dct = load(f)
    print(f)
    print(dct['report'])

# %%
data_directory = Path(f'./data/MVPA.CSP/S01_20220119')

files = list(data_directory.rglob('*.dump'))
for f in files:
    dct = load(f)
    print(f)
    print(dct)

# %%
