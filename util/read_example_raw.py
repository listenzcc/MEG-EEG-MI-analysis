from .io.ds_directory_operation import find_ds_directories, read_ds_directory
from .easy_import import *

# Read from file
subject_directory = Path('./rawdata/S01_20220119')
found = find_ds_directories(subject_directory)
md = read_ds_directory(found[0])

# Fetch raw
raw = md.raw

logger.debug(f'Read example md & raw: {md}, {raw}')
