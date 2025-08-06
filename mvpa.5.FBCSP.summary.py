# %%
from util.easy_import import *

data_directory = Path('./data/MVPA.FBCSP')

data_files = list(data_directory.rglob('cv_scores.npy'))
data_files.sort()
print(data_files)

data = []
for f in data_files:
    d = np.load(f)
    data.append(d)

data = np.array(data)
print(data)
print(np.mean(data))

plt.imshow(data)
plt.colorbar()
plt.show()

# %%
