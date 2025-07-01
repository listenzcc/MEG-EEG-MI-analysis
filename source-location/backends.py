import mne

print(mne.viz.get_3d_backend())
print(mne.viz.set_3d_backend('notebook'))
print(mne.viz.get_3d_backend())
