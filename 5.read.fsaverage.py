"""
File: 5.read.fsaverage.py
Author: Chuncheng Zhang
Date: 2025-06-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read fsaverage files for fwd.

    By looking at Display sensitivity maps for EEG and MEG sensors plot the sensitivity maps for EEG and compare it with the MEG, can you justify the claims that:
    <https://mne.tools/stable/auto_examples/forward/forward_sensitivity_maps.html#ex-sensitivity-maps>
    <https://mne.tools/stable/generated/mne.sensitivity_map.html#mne.sensitivity_map>

    - MEG is not sensitive to radial sources
    - EEG is more sensitive to deep sources

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-27 ------------------------
# Requirements and constants
import io
from contextlib import redirect_stdout

from util.easy_import import *
from util.read_example_raw import md

md.generate_epochs(**dict(tmin=-2, tmax=5, decim=6))
raw = md.raw
eeg_epochs = md.eeg_epochs
meg_epochs = md.meg_epochs
print(raw)
print(eeg_epochs)
print(meg_epochs)

n_jobs = 32


# %% ---- 2025-06-27 ------------------------
# Function and class


class SubjectFsaverage:
    # Local
    local_cache = Path('./data/fsaverage')

    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent
    src_path = subject_dir.joinpath('bem', 'fsaverage-ico-5-src.fif')
    bem_path = subject_dir.joinpath(
        'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fwd_fname_template = 'fsaverage-{}-fwd.fif'

    # MNE trans
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

    def __init__(self):
        self.local_cache.mkdir(exist_ok=True, parents=True)

    def check_files(self):
        print('---- Check files ----')
        files = [self.src_path, self.bem_path]
        dirs = [self.subject_dir, self.local_cache]
        [print(e.is_file(), e) for e in files]
        [print(e.is_dir(), e) for e in dirs]

    def pipeline(self):
        self.check_files()
        self.read_source_spaces()
        self.read_bem_solution()

    def read_forward_solution(self, info, t: str):
        p = self.local_cache.joinpath(self.fwd_fname_template.format(t))
        if t.lower() == 'meg':
            eeg = False
            meg = True
        elif t.lower() == 'eeg':
            eeg = True
            meg = False

        try:
            fwd = mne.read_forward_solution(p)
        except Exception:
            fwd = mne.make_forward_solution(info, trans=self.trans, src=self.src_path,
                                            bem=self.bem_path, eeg=eeg, meg=meg, mindist=5.0, n_jobs=n_jobs)
            mne.write_forward_solution(p, fwd)
        return fwd

    def read_source_spaces(self):
        self.src = mne.read_source_spaces(self.src_path)
        return self.src

    def read_bem_solution(self):
        self.bem = mne.read_bem_solution(self.bem_path)
        return self.bem

    def make_inverse_operator(self, info, fwd, noise_cov):
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            info, fwd, noise_cov)
        return inverse_operator


# Prepare subject
subject = SubjectFsaverage()
subject.pipeline()
fwd_eeg = subject.read_forward_solution(eeg_epochs.info, 'eeg')
fwd_meg = subject.read_forward_solution(meg_epochs.info, 'meg')
print('src', subject.src)
print('bem', subject.bem)

# Compute cov
print('Computing noise_cov')
with redirect_stdout(io.StringIO()):
    method = ['empirical']
    noise_cov = dict(
        eeg=mne.compute_covariance(eeg_epochs, tmax=0, method=method),
        meg=mne.compute_covariance(meg_epochs, tmax=0, method=method),
    )
print(noise_cov)

# Compute inverse operator
print('Computing inverse_operator')
with redirect_stdout(io.StringIO()):
    inverse_operator = dict(
        eeg=subject.make_inverse_operator(
            eeg_epochs.info, fwd_eeg, noise_cov['eeg']),
        meg=subject.make_inverse_operator(
            meg_epochs.info, fwd_meg, noise_cov['meg']),
    )
print(inverse_operator)


# Compute inverse
snr = 3.0  # Standard assumption for average data but using it for single trial
kwargs = dict(
    lambda2=1.0 / snr**2,
    method="dSPM"  # use dSPM method (could also be MNE or sLORETA)
)

# Compute EEG inverse
print('Computing EEG inverse')
with redirect_stdout(io.StringIO()):
    eeg_epochs.load_data()
    mne.set_eeg_reference(eeg_epochs, projection=True)
    eeg_stcs = mne.minimum_norm.apply_inverse_epochs(eeg_epochs,
                                                     inverse_operator['eeg'],
                                                     **kwargs)
print(eeg_stcs)

# Compute MEG inverse
print('Computing MEG inverse')
with redirect_stdout(io.StringIO()):
    meg_stcs = mne.minimum_norm.apply_inverse_epochs(meg_epochs,
                                                     inverse_operator['meg'],
                                                     **kwargs)
print(meg_stcs)


# %% ---- 2025-06-27 ------------------------
# Play ground
# parc, str, The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
parc = 'aparc_sub'

labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir)
labels_parc_dict = {e.name: e for e in labels_parc}
print(labels_parc_dict)

# %%
stc = meg_stcs[20]
print(stc)

# %%
brain = stc.plot(hemi='both')

# %%
s = stc.in_label(labels_parc_dict['postcentral_8-lh'])
inspect(s)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
y = np.mean(s.data, axis=0)
ax.plot(s.times, y)
plt.show()

# %% ---- 2025-06-27 ------------------------
# Pending


# %% ---- 2025-06-27 ------------------------
# Pending
