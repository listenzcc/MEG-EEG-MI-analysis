'''
! The python script requires the X.
! It also requires the exports to correctly work.

export MNE_3D_OPTION_MULTI_SAMPLES=1
export MNE_3D_OPTION_ANTIALIAS=false
'''

from util.easy_import import *
from util.read_example_raw import md

raw = md.raw

SUBJECT = 'fsaverage'
SUBJECT_DIR = mne.datasets.fetch_fsaverage()
SUBJECTS_DIR = SUBJECT_DIR.parent
print(SUBJECTS_DIR, SUBJECT)

mne.viz.plot_alignment(
    raw.info,
    trans=SUBJECT,  # SUBJECT_DIR / 'bem/fsaverage-head.fif',
    subject=SUBJECT,
    # dig=False,
    dig='fiducials',
    meg=["helmet", "sensors"],
    # eeg=['original', 'projected'],
    eeg=True,
    coord_frame='head',
    subjects_dir=SUBJECTS_DIR,
    surfaces=dict(brain=0.2, outer_skull=0.1, head=None),
)

input('Press enter to quit.')
