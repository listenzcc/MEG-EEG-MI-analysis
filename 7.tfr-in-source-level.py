'''
Reference:
- <https://mne.tools/stable/auto_examples/inverse/dics_epochs.html#sphx-glr-auto-examples-inverse-dics-epochs-py>
'''

# %%
import sys
from mne.beamformer import apply_dics_tfr_epochs, make_dics
from mne.time_frequency import csd_tfr
from util.easy_import import *
# from util.read_example_raw import md
from util.subject_fsaverage import SubjectFsaverage
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# subject_directory = Path('./rawdata/S01_20220119')

parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f'./data/tfr-stc-alpha/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

freqs = [e for e in range(8, 13)]  # alpha
# freqs = [e for e in range(15, 25)]  # beta

# %%
# md.generate_epochs(**dict(tmin=-2, tmax=5, decim=6))
# raw = md.raw
# eeg_epochs = md.eeg_epochs
# meg_epochs = md.meg_epochs
# print(raw)
# print(eeg_epochs)
# print(meg_epochs)

# %%


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    epochs_kwargs = {'tmin': -3, 'tmax': 5, 'decim': 6}
    use_latest_ds_directories = 8  # 8

    # Read from file
    mds = []
    found = find_ds_directories(subject_directory)
    mds.extend([read_ds_directory(p)
                for p in found[-use_latest_ds_directories:]])

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)
        md.eeg_epochs.load_data()
        md.meg_epochs.load_data()
        md.eeg_epochs.apply_baseline((-2, 0))
        md.meg_epochs.apply_baseline((-2, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    return eeg_epochs, meg_epochs


evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs = concat_epochs(mds)
print(eeg_epochs)
print(meg_epochs)


# %%
subject = SubjectFsaverage()
subject.pipeline()
fwd_eeg = subject.read_forward_solution(eeg_epochs.info, 'eeg')
fwd_meg = subject.read_forward_solution(meg_epochs.info, 'meg')
print('src', subject.src)
print('bem', subject.bem)

# %%


def compute_stc(epochs, fwd, freqs, tmin, tmax):
    epochs_tfr = epochs.compute_tfr(
        'morlet', freqs, n_cycles=freqs, return_itc=False, output="complex", average=False, n_jobs=n_jobs)
    # epochs_tfr.crop(tmin=tmin, tmax=tmax)
    print(epochs_tfr)

    # Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
    # We are interested in increases in power relative to the baseline period, so
    # we will make a separate CSD for just that period as well.
    csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=tmax)
    baseline_csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=0)

    # compute scalar DICS beamfomer
    filters = make_dics(
        epochs.info,
        fwd,
        csd,
        noise_csd=baseline_csd,
        pick_ori="max-power",
        reduce_rank=True,
        real_filter=True,
    )

    # project the TFR for each epoch to source space
    epochs_stcs = apply_dics_tfr_epochs(
        epochs_tfr, filters, return_generator=True)

    # average across frequencies and epochs
    data = np.zeros((fwd["nsource"], epochs_tfr.times.size))
    for epoch_stcs in epochs_stcs:
        for stc in epoch_stcs:
            data += (stc.data * np.conj(stc.data)).real

    stc.data = data / len(epochs) / len(freqs)
    # apply a baseline correction
    # stc.apply_baseline((tmin, 0))
    return stc


tmin = -0.5
tmax = 4

epochs = meg_epochs['1']
fwd = fwd_meg
for evt in evts:
    stc = compute_stc(eeg_epochs[evt], fwd_eeg, freqs, tmin, tmax)
    stc.save(data_directory.joinpath(f'eeg-evt{evt}.stc'), overwrite=True)
    stc = compute_stc(meg_epochs[evt], fwd_meg, freqs, tmin, tmax)
    stc.save(data_directory.joinpath(f'meg-evt{evt}.stc'), overwrite=True)

sys.exit(0)


brain = stc.plot(
    subjects_dir=subject.subjects_dir,
    hemi="both",
    # views="dorsal",
    # initial_time=1.2,
    # brain_kwargs=dict(show=False),
    # add_data_kwargs=dict(
    #     fmin=fmax / 10,
    #     fmid=fmax / 2,
    #     fmax=fmax,
    #     scale_factor=0.0001,
    #     colorbar_kwargs=dict(label_font_size=10),
    # ),
)
input('Press enter to quit.')
