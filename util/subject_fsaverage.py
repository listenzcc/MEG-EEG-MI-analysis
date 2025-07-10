from .easy_import import *

n_jobs = 64


class SubjectFsaverage:
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


if __name__ == '__main__':
    # Prepare subject
    subject = SubjectFsaverage()
    subject.pipeline()
