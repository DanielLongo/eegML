import numpy as np
from torch.utils import data
from mne.datasets import sample


class ForwardModelDataset(data.Dataset):
    def __init__(self, num_examples, num_channels=44, batch_size=64, length=1000):
        self.batch_size = batch_size
        self.length = length
        self.num_channels = num_channels
        self.num_examples = num_examples
        self.preloaded_examples_source = []
        self.preloaded_examples_eegs = []
        self.load_forward_model()
        self.load_examples()

    def __len__(self):
        return int(len(self.preloaded_examples / self.batch_size))

    def __getitem__(self, index):
        print("please use either .getEEGs or .getSources")
        return None

    def getEEGs(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        assert (end < len(self.preloaded_examples_eegs))
        EEG = self.preloaded_examples_eegs[start: end]
        sample = torch.from_numpy(np.asarray(EEG)).type('torch.FloatTensor')
        print("sample shape", sample.shape)
        return sample

    def load_forward_model(self):
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
        self.info = mne.io.read_info(raw_fname)
        fwd = mne.read_forward_solution("sample_forward_model")
        self.fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                      use_cps=True)
        leadfield = self.fwd_fixed['sol']['data']
        self.n_dipoles = leadfield.shape[1]
        self.vertices = [src_hemi['vertno'] for src_hemi in self.fwd_fixed['src']]

    def load_examples(self):
        for i in range(self.num_examples):
            self.preloaded_examples_eegs += [np.random.randn(self.n_dipoles, self.length)]
            self.preloaded_examples_eegs += [self.generate_eeg(i)]

    def generate_eeg(self, source_index):
        stc = mne.SourceEstimate(self.preloaded_examples_eegs[source_index], self.vertices, tmin=0., tstep=1 / 250)
        leadfield = mne.apply_forward(self.fwd_fixed, stc, self.info).data / 1e-9
        return leadfield

    def shuffle(self):
        random.shuffle(self.preloaded_examples)
        self.create_batches()

if __name__ == "__main__":
    FMD = ForwardModelDataset(20, batch_size=4)
    FMD.getEEG(2)