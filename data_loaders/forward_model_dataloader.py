import time
import numpy as np
import mne
import torch
from torch.utils import data
from mne.datasets import sample
import random
import multiprocessing as mp
from itertools import product
import os
# num_cpu = '9' # Set as a string
# os.environ['OMP_NUM_THREADS'] = num_cpu
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
        assert(len(self.preloaded_examples_source) == len(self.preloaded_examples_eegs))

    def __len__(self):
        return int(len(self.preloaded_examples_source) / self.batch_size)

    def __getitem__(self, index):
        print("please use either .getEEGs or .getSources")
        return None

    def getEEGs(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        assert (end <= len(self.preloaded_examples_eegs))
        EEG = self.preloaded_examples_eegs[start: end]
        sample = torch.from_numpy(np.asarray(EEG)).type('torch.FloatTensor')
        return sample

    def getSources(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        assert (end <= len(self.preloaded_examples_source))
        source = self.preloaded_examples_source[start: end]
        sample = torch.from_numpy(np.asarray(source)).type('torch.FloatTensor')
        return sample

    def load_forward_model(self):
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
        self.info = mne.io.read_info(raw_fname)
        fwd = mne.read_forward_solution("../forward_model/sample_forward_model")
        self.fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                      use_cps=True)
        leadfield = self.fwd_fixed['sol']['data']
        self.n_dipoles = leadfield.shape[1]
        self.vertices = [src_hemi['vertno'] for src_hemi in self.fwd_fixed['src']]

    def load_examples(self):
        for i in range(self.num_examples):
            self.preloaded_examples_source += [np.random.randn(self.n_dipoles, self.length) * 1e-9]
            self.preloaded_examples_eegs += [self.generate_eeg(i)]
        # processes =[mp.Process(target=self.generate_eeg, args=(i,)) for i in range(self.num_examples)]
        # for p in processes: p.start()
        # for p in processes: p.join()
        # pool = mp.Pool(mp.cpu_count() - 4) # minus to be safe
        # with mp.Pool(processes=mp.cpu_count() - 4) as pool:
            # results = pool.starmap(self.generate_eeg, product(list(range(self.num_examples))))
            # print(results)
        # self.preloaded_examples_source = [(np.random.randn(self.n_dipoles, self.length) * 1e-9) for i in range(self.num_examples)]
        # self.preloaded_examples_source = [x.get() for x in self.preloaded_examples_source]
        # pool.close()
        # pool.join()
        # jobs = []
        # for i in range(self.num_examples):
        #     p = mp.Process(target=self.generate_eeg, args=(i,), name='daemon')
        #     p.daemon = True
        #     jobs.append(p)
        # p.start()
        # time.sleep(1)
        # p.join()
        # # with mp.Pool(mp.cpu_count() - 4) as pool:
            # self.preloaded_examples_eegs = pool.map_async(self.generate_eeg, list(range(self.num_channels)))
        # self.preloaded_examples_eegs = [pool.(self.generate_eeg(i)) for i in range(self.num_examples)]
            # self.preloaded_examples_eegs = self.preloaded_examples_eegs.get() # [x.get() for x in self.preloaded_examples_eegs]
        # pool.close()
        # pool.join()
        # print("type eegs", len(self.preloaded_examples_eegs)) 
    # def generate_eeg(self, source_index):
    #     stc = mne.SourceEstimate(self.preloaded_examples_source[source_index], self.vertices, tmin=0., tstep=1 / 250)
    #     leadfield = mne.apply_forward(self.fwd_fixed, stc, self.info).data / 1e-9
    #     self.preloaded_examples_eegs = [leadfield[:self.num_channels]]
    #     print("finished", source_index, type(self.preloaded_examples_eegs[-1]))

    def generate_eeg(self, source_index):
        stc = mne.SourceEstimate(self.preloaded_examples_source[source_index], self.vertices, tmin=0., tstep=1 / 250)
        leadfield = mne.apply_forward(self.fwd_fixed, stc, self.info).data / 1e-9
        return leadfield[:self.num_channels]

    def shuffle(self):
        combined = list(zip(self.preloaded_examples_source, self.preloaded_examples_eegs))
        random.shuffle(combined)
        self.preloaded_examples_source, self.preloaded_examples_eegs = zip(*combined)

if __name__ == "__main__":
    FMD = ForwardModelDataset(10, batch_size=4)
    print("EEG Shape", FMD.getEEGs(1).shape, torch.sum(FMD.getEEGs(1)))
    print("Source shape", FMD.getSources(1).shape, torch.sum(FMD.getSources(1)))
    FMD.shuffle()
    print("EEG Shape", FMD.getEEGs(1).shape, torch.sum(FMD.getEEGs(1)))
    print("Source shape", FMD.getSources(1).shape, torch.sum(FMD.getSources(1)))