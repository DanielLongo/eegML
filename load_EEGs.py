import mne
import time
import datetime
import numpy as np
import os
import torch
from torch.utils import data
import h5py
import random

class EEGDataset(data.Dataset):
	def __init__(self, data_dir, num_channels=19, num_examples=-1, batch_size=64, length=1000):
		self.batch_size = batch_size
		self.length = length
		# self.examples_signal, self.examples_atribute = load_eeg_directory(data_dir, num_channels, min_length=100, max_length=999999, max_num=num_examples, length=length)
		self.filenames = get_filenames(data_dir, num_channels, length, min_length=100, max_length=999999, max_num=num_examples)
		self.batched_filenames = split_into_batches(self.filenames, batch_size)
		print("Number of files found:", len(self.filenames), "Length:", length)
		# self.batched_examples_atribute = split_into_batches(self.examples_atribute, batch_size)
		# self.batched_examples_signal = split_into_batches(self.examples_signal, batch_size)

	def __len__(self):
		# return len(self.batched_examples_atribute)
		return len(self.batched_filenames)

	def __getitem__(self, index):
		#old bad method
		# batch = self.batched_examples_signal[index]
		# sample = torch.from_numpy(np.asarray(batch))
		# sample = sample.view(-1, sample.shape[2], sample.shape[1]).type('torch.FloatTensor')

		#new version (doesn't kill vRAM)
		batch_filenames = self.batched_filenames[index]
		signals, attributes = read_filenames(batch_filenames, self.length)
		sample = torch.from_numpy(np.asarray(signals))
		sample = sample.view(-1, sample.shape[2], sample.shape[1]).type('torch.FloatTensor')

		return sample

	def shuffle(self):
		#old bad method
		# examples = list(zip(self.examples_signal, self.examples_atribute))
		# random.shuffle(examples)
		# self.examples_signal, self.examples_atribute = zip(*examples)
		# self.batched_examples_signal = split_into_batches(self.examples_signal, self.batch_size)
		# self.batched_examples_atribute = split_into_batches(self.examples_atribute, self.batch_size)

		#new better method
		random.shuffle(self.filenames)
		self.batched_filenames = split_into_batches(self.filenames, self.batch_size)


def load_eeg_file(filename):
	hdf = h5py.File(filename, "r")
	atributes = hdf["patient"].attrs
	rec = hdf["record-0"]
	signals = rec["signals"]
	atributes = parse_atributes(atributes)
	specs = {
		"sample_frequency" : rec.attrs["sample_frequency"],
		"number_channels" : rec.attrs["number_channels"]
	}
	return signals, atributes, specs

def parse_atributes(atributes):
	gender = atributes["gender"]
	if gender == "Male":
		gender = 1
	elif gender == "Female":
		gender = -1
	else:
		gender = 0
	gestatational_age_at_birth_days = float(atributes["gestatational_age_at_birth_days"])
	birthdate = atributes["birthdate"]
	age_in_seconds = (time.time() - time.mktime(datetime.datetime.strptime(birthdate, "%Y-%m-%d").timetuple()))
	# print("gender", gender, type(gender))
	# print("gestatational_age_at_birth_days", gestatational_age_at_birth_days, type(gestatational_age_at_birth_days))
	# print("age_in_seconds", age_in_seconds, type(age_in_seconds))
	out = np.asarray([gender, age_in_seconds, gestatational_age_at_birth_days])
	return out
	#todo, figure out born premature
	# born_premature = atributes["born_premature"]


def load_eeg_directory(path, num_channels, min_length=0, max_length=1e9999, max_num=-1, sample_frequency=200, length=1000):
	files = os.listdir(path)
	num_files_read = 0
	examples_signal = []
	examples_atribute = []
	for file in files:
		if (file.split(".")[-1] != "eeghdf") and (file.split(".")[-1] != "fif"):
			continue

		if file.split(".")[-1] == "eeghdf":
			signals, atributes, specs = load_eeg_file(path + file)
			# print(signals.shape[1])
			if signals.shape[1] < length + 5000:
				continue
			# start = signals.shape[1] / 2 
			# stop =
			signals = signals[:, 5000:length+5000]
			if (int(specs["number_channels"]) != num_channels):
				# print("Not correct num_channels", num_channels, specs["number_channels"])
				continue
			if (int(specs["sample_frequency"]) != sample_frequency):
				# print("Not correct sample_frequency", sample_frequency, specs["sample_frequency"])
				continue
			num_readings = signals.shape[1]
			# if num_readings < min_length:
			# 	continue
			# elif num_readings > max_length:
			# 	continue 
		
		else: #it's .fif
			signals = mne.io.read_raw_fif(path + file).to_data_frame().values
			atributes= [None]

		examples_signal += [signals]
		examples_atribute += [atributes]
		
		if num_files_read-1 == max_num:
			return examples_signal, examples_atribute
		num_files_read += 1

	return examples_signal, examples_atribute

def split_into_batches(x, examples_per_batch):
	final = []
	for start in range(0, len(x), examples_per_batch):
		end = start + examples_per_batch
		final += [x[start:end]]
	# print("final", np.shape(final))
	return final

def read_filenames(filenames, length):
	examples_signal = []
	examples_atribute = []
	for file in filenames:
		signals, atributes, specs = load_eeg_file(file)
		signals = signals[:, :length]
		examples_signal += [signals]
		examples_atribute += [atributes]
		
	return examples_signal, examples_atribute

def get_filenames(path, num_channels, length, min_length=0, max_length=1e9999, max_num=-1, sample_frequency=200):
	files = os.listdir(path)
	num_files_read = 0
	filenames = []
	for file in files:
		if (file.split(".")[-1] != "eeghdf"):# and (file.split(".")[-1] != "fif"):
			continue

		if file.split(".")[-1] == "eeghdf":
			signals, _, specs = load_eeg_file(path + file)
			if signals.shape[1] < length:
				continue
			if (int(specs["number_channels"]) != num_channels):
				# print("Not correct num_channels", num_channels, specs["number_channels"])
				continue
			if (int(specs["sample_frequency"]) != sample_frequency):
				# print("Not correct sample_frequency", sample_frequency, specs["sample_frequency"])
				continue
			num_readings = signals.shape[1]

		filenames += [path + file]
		
		if num_files_read == max_num:
			return filenames
		num_files_read += 1

	return filenames	

# dataset = Dataset("./eeg-hdfstorage/data/")
if __name__ == "__main__":
	dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=-1, num_channels=44, length=100000)
	dataset.shuffle()
	dataset[0]
	# dataset = EEGDataset("/Users/DanielLongo/server/mnt/home2/dlongo/eegML/generated_eegs/from_forward_model/", num_examples=10, num_channels=44)
# filename = "./eeg-hdfstorage/data/absence_epilepsy.eeghdf"
# # signals, atributes = load_eeg_file(filename)
# print("atributes", atributes.shape)
# signals, atributes = load_eeg_directory("./eeg-hdfstorage/data/", 19)
# print(np.shape(signals))
# print(np.shape(atributes))

# print(signals.shape)
# print(list(atributes.items()))
# print(atributes["patient_name"])
