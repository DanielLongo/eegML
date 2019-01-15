import h5py
import time
import datetime
import numpy as np
import os
import torch
from torch.utils import data
import h5py

class Dataset(data.Dataset):
	def __init__(self, data_dir, num_channels=19, num_examples=-1, batch_size=64):
		self.examples_signal, self.examples_atribute = load_eeg_directory(data_dir, num_channels, min_length=100, max_length=999999, max_num=num_examples)
		self.examples_atribute = split_into_batches(self.examples_atribute, batch_size)
		self.examples_signal = split_into_batches(self.examples_signal, batch_size)

	def __len__(self):
		return len(self.examples_signal)

	def __getitem__(self, index):
		# Select sample
		return examples_signal[index], examples_atribute[index]

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


def load_eeg_directory(path, num_channels, min_length=0, max_length=1e9999, max_num=-1, sample_frequency=200):
	files = os.listdir(path)
	num_files_read = 0
	examples_signal = []
	examples_atribute = []
	print("files", files)
	for file in files:
		if file.split(".")[-1] != "eeghdf":
			continue

		signals, atributes, specs = load_eeg_file(path + file)
		if (int(specs["number_channels"]) != num_channels):
			print("Not correct num_channels", num_channels, specs["number_channels"])
			continue
		if (int(specs["sample_frequency"]) != sample_frequency):
			print("Not correct sample_frequency", sample_frequency, specs["sample_frequency"])
			continue
		num_readings = signals.shape[1]
		if num_readings < min_length:
			continue
		elif num_readings > max_length:
			continue 
		examples_signal += [signals]
		examples_atribute += [atributes]
		
		if num_files_read == max_num:
			return examples_signal, examples_atribute

	return examples_signal, examples_atribute

def split_into_batches(x, examples_per_batch):
	final = []
	for start in range(0, len(x), examples_per_batch):
		end = start + examples_per_batch
		final += x[start:end]
	return final


dataset = Dataset("./eeg-hdfstorage/data/")
# filename = "./eeg-hdfstorage/data/absence_epilepsy.eeghdf"
# # signals, atributes = load_eeg_file(filename)
# # print("atributes", atributes.shape)
# signals, atributes = load_eeg_directory("./eeg-hdfstorage/data/", 19)
# print(np.shape(signals))
# print(np.shape(atributes))

# print(signals.shape)
# print(list(atributes.items()))
# print(atributes["patient_name"])