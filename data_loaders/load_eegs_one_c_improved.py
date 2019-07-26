import math
import mne
import sys
sys.path.append("../GANs/")
import time
import datetime
import numpy as np
import os
import torch
from torch.utils import data
import h5py
import random
import sklearn
from sklearn import preprocessing
import pandas as pd
from utils import save_EEG

class EEGDataset(data.Dataset):
	def __init__(self, data_dir, csv_file=None, num_channels=19, num_examples=-1, batch_size=64, length=1000, delay=10000):
		# num_examples -= 1 #for stagering
		self.delay = delay
		self.batch_size = batch_size
		self.length = length
		self.valid_unitl = 20 #channel until which to take recordings from
		self.recordings_per_batch = math.ceil(batch_size / self.valid_unitl)
		self.num_examples = num_examples
		# self.examples_signal, self.examples_atribute = load_eeg_directory(data_dir, num_channels, min_length=100, max_length=999999, max_num=num_examples, length=length)
		print(num_examples)
		num_files_needed = math.ceil(num_examples / self.valid_unitl)
		print("num_files_needed", num_files_needed)
		if csv_file == None:
			self.filenames = get_filenames(data_dir, num_channels, length, min_length=100, max_length=999999, max_num=num_files_needed, delay=self.delay)
		else:
			self.filenames = load_filenames_from_csv(csv_file)
			random.shuffle(self.filenames)
			self.filenames = check_files(self.filenames, num_channels, length, min_length=100, max_length=999999, max_num=num_files_needed, delay=self.delay)
		self.batched_filenames = split_into_batches(self.filenames, self.recordings_per_batch)
		print("Number of files found:", len(self.filenames), "Length:", length)
		self.preloaded_examples = []
		self.preloaded_batches = []
		self.load_data()
		self.create_batches()
		# self.batched_examples_atribute = split_into_batches(self.examples_atribute, batch_size)
		# self.batched_examples_signal = split_into_batches(self.examples_signal, batch_size)

	def create_batches(self):
		self.preloaded_batches = split_into_batches(self.preloaded_examples, self.batch_size)

	def load_data(self):
		examples = []
		for filename in self.filenames:
			examples += self.load_examples_from_file(filename)
		random.shuffle(examples)
		examples = examples[:self.num_examples]
		self.preloaded_examples = examples
			# self.load_file_data()
		# batched_data = []
		# print(len(self.batched_filenames), len(self.batched_filenames[0]))
		# for filenames in self.batched_filenames:
		# 	batched_data += [self.load_file_data(filenames)]
		# self.preloaded_data = batched_data

	def load_examples_from_file(self, filename):
		signals, attributes = read_filenames([filename], self.length, delay=self.delay)
		signals = np.asarray(signals)
		signals = signals[:, :self.valid_unitl, :] # remove poor channels
		single_channels = []
		for example in signals:
			for recording in example:
				single_channels += [recording]
		return single_channels

	def __len__(self):
		# return len(self.batched_examples_atribute)
		return len(self.batched_filenames)

	def __getitem__(self, index):
		if self.preloaded_batches[index] is None:
			# TODO: get better soltion
			print("preloaded batches is None")
			return self.__getitem__(index + 1)
		out = torch.FloatTensor(self.preloaded_batches[index])
		# print("out", out.shape)
		out = out.view(-1, self.length, 1)
		return out

	# def load_file_data(self, batch_filenames): # loads into batches
	# 	print("batch filenames", batch_filenames)
	# 	#old bad method
	# 	signals, attributes = read_filenames(batch_filenames, self.length, delay=self.delay)
	# 	signals = np.asarray(signals)
	# 	signals = signals[:, :self.valid_unitl, :] # remove poor channels
	# 	single_channels = []
	# 	for example in signals:
	# 		for recording in example:
	# 			single_channels += [recording]
	# 	sample = split_into_batches(single_channels, self.batch_size)
	# 	try:
	# 		out = torch.from_numpy(np.asarray(sample[0]).reshape(self.batch_size, self.length, 1)).type('torch.FloatTensor')
	# 	except ValueError:
	# 		print("shape error in singe channel data loader")
	# 		return None
	# 	return out

	def shuffle(self):
		random.shuffle(self.preloaded_examples)
		self.create_batches()


def load_eeg_file(filename, normalize=False):
	hdf = h5py.File(filename, "r")
	atributes = hdf["patient"].attrs
	rec = hdf["record-0"]
	signals = rec["signals"]
	atributes = parse_atributes(atributes)
	specs = {
		"sample_frequency" : rec.attrs["sample_frequency"],
		"number_channels" : rec.attrs["number_channels"]
	}
	if normalize:
		# print(type(signals.value))
		signals = sklearn.preprocessing.normalize((signals.value).T, axis=1).T
		# print(signals.shape)
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

def read_filenames(filenames, length, delay=10000):
	examples_signal = []
	examples_atribute = []
	# delay = 100000
	for file in filenames:
		# print("file", file)
		signals, atributes, specs = load_eeg_file(file, normalize=False)
		# print("delay", delay)
		# print("before", signals.shape)
		signals_staggered = signals[:, delay:length+delay]
		# print("after", signals.shape)
		examples_signal += [signals_staggered]
		examples_atribute += [atributes]
		
	return examples_signal, examples_atribute

def get_filenames(path, num_channels, length, min_length=0, max_length=1e9999, max_num=-1, sample_frequency=200, delay=10000):
	files = os.listdir(path)
	num_files_read = 0
	filenames = []
	for file in files:
		if (file.split(".")[-1] != "eeghdf"):# and (file.split(".")[-1] != "fif"):
			continue

		if file.split(".")[-1] == "eeghdf":
			signals, _, specs = load_eeg_file(path + file, normalize=False)

			if signals.shape[1] < length + delay:
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

def check_files(filenames, num_channels, length, min_length=0, max_length=1e9999, max_num=-1, sample_frequency=200, delay=10000):	
	num_files_checked = 0
	checked_filenames = []
	for file in filenames:
		if (file.split(".")[-1] != "eeghdf"):# and (file.split(".")[-1] != "fif"):
			continue

		if file.split(".")[-1] == "eeghdf":
			signals, _, specs = load_eeg_file(file, normalize=False)

			if signals.shape[1] <= length + delay:
				continue
			if (int(specs["number_channels"]) != num_channels):
				# print("Not correct num_channels", num_channels, specs["number_channels"])
				continue
			if (int(specs["sample_frequency"]) != sample_frequency):
				# print("Not correct sample_frequency", sample_frequency, specs["sample_frequency"])
				continue
			num_readings = signals.shape[1]

		checked_filenames += [file]
		
		if num_files_checked == max_num:
			return checked_filenames
		num_files_checked += 1

	return checked_filenames
def load_filenames_from_csv(csv_filename, filepath="/mnt/data1/eegdbs/SEC-0.1/", max_num=-1):
	df = pd.read_csv(csv_filename)
	filenames = []
	file_keys = df["nk_file_reportkey"]
	hopsital_types = df["database_source"]
	note_types = df["note_type"]
	for i in range(len(file_keys)):
		if note_types[i] != "spot":
			continue
		if hopsital_types[i] == "STANFORD_NK":
			filenames += [filepath + "stanford/" + file_keys[i] + "_1-1+.eeghdf"]
		if hopsital_types[i] == "LPCH_NK":
			filenames += [filepath + "lpch/" + file_keys[i] + "_1-1+.eeghdf"]
		if (len(filenames) == max_num):
			print("Number of filenames loaded from csv", len(filenames))
			return filenames
	print("Number of filenames loaded from csv", len(filenames))
	return filenames

if __name__ == "__main__":
	# csv_file = "/mnt/data1/eegdbs/all_reports_impress_blanked-2019-02-23.csv"
	# dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", csv_file=csv_file, num_examples=200, num_channels=44, length=1004)
	# dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/",  num_examples=438, num_channels=44, length=768)
	dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/",  num_examples=438, batch_size=64, num_channels=44, length=768)
	print("shuffling")
	dataset.shuffle()
	print("shuffling finsihed")
	print("loading")
	print("sample shape", dataset[0].shape)
	print("loading finsihed")
	# csv_file = "/Users/DanielLongo/server/mnt/data1/eegdbs/all_reports_impress_blanked-2019-02-23.csv"
	# dataset = EEGDataset("./eeg-hdfstorage", csv_file=csv_file, num_examples=64, num_channels=44, length=1004)
	# print("finna'l",dataset[0].shape)
	# save_EEG(dataset[0], None, None, "save_dataloader")
	# dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=20, num_channels=44, length=100000)
	# dataset.shuffle()
	# dataset[0]