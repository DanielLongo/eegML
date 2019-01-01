import h5py
import time
import datetime
import numpy as np
import os

def load_eeg_file(filename):
	hdf = h5py.File(filename, "r")
	atributes = hdf["patient"].attrs
	rec = hdf["record-0"]
	signals = rec["signals"]
	atributes = parse_atributes(atributes)
	return signals, atributes

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


def load_eeg_directory(path, min_length=0, max_length=1e9999, max_num=-1):
	files = os.listdir(path)
	num_files_read = 0
	examples_signal = []
	examples_atribute = []
	print("files", files)
	for file in files:
		if file.split(".")[-1] != "eeghdf":
			continue

		signals, atributes = load_eeg_file(path + file)
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
	for start in range(0, x.shape[0], examples_per_batch):
		end = start + examples_per_batch
		final += x[start:end]
	return final


filename = "./eeg-hdfstorage/data/absence_epilepsy.eeghdf"
# signals, atributes = load_eeg_file(filename)
# print("atributes", atributes.shape)
signals, atributes = load_eeg_directory("./eeg-hdfstorage/data/")
print(np.shape(signals))
print(np.shape(atributes))

# print(signals.shape)
# print(list(atributes.items()))
# print(atributes["patient_name"])