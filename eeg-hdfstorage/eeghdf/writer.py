# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

# goal: support python 2.7 -> python 3.5+
import future
from future.utils import iteritems
from builtins import range

import h5py
import numpy as np

"""
------
EEGHDF
------
EEGHDF is a format for hdf5 files to encapsulate EEG files and other, similar physiological files in a way that is easy to access in the machine learning world

by convention I have been using the ending <filename>.eeg.hdf5 or <basename>.eeghdf

Sources of motivation::
-----------------------
- European Data Format for EEG  (EDF+, BDF+) http://www.edfplus.info
- neo package for cross format physiological data

To start with, I have decide to initally stick fairly close to the EDF spec and
see how things evolve. But I will see that converters back to EDF, and over to
neo as well as MNE's format FIF are priorities.

adding information such as electrode geometries (cf. FIF) is considered for the future

Because all my EEGs appear to be uniformly sampled -- all channels sampled at the same frequency. My recording blocks will assume that they are all the same. Note that edf allows for different sampling rates.

If this occurs, I will raise an error --- maybe a simple way in the future to deal with this would be to put the different sampled rates in a separate Record block.

Each Record block is this a uniformly sampled set of datapoints with the same sampling frequency. The eeg sample data is stored in HDF5 dataset which is essentially a rectangular array.

For the future, I will be watching NWB - neurodata without borders - this specifies an hdf5 schema for 
neurophysiology data which is extensible

One big question going forward is how to store strings -
- option 1: everything is ascii, straighforward but ignores a lot of other possible names
- option 2: everything is unicode/utf-8 which looks like ascii until you need the extra characters
in python 2.7 these are being stored as str/bytes which means essentially ascii I think
based upon how I undertand h5py/hdf works but should double check

the variable length arrays are bytes/python2-strings

version 2: add studyadmincode to record-0 
version 1: initial version used create SEC 0.1
"""

# Notes on neo.io
# Neo's data structures are designed to hold a wide variety of
# electrophysiological data

#     - block list
#       - Block: contain to hold data, .name, .description, .file_origin, .file_datetime, .rec_datetime
#         - Container of: Segment, RecordingChannelGroup
#         - Segment: container for data holding a common time basis
#             - contains: Epoch, epochArray, Event, EventArray, AnaologSignal, AnalogSignalArray, IrregularSampledSignal, Spike, SpikeTrain
#             - AnalogSignalArray


# enumerate all the attribute data which can be read/writen
# just to make it clear
class EEGHDFWriter(object):
    version = 2  # first official release with 1000, maintain compat up to 2000

    def __init__(self, hdf_file_name, fileattr='w-'):
        """

        fileattr = 'w' will truncate the file if it exists
        fileattr = 'w-' will fail if the hdf file already exists
        r+	Read/write, file must exist
        w	Create file, truncate if exists
        w- or x	Create file, fail if exists
        a	Read/write if exists, create otherwise (default)

        """

        self.hdf_file_name = hdf_file_name
        self.hdf = h5py.File(hdf_file_name, fileattr)
        self.hdf.attrs['EEGHDFversion'] = self.version
        self.record_list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.hdf.close()
        return None  # True will suppress the exception and continue excecution

    def write_patient_info(self, patient_name='',
                           patientcode='',
                           gender='',
                           # suggest F,M,U for unknown, I for
                           # indeterminate/intermediate
                           birthdate_isostring='',
                           # store as standard iso date/time YYYY-MM-DD
                           gestational_age_at_birth_days=-1.0,
                           # -1.0 stands for unknown
                           born_premature='unknown',  # true,false, unknown
                           patient_additional=''):  # to support edf

        patient = self.hdf.create_group('patient')  # subject

        self.patient = patient
        patient.attrs['patient_name'] = patient_name
        patient.attrs['patientcode'] = patientcode
        patient.attrs['gender'] = gender
        # bytes(birthdate,encoding='ascii')
        patient.attrs['birthdate'] = birthdate_isostring
        patient.attrs['patient_additional'] = patient_additional
        # remove this because it is not intrinsic to patient and will change
        #patient.attrs['age_days'] = age_days  # float value for age in days
        ## attributes not in EDF
        patient.attrs[
            'gestatational_age_at_birth_days'] = gestational_age_at_birth_days
        # versus giving weeks/days with seconds do not need to worry about calendar issue
        # is 36.5 weeks 36 weeks + 3.5 days? or 36 weeks and 5 days
        # maybe days would be better 280 days is 40 weeks

        # ('unknown', True, False)
        patient.attrs['born_premature'] = born_premature

        # somewhat interesting question the patient's age at the time of a recording
        # may change between Records so should
        # attributes like: age_seconds be attached to the block record? I guess
        # so # long seconds or float

    def create_record_block(self,
                            record_duration_seconds,
                            start_isodatetime,
                            end_isodatetime,
                            number_channels,
                            num_samples_per_channel,
                            sample_frequency,
                            signal_labels,
                            signal_physical_mins,
                            signal_physical_maxs,
                            signal_digital_mins,
                            signal_digital_maxs,
                            physical_dimensions,
                            patient_age_days,  # auto calculate float form patienet info?
                            signal_prefilters=None,
                            signal_transducers=None,
                            bits_per_sample='auto',
                            technician='',
                            studyadmincode=''):
        """
        create the record block along with all the small structures
        that are reasonable to hold in memory all at once
        with the presumption that the actual waveform data may be large
        and so will be written via iteration
        """
        # now start storing the lists of things: labels, units...
        # number_channels = len(label_list)
        # 1. keep the text-vs-bytes distinction clear
        # 2. alays use "bytes" instead of "str" when you're sure you want a byte string.
        # for literals, can use "b" prefix, e.g. b'some bytes'
        # 3. for text strings use str or btter yet unicode, u'Hello'
        # 4. always use UTF-8 in code
        record = self.get_new_record_block()

        record.attrs['start_isodatetime'] = start_isodatetime
        record.attrs['end_isodatetime'] = end_isodatetime

        record.attrs['number_channels'] = number_channels
        record.attrs['number_samples_per_channel'] = num_samples_per_channel
        # uniform assumption
        record.attrs['sample_frequency'] = sample_frequency

        record['signal_physical_mins'] = signal_physical_mins
        record['signal_physical_maxs'] = signal_physical_maxs
        record['signal_digital_mins'] = signal_digital_mins
        record['signal_digital_maxs'] = signal_digital_maxs

        # now create the variable length string (bytes) arrays
        # variable bytes/ascii string (or b'' type)
        str_dt = h5py.special_dtype(vlen=bytes)

        label_ds = record.create_dataset(
            'signal_labels', (number_channels,), dtype=str_dt)
        label_ds[:] = signal_labels

        units_ds = record.create_dataset(
            'physical_dimensions', (number_channels,), dtype=str_dt)
        units_ds[:] = physical_dimensions

        transducer_ds = record.create_dataset(
            'transducers', (number_channels,), dtype=str_dt)
        transducer_ds[:] = signal_transducers

        prefilter_ds = record.create_dataset(
            'prefilters', (number_channels,), dtype=str_dt)
        prefilter_ds[:] = signal_prefilters

        if bits_per_sample == 'auto':
            if all(signal_digital_maxs < 2 **
                   15) and all(signal_digital_mins >= -2 ** 15):
                bits_per_sample = 16  # EDF
            elif all(signal_digital_maxs < 2 ** 23) and all(signal_digital_mins >= -2 ** 23):
                bits_per_sample = 24  # BDF 2^23 = 8388608 + 1 bit for sign
        record.attrs['bits_per_sample'] = bits_per_sample

        if bits_per_sample <= 16:
            eegdata = record.create_dataset('signals', (number_channels, num_samples_per_channel),
                                            dtype='int16',
                                            # if wanted 1 second chunks
                                            # chunks=(number_channels,sample_frequency),
                                            chunks=True,
                                            fletcher32=True,
                                            compression='gzip'  # most universal
                                            # maxshape=(256,None)
                                            )
        # handles up to 32
        elif bits_per_sample <= 32 and bits_per_sample > 16:
            eegdata = record.create_dataset('signals', (number_channels, num_samples_per_channel),
                                            dtype='int32',
                                            # chunks=(number_channels,sample_frequency),
                                            # # if wanted 1 second chunks
                                            chunks=True,
                                            fletcher32=True,
                                            compression='gzip'  # most universal
                                            # maxshape=(256,None)
                                            )
        record.attrs['patient_age_days'] = patient_age_days

        # if patient_age_seconds != 'auto':
        #     record.attrs['patient_age_seconds'] = patient_age_seconds
        # else:
        #     record.att

        record.attrs['technician'] = technician
        record.attrs['studyadmincode'] = studyadmincode
        return record 

    def create_masked_record_block(self, 
                                   record_duration_seconds,
                                   start_isodatetime,
                                   end_isodatetime,  # remove number_channles as this is calculated 
                                   num_samples_per_channel,
                                   sample_frequency,
                                   signal_labels,
                                   signal_physical_mins,
                                   signal_physical_maxs,
                                   signal_digital_mins,
                                   signal_digital_maxs,
                                   physical_dimensions,
                                   patient_age_days,  # auto calculate float form patienet info?
                                   signal_prefilters=None,
                                   signal_transducers=None,
                                   bits_per_sample='auto',
                                   technician='',
                                   studyadmincode='',
                                   signals_mask=None):
        
        """
        @signals_mask is a list or array which evalutes to True or False depending on if a signal should be inlcuded in the 
        eeghdf file 
        """

        orig_num_signals = len(signal_labels) # original number of signals 
 

        if signals_mask:
            # maybe should do this with arr_mask = arr[np.where(signals_mask) pattern
            
            signal_labels_masked = [ signal_labels[ii] for ii in range(len(signal_labels))
                                     if signals_mask[ii] ]

            Nmask = len(signal_labels_masked)
            number_channels_masked = Nmask
            
            signal_physical_mins_masked = np.array([ signal_physical_mins[ii] 
                                            for ii in range(len(signal_physical_mins))
                                            if signals_mask[ii] ])

            signal_physical_maxs_masked = np.array([ signal_physical_maxs[ii]
                                            for ii in range(len(signal_physical_maxs))
                                            if signals_mask[ii] ])
            
            signal_digital_mins_masked = np.array([ signal_digital_mins[ii] 
                                           for ii in range(len(signal_digital_mins)) 
                                           if signals_mask[ii] ])
            # signal_digital_maxs_masked = [ signal_digital_maxs[ii]
            #                                for ii in range(len(signal_digital_maxs))
            #                                if signals_mask[ii] ]

            signal_digital_maxs_masked =  signal_digital_maxs[np.where(signals_mask)]

            
            physical_dimensions_masked = [ physical_dimensions[ii] 
                                           for ii in range(len(physical_dimensions))
                                           if signals_mask[ii] ]

            if signal_prefilters:
                signal_prefilters_masked = [ signal_prefilters[ii] 
                                             for ii in range(len(signal_prefilters))
                                             if signals_mask[ii] ]
            else:
                #signal_prefilters_masked = [''] * Nmask
                signal_prefilters_masked = None
                
            if signal_transducers:
                signal_transducers_masked = [ signal_transducers[ii] 
                                              for ii in range(len(signal_transducers)) 
                                              if signals_mask[ii] ]
            else:
                signal_transducers_masked = None

            rec = self.create_record_block(record_duration_seconds=record_duration_seconds,
                                           start_isodatetime=start_isodatetime,
                                           end_isodatetime=end_isodatetime,
                                           number_channels=number_channels_masked, 
                                           num_samples_per_channel=num_samples_per_channel,
                                           sample_frequency=sample_frequency,
                                           signal_labels=signal_labels_masked,
                                           signal_physical_mins=signal_physical_mins_masked,
                                           signal_physical_maxs=signal_physical_maxs_masked,
                                           signal_digital_mins=signal_digital_mins_masked,
                                           signal_digital_maxs=signal_digital_maxs_masked,
                                           physical_dimensions=physical_dimensions_masked,
                                           patient_age_days=patient_age_days,
                                           signal_prefilters=signal_prefilters_masked,
                                           signal_transducers=signal_transducers_masked,
                                           technician=technician,
                                           studyadmincode=studyadmincode)
            return rec
        else: # no mask provided
            rec = self.create_record_block(record_duration_seconds=record_duration_seconds,
                                           start_isodatetime=start_isodatetime,
                                           end_isodatetime=end_isodatetime,
                                           number_channels=orig_num_signals,
                                           num_samples_per_channel=num_samples_per_channel,
                                           sample_frequency=sample_frequency,
                                           signal_labels=signal_labels,
                                           signal_physical_mins=signal_physical_min,
                                           signal_physical_maxs=signal_physical_maxs,
                                           signal_digital_mins=signal_digital_mins,
                                           signal_digital_maxs=signal_digital_maxs,
                                           physical_dimensions=physical_dimensions,
                                           patient_age_days=patient_age_days,
                                           signal_prefilters=signal_prefilters,
                                           signal_transducers=signal_transducers,
                                           technician=technician,
                                           studyadmincode=studyadmincode)
            return rec 
            



    def get_new_record_block(self):
        """create a standard name for a record block group"""
        nextnum = len(self.record_list)
        group_name = 'record-%d' % nextnum
        record = self.hdf.create_group(group_name)
        self.record_list.append(record)
        return record

    def get_current_record(self):
        if len(self.record_list):
            return self.record_list[-1]
        else:
            return None

    def write_annotations_b(self, annotations_list, record=None):
        """
        write_annotations(self, annotations_list, record)
        - write a list of annotations to a group in a @record, or the current record if none is given

        @annotations_list is a list with items of form
        [(long int starts_100ns, char[16] durations, bytes text), ...]

        the text is already supposed to be encoded in UTF-8
        durations bytes should be ascii encoded +/-float (e.g. +5.000 or -3.00)
        """
        if not record:
            record = self.get_current_record()
        edf_annots = record.create_group('edf_annotations')
        num_annot = len(annotations_list)

        starts = edf_annots.create_dataset(
            'starts_100ns', (num_annot,), dtype=np.int64)

        # curiously these durations seem to be stored as strings but of floating
        # point values "5.00000" for 5 second duration

        durations = edf_annots.create_dataset(
            'durations_char16',
            (num_annot,
             ),
            dtype='S16')  # S16 !!! check py3 compatibility

        # variable ascii string (or b'' type)
        str_dt = h5py.special_dtype(vlen=bytes)
        texts = edf_annots.create_dataset('texts', (num_annot,), dtype=str_dt)

        # start with a loop
        for ii in range(num_annot):
            starts[ii] = annotations_list[ii][0]

            # note: so far I have ony seen type(annotations_list[ii][1] -> <type 'str'> and they look like ascii strings
            # of floating point number of seconds for a duration
            # print('type(annotations_list[ii][1]):', type(annotations_list[ii][1]))

            durations[ii] = annotations_list[ii][1]
            texts[ii] = annotations_list[ii][2].strip()

        return edf_annots

    def stream_dig_signal_to_record_block(self, record_block, block_iterator):
        """
        used to write the array of data to the hdf file
        this is a rectangular array shape=(number_channels, sample_per_chunk)

        to use it you need to provide an iterator which will yield a filled buffer
        of int32 or int16 numpy array with shape (number_channels, sample_per_chunk)
        that can be serialized into the hdf dataset

        Note, the left over samples: total_samples - nchunks * samples_per_chunk
        need to be written at the end.
        """
        record = record_block  # hdf group
        signals = record['signals']

        # mark = 0
        # simpliest interator
        for buf, mark, num in block_iterator:
            signals[:, mark:mark + num] = buf
            # mark += samples_per_chunk

        return signals


def test_EEGHDF_creation():
    TEST_EEGHDF_FILENAME = 'test.eeg.hdf5'
    hf = EEGHDFWriter(hdf_file_name=TEST_EEGHDF_FILENAME, fileattr='w')
    print('EEGHDF version:', hf.hdf.attrs['EEGHDFversion'])

    assert hf.hdf.attrs['EEGHDFversion'] == 1


def test_EEGHDF_patient_creation():
    TEST_EEGHDF_FILENAME = 'test.eeg.hdf5'
    hf = EEGHDFWriter(hdf_file_name=TEST_EEGHDF_FILENAME, fileattr='w')
    print('EEGHDF version:', hf.hdf.attrs['EEGHDFversion'])

    hf.write_patient_info(patientname='Smith, Jill',
                          patientcode='TEST',
                          gender='female',
                          birthdate_isostring='2005-11-23',
                          gestational_age_at_birth_days=280,  # full term
                          born_premature='false',
                          patient_additional='')

    # print(list(hf.patient.attrs.items()))
    hf.hdf.close()
    del hf
    comparison = [(u'patientname', 'Smith, Jill'),
                  (u'gender', 'female'), (u'birthdate', '2005-11-23'),
                  (u'patient_additional',
                   ''), (u'gestatational_age_at_birth_days', 280),
                  (u'born_premature', 'false')]

    nhf = h5py.File(TEST_EEGHDF_FILENAME, 'r+')
    patient = nhf['patient']
    # compat:py2.7 list vs py3+ iterator
    readitems = list(patient.attrs.items())

    assert comparison == readitems


def test_stream_dig_signal_to_record_block():
    nchan = 3
    nsamples = 10000
    samples_per_chunk = 512
    fs = float(samples_per_chunk)
    nchunks = nsamples // samples_per_chunk
    left_over = nsamples - nchunks * samples_per_chunk
    print('left over samples:', left_over)

    t = np.arange(nsamples)
    t = t / fs
    arr = np.zeros((nchan, nsamples), dtype='int16')

    # 10 Hz sine wave +/- 10000
    arr[0, :] = 10000 * np.sin(2 * np.pi * 10.0 * t)
    # 20 Hz sine wave +/- 10000
    arr[1, :] = 10000 * np.sin(2 * np.pi * 20.0 * t)
    arr[2, :] = np.arange(nsamples, dtype='int16')  # ramp

    def fancy_block_generator(nchan, nsamples, samples_per_chunk):
        """
        yields buffer (int16), mark (in the buffer where starts), num (samples to write)
        """
        nchunks = nsamples // samples_per_chunk
        print('nchunks:', nchunks)
        left_over = nsamples - nchunks * samples_per_chunk

        mark = 0
        for ii in range(nchunks):
            yield (arr[:, mark:mark + samples_per_chunk], mark, samples_per_chunk)
            mark += samples_per_chunk
            print('mark:', mark)

        # the left over
        if left_over > 0:
            yield (arr[:, mark:mark + left_over], mark, left_over)

    TEST_EEGHDF_FILENAME = 'test.eeg.hdf5'
    with EEGHDFWriter(hdf_file_name=TEST_EEGHDF_FILENAME, fileattr='w') as hf:

        # get a fake record block with no other info
        record = hf.get_new_record_block()
        record.create_dataset(
            'signals',
            (nchan,
             nsamples),
            dtype='int16',
            chunks=True,
            fletcher32=True)

        genitr = fancy_block_generator(nchan, nsamples, samples_per_chunk)
        hf.stream_dig_signal_to_record_block(
            record,
            nchan,
            samples_per_chunk,
            genitr)

    # now test
    with h5py.File(TEST_EEGHDF_FILENAME, 'r') as hf:
        signals = hf['record-0']['signals']
        assert np.all(signals[0, :] == arr[0, :])
        assert np.all(signals[1, :] == arr[1, :])
        assert np.all(signals[2, :] == arr[2, :])

    return

# references of creation __new__, __init__, __del__,
# stack overflow http://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
# http://www.andy-pearce.com/blog/posts/2013/Apr/python-destructor-drawbacks/
# https://docs.python.org/2/library/contextlib.html#module-contextlib
# http://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python
# @contextlib.contextmanager
# def packageResource():
#     class Package:
#         ...
#     package = Package()
#     yield package
#     package.cleanup()


# class Package(object):
#     def __new__(cls, *args, **kwargs):
#         @contextlib.contextmanager
#         def packageResource():
#             # adapt arguments if superclass takes some!
#             package = super(Package, cls).__new__(cls)
#             package.__init__(*args, **kwargs)
#             yield package
#             package.cleanup()

#     def __init__(self, *args, **kwargs):
#         ...

# or shorter version using closing decorator if name clean up function close()

# class Package(object):
#     def __new__(cls, *args, **kwargs):
#         package = super(Package, cls).__new__(cls)
#         package.__init__(*args, **kwargs)
#         return contextlib.closing(package)

# inheritance
# class SubPackage(Package):
#     def close(self):
#         pass


# scan gc and close all objects
# source: http://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
# import gc
# for obj in gc.get_objects():   # Browse through ALL objects
#     if isinstance(obj, h5py.File):   # Just HDF5 files
#         try:
#             obj.close()
#         except:
#             pass # Was already closed
