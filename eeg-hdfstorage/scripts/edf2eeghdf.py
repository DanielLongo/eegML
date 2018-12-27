# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function  # py2.6  with_statement

import sys
import pprint
import h5py
import numpy as np
import os.path
# date related stuff
import datetime
import dateutil
import dateutil.tz
import dateutil.parser
import arrow


# compatibility
import future
from future.utils import iteritems
from builtins import range  # range and switch xrange -> range
# from past.builtins import xrange # later, move to from builtins import

import edflib
import eeghdf 



# really need to check the original data type and then save as that datatype along with the necessary conversion factors
# so can convert voltages on own

# try with float32 instead?

# LPCH often uses these labels for electrodes

LPCH_COMMON_1020_LABELS = [
    'Fp1',
    'Fp2',
    'F3',
    'F4',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'F7',
    'F8',
    'T3',
    'T4',
    'T5',
    'T6',
    'Fz',
    'Cz',
    'Pz',
    'E',
    'PG1',
    'PG2',
    'A1',
    'A2',
    'T1',
    'T2',
    'X1',
    'X2',
    'X3',
    'X4',
    'X5',
    'X6',
    'X7',
    'EEG Mark1',
    'EEG Mark2',
    'Events/Markers']

# common 10-20 extended clinical (T1/T2 instead of FT9/FT10)
# will need to specify these as bytes I suppose (or is this ok in utf-8 given the ascii basis)
# keys should be all one case (say upper)
lpch2edf_fixed_len_labels = dict(
    FP1='EEG Fp1         ',
    F7='EEG F7          ',
    T3='EEG T3          ',
    T5='EEG T5          ',
    O1='EEG O1          ',
    F3='EEG F3          ',
    C3='EEG C3          ',
    P3='EEG P3          ',

    FP2='EEG Fp2         ',
    F8='EEG F8          ',
    T4='EEG T4          ',
    T6='EEG T6          ',
    O2='EEG O2          ',

    F4='EEG F4          ',
    C4='EEG C4          ',
    P4='EEG P4          ',

    CZ='EEG Cz          ',
    FZ='EEG Fz          ',
    PZ='EEG Pz          ',
    T1='EEG FT9         ',  # maybe I should map this to FT9/T1
    T2='EEG FT10        ',  # maybe I should map this to FT10/T2
    A1='EEG A1          ',
    A2='EEG A2          ',
    # these are often (?always) EKG at LPCH, note edfspec says use ECG instead
    # of EKG
    X1='ECG X1          ',  # is this invariant? usually referenced to A1
    # this is sometimes ECG but not usually (depends on how squirmy)
    X2='X2              ',
    PG1='EEG Pg1         ',
    PG2='EEG Pg2         ',

    # now the uncommon ones
    NZ='EEG Nz          ',
    FPZ='EEG Fpz         ',

    AF7='EEG AF7         ',
    AF8='EEG AF8         ',
    AF3='EEG AF3         ',
    AFz='EEG AFz         ',
    AF4='EEG AF4         ',

    F9='EEG F9          ',
    # F7
    F5='EEG F5          ',
    # F3 ='EEG F3          ',

    F1='EEG F1          ',
    # Fz
    F2='EEG F2          ',
    # F4
    F6='EEG F6          ',
    # F8
    F10='EEG F10         ',
    FT9='EEG FT9         ',
    FT7='EEG FT7         ',
    FC5='EEG FC5         ',
    FC3='EEG FC3         ',
    FC1='EEG FC1         ',
    FCz='EEG FCz         ',
    FC2='EEG FC2         ',
    FC4='EEG FC4         ',
    FC6='EEG FC6         ',
    FT8='EEG FT8         ',
    FT10='EEG FT10        ',

    T9='EEG T9          ',
    T7='EEG T7          ',
    C5='EEG C5          ',
    # C3 above
    C1='EEG C1          ',
    # Cz above
    C2='EEG C2          ',
    # C4 ='EEG C4          ',
    C6='EEG C6          ',
    T8='EEG T8          ',
    T10='EEG T10         ',

    # A2
    # T3
    # T4
    # T5
    # T6
    TP9='EEG TP9         ',
    TP7='EEG TP7         ',
    CP5='EEG CP5         ',
    CP3='EEG CP3         ',
    CP1='EEG CP1         ',
    CPZ='EEG CPz         ',
    CP2='EEG CP2         ',
    CP4='EEG CP4         ',
    CP6='EEG CP6         ',
    TP8='EEG TP8         ',
    TP10='EEG TP10        ',
    P9='EEG P9          ',
    P7='EEG P7          ',
    P5='EEG P5          ',
    # P3
    P1='EEG P1          ',
    # Pz
    P2='EEG P2          ',
    # P4
    P6='EEG P6          ',
    P8='EEG P8          ',
    P10='EEG P10         ',
    PO7='EEG PO7         ',
    PO3='EEG PO3         ',
    POZ='EEG POz         ',
    PO4='EEG PO4         ',
    PO8='EEG PO8         ',

    # O1
    OZ='EEG Oz          ',
    # O2
    IZ='EEG Iz          ',
)
lpch2edf_fixed_len_labels

# print("lpch2edf_fixed_len_labels::\n")
# pprint.pprint(lpch2edf_fixed_len_labels)

LPCH_TO_STD_LABELS_STRIP = {k: v.strip()
                            for k, v in iteritems(lpch2edf_fixed_len_labels)}

# print('LPCH_TO_STD_LABELS_STRIP::\n')
# pprint.pprint(LPCH_TO_STD_LABELS_STRIP)

LPCH_COMMON_1020_LABELS_to_EDF_STANDARD = {

}


def normalize_lpch_signal_label(label):
    uplabel = label.upper()
    if uplabel in LPCH_TO_STD_LABELS_STRIP:
        return LPCH_TO_STD_LABELS_STRIP[uplabel]
    else:
        return label


def edf2h5_float32(fn, outfn='', hdf_dir='', anonymous=False):
    """
    convert an edf file to hdf5 using a straighforward mapping
    convert to real-valued signals store as float32's

    justing getting started here
    --- metadata ---
    number_signals
    sample_frequency
    nsamples
    age
    signal_labels

    Post Menstrual Age
    """
    if not outfn:
        base = os.path.basename(fn)
        base, ext = os.path.splitext(base)

        base = base + '.eeghdf5'
        outfn = os.path.join(hdf_dir, base)
        print('outfn:', outfn)
        # outfn = fn+'.eeg.h5'
    with edflib.EdfReader(fn) as ef:
        nsigs = ef.signals_in_file
        # again know/assume that this is uniform sampling across signals
        fs = [ef.samplefrequency(ii) for ii in range(nsigs)]
        fs0 = fs[0]

        if any([ fs0 != xx for xx in fs]):
            print("caught multiple sampling frquencies in edf files!!!")
            sys.exit(0)
        
        nsamples0 = ef.samples_in_file(0)

        print('nsigs=%s, fs0=%s, nsamples0=%s' % (nsigs, fs0, nsamples0))

        # create file 'w-' -> fail if exists , w -> truncate if exists
        hdf = h5py.File(outfn, 'w')
        # use compression? yes! give it a try
        eegdata = hdf.create_dataset('eeg', (nsigs, nsamples0), dtype='float32',
                                     # chunks=(nsigs,fs0),
                                     chunks=True,
                                     fletcher32=True,
                                     # compression='gzip',
                                     # compression='lzf',
                                     # maxshape=(256,None)
                                     )
        # no compression     -> 50 MiB     can view eegdata in vitables
        # compression='gzip' -> 27 MiB    slower
        # compression='lzf'  -> 35 MiB
        # compression='lzf' maxshape=(256,None) -> 36MiB
        # szip is unavailable
        patient = hdf.create_group('patient')

        # add meta data
        hdf.attrs['number_signals'] = nsigs
        hdf.attrs['sample_frequency'] = fs0
        hdf.attrs['nsamples0'] = nsamples0
        patient.attrs['gender_b'] = ef.gender_b
        patient.attrs['patientname'] = ef.patient_name  # PHI
        print('birthdate: %s' % ef.birthdate_b, type(ef.birthdate_b))
        # this is a string -> date (datetime)
        if not ef.birthdate_b:
            print("no birthday in this file")
            birthdate = None
        else:
            birthdate = dateutil.parser.parse(ef.birthdate_b)
            print('birthdate (date object):', birthdate_b)

        start_date_time = datetime.datetime(
            ef.startdate_year,
            ef.startdate_month,
            ef.startdate_day,
            ef.starttime_hour,
            ef.starttime_minute,
            ef.starttime_second)  # ,tzinfo=dateutil.tz.tzlocal())
        print(start_date_time)
        if start_date_time and birthdate:
            age = start_date_time - birthdate
            print('age:', age)
        else:
            age = None

        if age:
            patient.attrs['post_natal_age_days'] = age.days
        else:
            patient.attrs['post_natal_age_days'] = -1

        # now start storing the lists of things: labels, units...
        # nsigs = len(label_list)
        # variable ascii string (or b'' type)
        str_dt = h5py.special_dtype(vlen=str)
        label_ds = hdf.create_dataset('signal_labels', (nsigs,), dtype=str_dt)
        units_ds = hdf.create_dataset('signal_units', (nsigs,), dtype=str_dt)
        labels = []
        units = list()
        # signal_nsamples = []
        for ii in range(nsigs):
            labels.append(ef.signal_label(ii))
            units.append(ef.physical_dimension(ii))
            # self.signal_nsamples.append(self.cedf.samples_in_file(ii))
            # self.samplefreqs.append(self.cedf.samplefrequency(ii))
        # eegdata.signal_labels = labels
        # labels are fixed length strings
        labels_strip = [ss.strip() for ss in labels]
        label_ds[:] = labels_strip
        units_ds[:] = units
        # should be more and a switch for anonymous or not

        # need to change this to

        nchunks = int(nsamples0 // fs0)
        samples_per_chunk = int(fs0)
        buf = np.zeros((nsigs, samples_per_chunk),
                       dtype='float64')  # buffer is float64_t

        print('nchunks: ', nchunks, 'samples_per_chunk:',  samples_per_chunk)

        bookmark = 0  # mark where were are in samples
        for ii in range(nchunks):
            for jj in range(nsigs):
                # readsignal(self, signalnum, start, n,
                # np.ndarray[np.float64_t, ndim = 1] sigbuf)
                # read_phys_signal(chn, 0, nsamples[chn], v)
                #read_phys_signal(self, signalnum, start, n, np.ndarray[np.float64_t, ndim=1] sigbuf)
                print(ii,jj)
                ef.read_phys_signal(jj, bookmark, samples_per_chunk, buf[jj])  # readsignal converts into float
            # conversion from float64 to float32
            eegdata[:, bookmark:bookmark + samples_per_chunk] = buf
            # bookmark should be ii*fs0
            bookmark += samples_per_chunk
        left_over_samples = nsamples0 - nchunks * samples_per_chunk
        print('left_over_samples:', left_over_samples)

        if left_over_samples > 0:
            for jj in range(nsigs):
                ef.read_phys_signal(jj, bookmark, left_over_samples, buf[jj])
            eegdata[:,
                    bookmark:bookmark + left_over_samples] = buf[:,
                                                                 0:left_over_samples]
        hdf.close()


def edf_block_iter_generator(
        edf_file, nsamples, samples_per_chunk, dtype='int32'):
    """
    factory to produce generators for iterating through an edf file and filling
    up an array from the edf with the signal data starting at 0. You choose the
    number of @samples_per_chunk, and number of samples to do in total
    @nsamples as well as the dtype. 'int16' is reasonable as well 'int32' will
    handle everything though


    it yields -> (numpy_buffer, mark, num)
        numpy_buffer,
        mark, which is where in the file in total currently reading from
        num   -- which is the number of samples in the buffer (per signal) to transfer
    """

    nchan = edf_file.signals_in_file

    # 'int32' will work for int16 as well
    buf = np.zeros((nchan, samples_per_chunk), dtype=dtype)

    nchunks = nsamples // samples_per_chunk
    left_over_samples = nsamples - nchunks * samples_per_chunk

    mark = 0
    for ii in range(nchunks):
        for cc in range(nchan):
            edf_file.read_digital_signal(cc, mark, samples_per_chunk, buf[cc])

        yield (buf, mark, samples_per_chunk)
        mark += samples_per_chunk
        # print('mark:', mark)
    # left overs
    if left_over_samples > 0:
        for cc in range(nchan):
            edf_file.read_digital_signal(cc, mark, left_over_samples, buf[cc])

        yield (buf[:, 0:left_over_samples], mark, left_over_samples)


def dig2phys(eeghdf, start, end, chstart, chend):
    # edfhdr->edfparam[i].bitvalue = (edfhdr->edfparam[i].phys_max - edfhdr->edfparam[i].phys_min) / (edfhdr->edfparam[i].dig_max - edfhdr->edfparam[i].dig_min);
    # edfhdr->edfparam[i].offset = edfhdr->edfparam[i].phys_max /
    # edfhdr->edfparam[i].bitvalue - edfhdr->edfparam[i].dig_max;
    dmins = eeghdf['signal_digital_mins'][:]
    dmaxs = eeghdf['signal_digital_maxs'][:]
    phys_maxs = eeghdf['signal_physical_maxs'][:]
    phys_mins = eeghdf['signal_physical_mins'][:]
    print('dmaxs:', repr(dmaxs))
    print('dmins:', repr(dmins))
    print('dmaxs[:] - dmins[:]', dmaxs - dmins)
    print('phys_maxs', phys_maxs)
    print('phys_mins', phys_mins)
    bitvalues = (phys_maxs - phys_mins) / (dmaxs - dmins)
    offsets = phys_maxs / bitvalues - dmaxs
    print('bitvalues, offsets:', bitvalues, offsets)
    print('now change their shape to column vectors')
    for arr in (bitvalues, offsets):
        if len(arr.shape) != 1:
            print('logical errror %s shape is unexpected' % arr.shape)
            raise Exception
        s = arr.shape
        arr.shape = (s[0], 1)
    print('bitvalues, offsets:', bitvalues, offsets)
    # buf[i] = phys_bitvalue * (phys_offset + (double)var.two_signed[0]);
    dig_signal = eeghdf['signals'][chstart:chend, start:end]
    # signal = bitvalues[chstart:chend] *(dig_signal[chstart:chend,:] + offsets[chstart:chend])
    phys_signals = (dig_signal[:, start:end] + offsets) * bitvalues
    # return signal, bitvalues, offsets
    return phys_signals


# TODO: create edf -> hdf version 1000
# hdf -> edf for hdf version 1000
# tests to verify that round trip is lossless
# [] writing encoding of MRN
# [] and entry of mapped pt_code into database coe

# Plan
# v = ValidateTrackHeader(header=h)
# if v.is_valid():
#     process(v.cleaned_data)
# else:
#    mark_as_invalid(h)

def first(mapping):
    if mapping:
        return mapping[0]
    else:
        return mapping # say mapping = [] or None

class ValidateTrackHeaderLPCH:
    # after validated place all data in cleaned_data field
    def __init__(self, header):
        # TOOO: validate that databae_source_label is in accepted sources
        self.hdr = header.copy()
        self.validated = False
        # self.clean = False
        self.cleaned_data = {} # vs update/copy from header

    def is_valid(self):
        # if name contains "Test" then we should skip this file and log it
        mrnobj = None
        try:
            if name_is_test(self.hdr['patient_name']):
                raise ValidationError('test file encountered', code='test file', params=self.hdr)

            # if we have a valid mrn, then we can potentially look up the patient or even the study
            mrn_ok = valid_lpch_mrn(self.hdr['patientcode'])
            if mrn_ok:
                mrn = self.hdr['patientcode'].strip()
                self.cleaned_data['patientcode'] = mrn
            else:
                raise ValidationError('bad MRN', code='bad mrn', params=self.hdr['patientcode'])

            if valid_lpch_name(self.hdr['patient_name']):
                self.cleaned_data['patient_name'] = self.hdr['patient_name'].strip()
            else:
                if mrn_ok:               # try to look up patient in databases
                    # look up name, dob here based upon mrn in nk_db and/or epic_db
                    mrnobj = models.NkMrn.query.filter_by(mrn=mrn).first()
                    if mrnobj:
                        self.cleaned_data['patient_name'] = mrnobj.nkpatient.name
                else:
                    raise ValidationError('invalid patient name', 'invalid name',
                                          params=self.hdr)

            eegno_ok = valid_lpch_eegno(self.hdr['admincode'])
            if eegno_ok:
                self.cleaned_data['admincode'] = _csu(self.hdr['admincode'])
            else:
                raise ValidationError('bad eegno/admincode', code='invalid admincode', params=self.hdr)
            
            if self.hdr['birthdate_date']:
                self.cleaned_data['birthdate_date'] = self.hdr['birthdate_date']
            else:
                # then couldn't make a date, see if can find birthday in database
                if mrn_ok:
                    mrnobj = mrnobj if mrnobj else models.NkMrn.query.filter_by(mrn=mrn).first()

                    if not mrnobj:
                        raise ValidationError('bad birthdate_date','birthdate error', params=self.hdr)
                    else:
                        nbday = mrnobj.nkpatient.dob
                        self.cleaned_data['birthdate_date'] = nbday

                else:
                    raise ValidationError('bad birthday','birthday error', params=self.hdr)
            
            # copy over other header members
            # todo: should do more validation of 'gender'
            self.cleaned_data['gender'] = self.hdr['gender'] 

            self.cleaned_data['file_name'] = self.hdr['file_name']
            self.cleaned_data['filetype'] = self.hdr['filetype']
            self.cleaned_data['signals_in_file'] = self.hdr['signals_in_file']
            self.cleaned_data['datarecords_in_file'] = self.hdr['datarecords_in_file']
            self.cleaned_data['file_duration_100ns'] = self.hdr['file_duration_100ns']
            self.cleaned_data['file_duration_seconds'] = self.hdr['file_duration_seconds']
            self.cleaned_data['startdate_date'] = self.hdr['startdate_date']
            self.cleaned_data['start_datetime'] = self.hdr['start_datetime']
            self.cleaned_data['starttime_subsecond_offset'] = self.hdr['starttime_subsecond_offset']
            self.cleaned_data['patient_additional'] = self.hdr['patient_additional'].strip()
            self.cleaned_data['technician'] = self.hdr['technician'].strip()
            self.cleaned_data['equipment'] = self.hdr['equipment'].strip()
            self.cleaned_data['recording_additional'] = self.hdr['recording_additional'].strip()
            self.cleaned_data['datarecord_duration_100ns'] = self.hdr['datarecord_duration_100ns']

            self.validated = True
            return True
        except ValidationError as ve:
            self.errors = ve.message
            self.error_code = ve.code
            self.error_params = ve.params
            debug(ve.message)
            return False

class AnonymizeTrackHeaderLPCH(ValidateTrackHeaderLPCH):
    LPCH_DEFAULT_BIRTH_DATETIME = datetime.datetime(year=1990, month=1, day=1)
    # datatbase sources
    LPCH_NK = 'LPCH_NK'
    STANFORD_NK = 'STANFORD_NK'

    def __init__(self, header, source_database_label=LPCH_NK):
        super().__init__(header)
        with app.app_context():
            self.anonymous_header = models.register_and_create_anonymous_header(self.hdr, source_database_label=source_database_label)

            # will need to track: patient, study, file
            # file needs source and key NK origin


class ValidateTrackHeaderStanford:
    # after validated place all data in cleaned_data field
    def __init__(self, header):
        # TOOO: validate that databae_source_label is in accepted sources
        self.hdr = header.copy()
        self.validated = False
        # self.clean = False
        self.cleaned_data = {} # vs update/copy from header

    def is_valid(self):
        # if name contains "Test" then we should skip this file and log it
        mrnobj = None
        try:
            if name_is_test(self.hdr['patient_name']):
                raise ValidationError('test file encountered', code='test file', params=self.hdr)

            # if we have a valid mrn, then we can potentially look up the patient or even the study
            mrn_ok = valid_stanford_mrn(self.hdr['patientcode']) 
            if mrn_ok:
                mrn = self.hdr['patientcode'].strip()
                self.cleaned_data['patientcode'] = mrn
            else:
                raise ValidationError('bad MRN', code='bad mrn', params=self.hdr['patientcode'])

            if valid_stanford_name(self.hdr['patient_name']):
                self.cleaned_data['patient_name'] = self.hdr['patient_name'].strip()
            else:
                if mrn_ok:               # try to look up patient in databases
                    # look up name, dob here based upon mrn in nk_db and/or epic_db
                    mrnobj = models.NkMrn.query.filter_by(mrn=mrn).first()
                    if mrnobj:
                        self.cleaned_data['patient_name'] = mrnobj.nkpatient.name
                else:
                    raise ValidationError('invalid patient name', 'invalid name',
                                          params=self.hdr)

            eegno_ok = valid_stanford_eegno(self.hdr['admincode'])
            if eegno_ok:
                self.cleaned_data['admincode'] = _csu(self.hdr['admincode'])
            else:
                raise ValidationError('bad eegno/admincode', code='invalid admincode', params=self.hdr)
            
            if self.hdr['birthdate_date']:
                self.cleaned_data['birthdate_date'] = self.hdr['birthdate_date']
            else:
                # then couldn't make a date, see if can find birthday in database
                if mrn_ok:
                    mrnobj = mrnobj if mrnobj else models.NkMrn.query.filter_by(mrn=mrn).first()

                    if not mrnobj:
                        raise ValidationError('bad birthdate_date','birthdate error', params=self.hdr)
                    else:
                        nbday = mrnobj.nkpatient.dob
                        self.cleaned_data['birthdate_date'] = nbday

                else:
                    raise ValidationError('bad birthday','birthday error', params=self.hdr)
            
            # copy over other header members
            # todo: should do more validation of 'gender'
            self.cleaned_data['gender'] = self.hdr['gender'] 

            self.cleaned_data['file_name'] = self.hdr['file_name']
            self.cleaned_data['filetype'] = self.hdr['filetype']
            self.cleaned_data['signals_in_file'] = self.hdr['signals_in_file']
            self.cleaned_data['datarecords_in_file'] = self.hdr['datarecords_in_file']
            self.cleaned_data['file_duration_100ns'] = self.hdr['file_duration_100ns']
            self.cleaned_data['file_duration_seconds'] = self.hdr['file_duration_seconds']
            self.cleaned_data['startdate_date'] = self.hdr['startdate_date']
            self.cleaned_data['start_datetime'] = self.hdr['start_datetime']
            self.cleaned_data['starttime_subsecond_offset'] = self.hdr['starttime_subsecond_offset']
            self.cleaned_data['patient_additional'] = self.hdr['patient_additional'].strip()
            self.cleaned_data['technician'] = self.hdr['technician'].strip()
            self.cleaned_data['equipment'] = self.hdr['equipment'].strip()
            self.cleaned_data['recording_additional'] = self.hdr['recording_additional'].strip()
            self.cleaned_data['datarecord_duration_100ns'] = self.hdr['datarecord_duration_100ns']

            self.validated = True
            return True
        except ValidationError as ve:
            self.errors = ve.message
            self.error_code = ve.code
            self.error_params = ve.params
            debug(ve.message)
            return False

class AnonymizeTrackHeaderStanford(ValidateTrackHeaderStanford):
    STANFORD_DEFAULT_BIRTH_DATETIME = datetime.datetime(year=1910, month=1, day=1)
    # datatbase sources
    LPCH_NK = 'LPCH_NK'
    STANFORD_NK = 'STANFORD_NK'

    def __init__(self, header, source_database_label='STANFORD_NK'):
        super().__init__(header)
        with app.app_context():
            self.anonymous_header = models.register_and_create_anonymous_header(self.hdr, source_database_label=source_database_label)

            # will need to track: patient, study, file
            # file needs source and key NK origin


def find_blocks(arr):
    blocks = []
    print("total arr:", arr)
    dfs = np.diff(arr)
    dfs_ind = np.where(dfs != 0.0)[0]
    last_ind = 0
    for dd in dfs_ind+1:
        print("block:",arr[last_ind:dd])
        blocks.append((last_ind,dd))
        last_ind = dd
    print("last block:", arr[last_ind:])
    blocks.append( (last_ind,len(arr)))
    return blocks


def find_blocks2(arr):
    blocks = []
    N = len(arr)
    print("total arr:", arr)
    last_ind = 0
    last_val = arr[0]
    for ii in range(1,N):
        if last_val == arr[ii]:
            pass
        else:
            blocks.append((last_ind,ii))
            last_ind = ii
            last_val = arr[ii]
    blocks.append((last_ind,N))
    return blocks



def test_find_blocks1():
    s = [250.0, 250.0, 250.0, 1.0, 1.0, 1000.0, 1000.0]
    blocks = find_blocks(s)
    print("blocks:")
    print(blocks)


def test_find_blocks2():
    s = [250.0, 250.0, 250.0, 1.0, 1.0, 1000.0, 1000.0]
    blocks = find_blocks2(s)
    print("blocks:")
    print(blocks)

def test_find_blocks2_2():
    s = [100,100,100,100,100,100,100,100]
    blocks = find_blocks2(s)
    print("blocks:")
    print(blocks)

    

def edf2hdf2(fn, outfn='', hdf_dir='', anonymize=False):
    """
    convert an edf file to hdf5 using fairly straightforward mapping
    return True if successful
    
    @database_sourcel_label tells us which database it came from LPCH_NK or STANFORD_NK
       this is important!
    """

    if not outfn:
        base = os.path.basename(fn)
        base, ext = os.path.splitext(base)

        base = base + '.eeghdf'
        outfn = os.path.join(hdf_dir, base)
        # print('outfn:', outfn)
        # all the data point related stuff

    with edflib.EdfReader(fn) as ef:

        # read all EDF+ header information in just the way I want it
        header = {
            'file_name': os.path.basename(fn),
            'filetype': ef.filetype,
            'patient_name': ef.patient_name,
            'patientcode': ef.patientcode,
            'studyadmincode': ef.admincode,
            'gender': ef.gender,

            'signals_in_file': ef.signals_in_file,
            'datarecords_in_file': ef.datarecords_in_file,
            'file_duration_100ns': ef.file_duration_100ns,
            'file_duration_seconds': ef.file_duration_seconds,
            'startdate_date': datetime.date(ef.startdate_year, ef.startdate_month, ef.startdate_day),
            'start_datetime': datetime.datetime(ef.startdate_year, ef.startdate_month, ef.startdate_day,
                                                ef.starttime_hour, ef.starttime_minute, ef.starttime_second),
            'starttime_subsecond_offset': ef.starttime_subsecond,

            'birthdate_date': ef.birthdate_date,
            'patient_additional': ef.patient_additional,
            'admincode': ef.admincode,  # usually the study eg. C13-100
            'technician': ef.technician,
            'equipment': ef.equipment,
            'recording_additional': ef.recording_additional,
            'datarecord_duration_100ns': ef.datarecord_duration_100ns,
        }
        pprint.pprint(header)

        #### validation code #####
        validator = None
        # if source_database_label=='LPCH_NK':
        #     validator = ValidateTrackHeaderLPCH(header=header)
        # elif source_database_label== 'STANFORD_NK':
        #     validator = ValidateTrackHeaderStanford(header=header)
        # else:
        #     raise ValidationError

        # if not validator.is_valid():
        #     print('problem with this file:', fn)
        #     print(validator.errors,validator.error_code,
        #           validator.error_params)

        #     return False, validator
        # else:
        #     print('\nvalid header::')
        #     pprint.pprint(validator.cleaned_data)
        #     header = validator.cleaned_data            

        # from here on the header is valid and cleaned

        
        #  use arrow
        start_datetime = header['start_datetime']

        # end_date_time = datetime.datetime(ef.enddate_year, ef.enddate_month, ef.enddate_day, ef.endtime_hour,
        # ef.endtime_minute, ef.endtime_second) # tz naive
        # end_date_time - start_date_time
        duration = datetime.timedelta(seconds=header['file_duration_seconds'])
        

        # derived information
        birthdate = header['birthdate_date']
        if birthdate:
            age = arrow.get(start_datetime) - arrow.get(header['birthdate_date'])

            debug('predicted age: %s' % age)
            # total_seconds() returns a float
            debug('predicted age (seconds): %s' % age.total_seconds())
        else:
            age = datetime.timedelta(seconds=0)

        # if anonymize:
        #     if source_database_label== 'LPCH_NK':
        #         anonymizer = AnonymizeTrackHeaderLPCH(header, source_database_label=source_database_label)
        #     if source_database_label == 'STANFORD_NK':
        #         anonymizer = AnonymizeTrackHeaderStanford(header, source_database_label=source_database_label)

        #     header = anonymizer.anonymous_header  # replace the original header with the anonymous one
        #     print('anonymized header')
        #     pprint.pprint(header)

        # anonymized version if necessary
        header['end_datetime'] = header['start_datetime'] + duration

        ############# signal array information ##################

        # signal block related stuff
        nsigs = ef.signals_in_file

        # again know/assume that this is uniform sampling across signals
        fs0 = ef.samplefrequency(0)
        signal_frequency_array = ef.get_signal_freqs()
        dfs = np.diff(signal_frequency_array)
        dfs_ind = np.where(dfs != 0.0)
        dfs_ind = dfs_ind[0]
        last_ind = 0
        for dd in dfs_ind+1:
            print("block:",signal_frequency_array[last_ind:dd])
            last_ind = dd
        print("last block:", signal_frequency_array[last_ind:])
        
        print("where does sampling rate change?", np.where(dfs != 0.0))
        print("elements:", signal_frequency_array[np.where(dfs != 0.0)])
        print("signal_frequency_array::\n", repr(signal_frequency_array))
        print("len(signal_frequency_array):", len(signal_frequency_array))
        
        assert all(signal_frequency_array[:-3] == fs0)

        nsamples0 = ef.samples_in_file(0)  # samples per channel
        print('nsigs=%s, fs0=%s, nsamples0=%s\n' % (nsigs, fs0, nsamples0))

        num_samples_per_signal = ef.get_samples_per_signal()  # np array
        print("num_samples_per_signal::\n", repr(num_samples_per_signal), '\n')

        # assert all(num_samples_per_signal == nsamples0)

        file_duration_sec = ef.file_duration_seconds
        #print("file_duration_sec", repr(file_duration_sec))

        # Note that all annotations except the top row must also specify a duration.

        # long long onset; /* onset time of the event, expressed in units of 100
        #                     nanoSeconds and relative to the starttime in the header */

        # char duration[16]; /* duration time, this is a null-terminated ASCII text-string */

        # char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1]; /* description of the
        #                             event in UTF-8, this is a null term string of max length 512*/

        # start("x.y"), end, char[20]
        # annotations = ef.read_annotations_as_array() # get numpy array of
        # annotations

        annotations_b = ef.read_annotations_b_100ns_units()

        # print("annotations_b::\n")
        # pprint.pprint(annotations_b)  # get list of annotations

        signal_text_labels = ef.get_signal_text_labels()
        print("signal_text_labels::\n")
        pprint.pprint(signal_text_labels)
        print("normalized text labels::\n")
        signal_text_labels_lpch_normalized = [
            normalize_lpch_signal_label(label) for label in signal_text_labels]
        pprint.pprint(signal_text_labels_lpch_normalized)

        # ef.recording_additional

        # print()
        signal_digital_mins = np.array(
            [ef.digital_min(ch) for ch in range(nsigs)])
        signal_digital_total_min = min(signal_digital_mins)

        print("digital mins:", repr(signal_digital_mins))
        print("digital total min:", repr(signal_digital_total_min))

        signal_digital_maxs = np.array(
            [ef.digital_max(ch) for ch in range(nsigs)])
        signal_digital_total_max = max(signal_digital_maxs)
      
        print("digital maxs:", repr(signal_digital_maxs))
        #print("digital total max:", repr(signal_digital_total_max))

        signal_physical_dims = [
            ef.physical_dimension(ch) for ch in range(nsigs)]
        # print('signal_physical_dims::\n')
        # pprint.pprint(signal_physical_dims)
        #print()

        signal_physical_maxs = np.array(
            [ef.physical_max(ch) for ch in range(nsigs)])

        #print('signal_physical_maxs::\n', repr(signal_physical_maxs))

        signal_physical_mins = np.array(
            [ef.physical_min(ch) for ch in range(nsigs)])

        #print('signal_physical_mins::\n', repr(signal_physical_mins))

        # this don't seem to be used much so I will put at end
        signal_prefilters = [ef.prefilter(ch).strip() for ch in range(nsigs)]
        #print('signal_prefilters::\n')
        # pprint.pprint(signal_prefilters)
        #print()
        signal_transducers = [ef.transducer(ch).strip() for ch in range(nsigs)]
        #print('signal_transducers::\n')
        #pprint.pprint(signal_transducers)

        with eeghdf.EEGHDFWriter(outfn, 'w') as eegf:
            eegf.write_patient_info(patient_name=header['patient_name'],
                                    patientcode=header['patientcode'],
                                    gender=header['gender'],
                                    birthdate_isostring=header['birthdate_date'],
                                    # gestational_age_at_birth_days
                                    # born_premature
                                    patient_additional=header['patient_additional'])

            signal_text_labels_lpch_normalized = [
                normalize_lpch_signal_label(label) for label in signal_text_labels]

            rec = eegf.create_record_block(record_duration_seconds=header['file_duration_seconds'],
                                           start_isodatetime=str(header['start_datetime']),
                                           end_isodatetime=str(header['end_datetime']),

                                           number_channels=header['signals_in_file'],
                                           num_samples_per_channel=nsamples0,
                                           sample_frequency=fs0,
                                           signal_labels=signal_text_labels_lpch_normalized,
                                           signal_physical_mins=signal_physical_mins,
                                           signal_physical_maxs=signal_physical_maxs,
                                           signal_digital_mins=signal_digital_mins,
                                           signal_digital_maxs=signal_digital_maxs,
                                           physical_dimensions=signal_physical_dims,
                                           patient_age_days=age.total_seconds() / 86400.0,
                                           signal_prefilters=signal_prefilters,
                                           signal_transducers=signal_transducers,
                                           technician=header['technician'],
                                           studyadmincode=header['studyadmincode'])

            eegf.write_annotations_b(annotations_b)  # may be should be called record annotations

            edfblock_itr = edf_block_iter_generator(
                ef,
                nsamples0,
                100 * ef.samples_in_datarecord(0)*header['signals_in_file'], # samples_per_chunk roughly 100 datarecords at a time
                dtype='int32')

            signals = eegf.stream_dig_signal_to_record_block(rec, edfblock_itr)
    
        return True, validator # we succeeded


def test_edf2hdf_info():
    # on chris's macbook
    EDF_DIR = r'/Users/clee/code/eegml/nk_database_proj/private/lpch_edfs'
    fn = os.path.join(EDF_DIR, 'XA2731AX_1-1+.edf')
    edf2hdf(filename)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        file_name = sys.argv[1]
        edf2hdf2(file_name)
