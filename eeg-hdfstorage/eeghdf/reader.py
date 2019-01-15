# -*- coding: utf-8 -*-
"""
functions to help with reading eeghdf files
versions 1...
"""
# python 2/3 compatibility - write as if in python 3.5
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
from past.utils import old_div

import logging
import numpy as np
import h5py
import pandas as pd # maybe shouldn't require pandas in basic file so look to eliminate


def record_edf_annotations_to_lists(raw_edf_annotations):
    """
    usage: 

    >>> rec = hdf['record-0']
    >>> texts, times_100ns = record_edf_annotations_to_lists(rec['edf_annotations'])
    """
    byte_texts = raw_edf_annotations['texts'] # still byte encoded
    antexts = [s.decode('utf-8') for s in byte_texts[:]]

    starts100ns_arr = raw_edf_annotations['starts_100ns'][:]
    starts100ns = [xx for xx in starts100ns_arr]
    return antexts, starts100ns

def record_edf_annotations_to_sec_items(raw_edf_annotations):
    """
    rec = hdf['record-0']
    annotation_items = record_edf_annotations_to_sec_items(rec['edf_annotations'])
    # returns (text, <start time (sec)>) pairs
    """
    byte_texts = raw_edf_annotations['texts'] # still byte encoded
    antexts = [s.decode('utf-8') for s in byte_texts[:]]

    starts100ns_arr = raw_edf_annotations['starts_100ns'][:]
    starts_sec_arr = starts100ns_arr/10000000  #  (10**7) * 100ns = 1 second 
    items = zip(antexts, starts_sec_arr)
    return items


# so given a hdf 
# signals(integers) -> optional-uV-conversion -> optional montage conversion (???scope of this project)
# 

## let's try a first draft to get a feel for things

class PhysSignal:
    """This does a very limited imitation of the normal hdf signal buffer
    with additional transformation of the returned array coverted to be in physical units
    e.g. uV or mV
    @s2u is the array of scaling factors turning samples to units
    @S2U = diagonal(s2u)
    @offset is/will be offset to add to sample before scaling
    data is hdf dataset of sampled data(or numpy array) to transform
    First implement the zero offset version

    $Y = S2U [X + Offset]$ 

    This is meant to be accessed via the bracket operator
    phys_signals[a,b]
    phys_signals[a:b, c:d]
    phys_signal[a, b:c]
    phys_signals[a:b, c]

    Anything else has not been tested - I am only testing the first two values of the tuple
    """
    def __init__(self, data,s2u, S2U, offset):
        self.data = data # hdf data array
        self.s2u = s2u
        self.S2U = S2U
        self.offset = offset

    def __getitem__(self,slcitm):
        if isinstance(slcitm, slice): 
            # e.g. s[3:5] returning full versions of channels 3,4 with all samples so slcitm represents
            # channels 
            # print("slice path:", slcitm) # debug
            ch_slice = slcitm
            buf = self.data[ch_slice] + self.offset[ch_slice,np.newaxis] # may need to be [ch_slice,np.newaxis]
            sc = self.s2u[slcitm] # scaling 
            res = sc * buf.T # broadcasting
            return res.T

        elif isinstance(slcitm, tuple):
            # print("multi-dim:", slcitm) # debug
            if isinstance(slcitm[0],slice):
                ch_slice = slcitm[0]
                s1 = slcitm[1]

                if isinstance(slcitm[1],int):
                    # print('see int in second element of tuple') # debug
                    tmpdata = self.data[slcitm] + self.offset[slcitm[0]]
                    #collapses shape to (M,)
                    return (tmpdata.T * self.s2u[ch_slice]).T

                # slice a subset of channels

                sc = self.s2u[ch_slice]
                tmp =  sc * (self.data[ch_slice, s1] + self.offset[ch_slice,np.newaxis]).T
                return tmp.T
            
            if  isinstance(slcitm[0],(list,tuple)):
                # print('list/tuple path fancy indexing') # debug
                ch_slice = slcitm[0]
                sc = self.s2u[ch_slice]
                tmp_all_chan = self.data[:,slcitm[1]] # this may make a copy 
                buf = tmp_all_chan[ch_slice,:] + self.offset[ch_slice,np.newaxis]
                return (sc * buf.T).T
            
            elif isinstance(slcitm[0],int): # a single channel with subset of samples
                # print('int ch path:', slcitm) # debug
                a = self.s2u[slcitm[0]]
                return a * (self.data[slcitm] + self.offset[slcitm[0]] )
            # multidim

        else:
            # print('return a channel:', slcitm)
            # just a single integer
            a = self.s2u[slcitm]
            return a * (self.data[slcitm] + self.offset[slcitm])


    @property
    def shape(self):
        return self.data.shape


class PhysSignalZeroOffset:
    """This does a very limited imitation of the normal hdf signal buffer
    with additional transformation of the returned array coverted to be in physical units
    e.g. uV or mV
    @s2u is the array of scaling factors turning samples to units
    @S2U = diagonal(s2u)
    @offset is/will be offset to add to sample before scaling
    data is hdf dataset of sampled data(or numpy array) to transform
    First implement the zero offset version

    $Y = S2U [X + Offset]$ <- but for now using just offset zero

    This is meant to be accessed via the bracket operator
    phys_signals[a,b]
    phys_signals[a:b, c:d]
    phys_signal[a, b:c]
    phys_signals[a:b, c]

    Anything else has not been tested - I am only testing the first two values of the tuple
    """
    def __init__(self, data,s2u, S2U, offset):
        """@s2u is the array with each components scaling
        @S2U = diag(s2u) usually"""
        self.data = data # hdf data array
        self.s2u = s2u
        self.S2U = S2U
        
        # self.offset = offset

    def __getitem__(self,slcitm):
        # print('PhysSignalZeroOffset-slcitm type:', type(slcitm), '\nvalue:', slcitm) # debug
        if isinstance(slcitm, slice): 
            # e.g. s[3:5] returning full versions of channels 3,4 with all samples
            # print("slice", slcitm)
            buf = self.data[slcitm]
            S = self.s2u[slcitm]
            res = S * buf.T # broadcasting
            return res.T

        elif isinstance(slcitm, tuple):
            # print("multi-dim:", slcitm)
            assert len(slcitm) == 2

            if isinstance(slcitm[0],slice): 
                #print('gn down slice path')
                #print('self.S2U:', self.S2U)
                ## A = self.S2U[slcitm[0],slcitm[0]]
                #print('A.shape:', A.shape)
                ## buf = self.data.__getitem__(slcitm)
                ## return np.dot(A,buf)
                ch_slice = slcitm[0]
                su = self.s2u
                if isinstance(slcitm[1],int):
                    tmpdata = self.data[slcitm] 
                    #collapses shape to (M,)
                    #return tmp[:, np.newaxis] * self.s2u[ch_slice,np.newaxis]
                    return (tmpdata.T * self.s2u[ch_slice]).T
                # not int
                # return self.data[slcitm] * su[ch_slice,np.newaxis]
                return (self.data[slcitm].T * su[ch_slice]).T
                # return self.data[ch_slice, slcitm[1]] * su[ch_slice,np.newaxis]
            
            if  isinstance(slcitm[0],(list,tuple)):
                #print('list/tuple path fancy indexing')
                ## a = self.s2u[slcitm[0]]
                ## A = np.diag(a)
                # print('A.shape:', A.shape)

                # need to do this because h5py can't handle out of order list of indexes
                # could avoid if new slcitm[0] was an ordered list or tuple
                # note that fancy indexing with a list preserves the number of dimensions
                tmp_all_chan = self.data[:,slcitm[1]]
                if tmp_all_chan.ndim == 1:
                    tmp_all_chan.shape = (tmp_all_chan.shape[0], 1)

                buf = tmp_all_chan[slcitm[0],:] # use fancy indexing on channels
                ##return np.dot(A,buf)

                # if isinstace(slcitm[1],int):
                #     tmp = self.data[slcitm] #collapses shape to (M,)
                #     return tmp[:, np.newaxis] * self.s2u[ch_slice,np.newaxis]

                return buf * self.s2u[slcitm[0], np.newaxis]
        
            if isinstance(slcitm[0],slice):
                ## A = self.S2U[slcitm[0],slcitm[0]]
                ## buf = self.data.__getitem__(slcitm)
                ## return np.dot(A,buf)
                return self.data[slcitm] * self.s2u[slcitm[0], np.newaxis]

            elif isinstance(slcitm[0],int): # a single channel with subset of samples
                a = self.s2u[slcitm[0]]
                return a * self.data[slcitm]

            # end multidim

        else:
            # print('return a chanel:', slcitm)
            a = self.s2u[slcitm]
            return a * self.data[slcitm]

            # just a single integer
        raise Exception('oops unhandled case in __getitem__')

    @property
    def shape(self):
        return self.data.shape




class Eeghdf:
    __version__ = 1
    def __init__(self, fn, mode='r'):
        """
        version 1: assumes only one record-0 waveform
        but may allow for record_list in future

        h5py.File mode options
        r readonly
        r+ read/write, file must exist
        w create file, truncate if exists
        w- or x create file, fail if exists
        a read/write if exists, create otherwise
        """
        self.file_name = fn
        self.hdf = h5py.File(fn, mode=mode) # readonly by default



        # waveform record info
        self.rec0 = self.hdf['record-0']
        rec0 = self.rec0
        self.age_years = rec0.attrs['patient_age_days'] / 365 # age of patient at time of record

        self.rawsignals = rec0['signals']
        labels_bytes = rec0['signal_labels']
        self.electrode_labels = [str(s,'ascii') for s in labels_bytes]

        self._SAMPLE_TO_UNITS = False # indicate if have calculated conversion factors flag


        # read annotations and format them for easy use
        annot = rec0['edf_annotations'] # but these are in a funny 3 array format
        antext = [s.decode('utf-8') for s in annot['texts'][:]]
        self._annotation_text = antext
        starts100ns = [xx for xx in annot['starts_100ns'][:]]
        self._annotation_start100s = starts100ns
        start_time_sec = [xx/10**7 for xx in starts100ns] # 10**7 * 100ns = 1sec
        df = pd.DataFrame(data=antext,columns=['text'])
        df['starts_sec'] = start_time_sec
        df['starts_100ns'] = starts100ns
        self.edf_annotations_df = df

        self._physical_dimensions = None

        # what about units and conversion factors
        self.start_isodatetime = rec0.attrs['start_isodatetime'] # = start_isodatetime
        self.end_isodatetime = rec0.attrs['end_isodatetime'] # = end_isodatetime

        self.number_channels = rec0.attrs['number_channels'] # = number_channels
        self.number_samples_per_channel = rec0.attrs['number_samples_per_channel'] # = num_samples_per_channel
        self.sample_frequency = rec0.attrs['sample_frequency'] # = sample_frequency

        # note I am requiring this duration_seconds value to be an integer at
        # this time given edf conventions, but this may need to change
        self.duration_seconds = int(round(self.number_samples_per_channel/ self.sample_frequency,0))



    # @property
    # def duration_seconds(self):
    #     """
    #     >>> import eeghdf
    #     >>> eeg = Eeghdf("../notebooks/archive/DA05505C_1-1+.eeghdf")
    #     >>> eeg.duration_seconds
    #     3046.0
    #     """
    #     #startdt = self.rec0['start_isodatetime']
    #     ##enddt = self.rec0['end_isodatetime
    #     duration_secs = self.number_samples_per_channel/ self.sample_frequency
    #     return duration_secs


        # rec0.attrs['technician'] = technician
        # patient
        self.patient = dict(self.hdf['patient'].attrs)

        self.duration_seconds = self.number_samples_per_channel/self.sample_frequency

    def annotations_contain(self, pat, case=False):
        df = self.edf_annotations_df
        return df[df.text.str.contains(pat,case=case)]

    @property
    def physical_dimensions(self):
        if not self._physical_dimensions:
            self._physical_dimensions = [s.decode('utf-8') for s in self.rec0['physical_dimensions'][:]]
        return self._physical_dimensions

    @property
    def signal_physical_mins(self):
        """return numpy ndarray of waveform physical_mins"""
        return self.hdf['record-0']['signal_physical_mins'][:]

    @property
    def signal_physical_maxs(self):
        """return numpy ndarray of waveform physical_maxs"""
        return self.hdf['record-0']['signal_physical_maxs'][:]

    @property
    def signal_digital_mins(self):
        return self.hdf['record-0']['signal_digital_mins'][:]

    @property
    def signal_digital_maxs(self):
        return self.hdf['record-0']['signal_digital_maxs'][:]

    @property
    def shape(self):
        return self.rawsignals.shape
    
    
    def _calc_sample2units(self):
        """
        calculate the arrays and matrices used to convert sample values to real physical values
        S2U is the scaling/units conversion matrix defined below (units/samples)
        $ V_{physical} = S2U [V_{sample} + offset] $
        note the offset is often zero with many recordings so can be dropped in that case
        :return: None
        """
        pMax = self.signal_physical_maxs
        pMin = self.signal_physical_mins

        dMax = self.signal_digital_maxs
        dMin = self.signal_digital_mins
        s2u = (pMax - pMin)/(dMax-dMin)
        S2U = np.diag(s2u)
        phys_offset = pMax/s2u - dMax # sample units
        self._phys_offset = phys_offset
        self._S2U = S2U
        self._s2u = s2u

                            
        if np.all(np.abs(phys_offset) < 1.0):
            self._phys_signals = PhysSignalZeroOffset(self.rawsignals, self._s2u, self._S2U, self._phys_offset)
        else:
            self._phys_signals = PhysSignal(self.rawsignals, self._s2u, self._S2U, self._phys_offset)

        self._SAMPLE_TO_UNITS = True
                            


    @property
    def phys_signals(self): # -> object:
        if not self._SAMPLE_TO_UNITS:
            self._calc_sample2units()
            print('self._phys_offset:',self._phys_offset)
            # assert np.all(np.abs(self._phys_offset) < 1.0)
            
        return self._phys_signals
    
    #    self.# record['signal_physical_maxs'] = signal_physical_maxs
    # record['signal_digital_mins'] = signal_digital_mins


class Eeghdf_ver2(Eeghdf):
    __version__ = 2
    """
    version 2 supports studyadmincode in record-0 attributes
    but still presumes only one record block ('record-0')
    """

    def __init__(self, fn, mode='r'):
        super().__init__(fn,mode)
        rec0 = self.rec0
        if 'studyadmincode' in rec0.attrs:
            self.studyadmincode = rec0.attrs['studyadmincode']
        else: # fall back if not defined, though it should have been and raise an warning?
            logging.warning("record-0.attrs['studyadmincode'] not defined, likely opened a version 1 file")
            
            self.studyadmincode = ''
