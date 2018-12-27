# -*- coding: utf-8 -*
# eeg_app1.py first try
"""
"""
from __future__ import print_function, division, unicode_literals
import pandas as pd
import numpy as np
import h5py
from pprint import pprint

import bokeh.plotting as bplt
import bokeh.models
import bokeh.layouts as layouts  # column, row, ?grid
# import bokeh.models.widgets as bmw
# import bokeh.models.sources as bms
from bokeh.models import FuncTickFormatter
from bokeh.models.tickers import FixedTicker

from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

import eegvis.montageview 

doc = curdoc()


class EEGBrowser:
    """
    work in bokeh app 
    given an hdf @signal array-like object
    allow:
       - scrolling 
       - goto
       ? filtering
       ? montaging (linear combinations)


    question: explicit DataSource ? vs use implicit datasource in line or multi_line glyph
       r1 = fig.line(xarr, yarr)
       r.data['x'] = new_xarr

    """

    def __init__(self, page_width_sec=15):
        # data related
        self.hdf = h5py.File('archive/YA2741BS_1-1+.eeghdf')
        self.rec = self.hdf['record-0']
        self.signals = self.rec['signals']
        self.num_rows, self.num_samples = self.signals.shape
        self.fs = self.rec.attrs['sample_frequency']

        # extract bytes to strings
        self.electrode_labels = [str(s, 'ascii') for s in self.rec['signal_labels']]

        # plot display related
        self.page_width_secs = page_width_sec

        self.ylabels = self.electrode_labels  # good starting point
        self.yscale = 3.0

        self.bk_handle = None
        self.fig = None
        self.multi_line_glyph = None

        # controling models
        self.data_source = bokeh.models.ColumnDataSource(data=dict(xs=[0], ys=[0]))  # is this ok?

        self.loc_sec = self.page_width_secs / 2.0  # default start for current location
        self.ch_start = 0  # change this to a list of channels for fancy slicing
        self.ch_stop = 19  # self.signals.shape[0]

    def create_plot(self):
        self.fig = self.show_epoch_centered()
        self.fig.xaxis.axis_label = 'seconds'

    def show(self):
        # self.bk_handle = bplt.show(self.fig,notebook_handle=True)
        pass

    def update(self, goto_sec=None):

        numRows = len(self.ticklocs)  # this is a silly way to get this number

        goto_sample = int(self.fs * self.loc_sec)
        page_width_samples = int(self.page_width_secs * self.fs)
        hw = half_width_epoch_sample = int(page_width_samples / 2)

        s0 = self.limit_sample_check(goto_sample - hw)
        s1 = self.limit_sample_check(goto_sample + hw)
        window_samples = s1 - s0
        data = self.signals[self.ch_start:self.ch_stop, s0:s1]  # note transposed
        t = self.page_width_secs * np.arange(window_samples, dtype=float) / window_samples
        t = t + s0 / self.fs  # t = t + start_time
        # t = t[:s1-s0]
        ## this is not quite right if ch_start is not 0
        xs = [t for ii in range(numRows)]
        ys = [self.yscale * data[ii, :] + self.ticklocs[ii] for ii in range(numRows)]

        # for ii in range(len(lg)):
        #     lg[ii].data_source.data["x"] = t
        #     lg[ii].data_source.data['y'] = self.yscale * data[ii,:] + self.ticklocs[ii]
        #     # fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to figure

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        new_data = {'xs': xs, 'ys': ys}
        self.data_source.data = new_data
        # print(new_data)

        # push_notebook(handle=self.bk_handle)

    def stackplot_t(self, tarray, seconds=None, start_time=None, ylabels=None, yscale=1.0, topdown=True, **kwargs):
        """
        will plot a stack of traces one above the other assuming
        @tarray is an nd-array like object with format
        tarray.shape =  numSamples, numRows

        @seconds = with of plot in seconds for labeling purposes (optional)
        @start_time is start time in seconds for the plot (optional)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount

        @topdown if true, indicates while palce first signal at top of plot, True default
        """
        data = tarray
        numSamples, numRows = tarray.shape
        # for testing
        # data = np.random.randn(numSamples,numRows) # test data
        # data.shape = numSamples, numRows
        if seconds:
            t = seconds * np.arange(numSamples, dtype=float) / numSamples
            # import pdb
            # pdb.set_trace()
            if start_time:
                t = t + start_time
                xlm = (start_time, start_time + seconds)
            else:
                xlm = (0, seconds)

        else:
            t = np.arange(numSamples, dtype=float)
            xlm = (0, numSamples)

        ticklocs = []
        if not 'width' in kwargs:
            kwargs['width'] = 950  # a default width that is wider but can just fit in jupyter
        fig = bplt.figure(tools="pan,box_zoom,reset,resize,previewsave,lasso_select",
                          **kwargs)  # subclass of Plot that simplifies plot creation

        ## xlim(*xlm)
        # xticks(np.linspace(xlm, 10))
        dmin = data.min()
        dmax = data.max()
        print("dmin, dmax = ", dmin, dmax)
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        print("dr:", dr)
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ## ylim(y0, y1)

        ticklocs = [ii * dr for ii in range(numRows)]
        # print("ticklocs:", ticklocs)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        self.ticklocs = ticklocs
        self.time = t

        ## instead build a data_dict and use datasource with multi_line
        xs = [t for ii in range(numRows)]
        ys = [yscale * data[:, ii] + ticklocs[ii] for ii in range(numRows)]
        print("creating figure lines")
        # self.data_source.data = dict(xs=xs,ys=ys)
        self.multi_line_glyph = fig.multi_line(xs=xs, ys=ys)  # , line_color='firebrick')
        self.data_source = self.multi_line_glyph.data_source

        # set the yticks to use axes coords on the y axis
        ## ax.set_yticks(ticklocs)
        # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
        if not ylabels:
            ylabels = ["%d" % ii for ii in range(numRows)]
        ylabel_dict = dict(zip(ticklocs, ylabels))
        # print('ylabel_dict:', ylabel_dict)
        fig.yaxis.ticker = FixedTicker(ticks=ticklocs)  # can also short cut to give list directly
        fig.yaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % ylabel_dict)

        return fig

    def stackplot(self, marray, seconds=None, start_time=None, ylabels=None, yscale=1.0, **kwargs):
        """
        will plot a stack of traces one above the other assuming
        @marray contains the data you want to plot
        marray.shape = numRows, numSamples

        @seconds = with of plot in seconds for labeling purposes (optional)
        @start_time is start time in seconds for the plot (optional)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount
        """
        tarray = np.transpose(marray)
        return self.stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels, yscale=yscale,
                                **kwargs)

    def show_epoch_centered(self):

        """
        @signals array-like object with signals[ch_num, sample_num]
        @goto_sec where to go in the signal to show the feature
        @page_width_sec length of the window to show in secs
        @chstart   which channel to start
        @chstop    which channel to end
        @labels_by_channel
        @yscale
        @fs sample frequency (num samples per second)
        """
        fs = self.fs
        goto_sample = int(fs * self.loc_sec)
        hw = half_width_epoch_sample = int(self.page_width_secs * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = self.ch_start, self.ch_stop

        s0 = self.limit_sample_check(goto_sample - hw)
        s1 = self.limit_sample_check(goto_sample + hw)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        return self.stackplot(self.signals[ch0:ch1, s0:s1], start_time=start_time_sec,
                              seconds=self.page_width_secs,
                              ylabels=self.ylabels[ch0:ch1], yscale=self.yscale)

    def limit_sample_check(self, x):
        """
        return a valid value for a slice into signals sample dimension
        """
        if x < 0:
            return 0

        if x > self.num_samples:
            return self.num_samples
        return x


eeg = EEGBrowser(page_width_sec=10)
eeg.create_plot()


# doc.add_root(layouts.column([buttonF,buttonB, eeg.fig]))
# doc.add_root(eeg.fig)

def forward1sec():
    eeg.loc_sec = eeg.limit_sample_check(eeg.loc_sec + 1)
    eeg.update()

def forward10():
    eeg.loc_sec = eeg.limit_sample_check(eeg.loc_sec + 10)
    # eeg.loc_sec += 10
    eeg.update()

def backward1sec():
    eeg.loc_sec = eeg.limit_sample_check(eeg.loc_sec - 1)
    eeg.update()
    
def backward10(eeg=eeg):
    eeg.loc_sec = eeg.limit_sample_check(eeg.loc_sec - 10)
    eeg.update()

buttonF = Button(label="forward 10s")
buttonF.on_click(forward10)

buttonB = Button(label="backward 10s")
buttonB.on_click(backward10)

buttonF1 = Button(label=">")
buttonF1.on_click(forward1sec)

buttonB1 = Button(label="<")
buttonB1.on_click(backward1sec)



l = layouts.layout([
    [eeg.fig],
    [buttonB,buttonB1, buttonF1, buttonF]], sizing_mode='stretch_both') # stretch_both expands to fill size of container

doc.add_root(l)
