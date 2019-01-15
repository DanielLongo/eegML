# -*- coding: utf-8 -*
"""
based stackplot using matplotlib but adapted to use bokeh
 on multilineplot example in matplotlib with MRI data (I think)
uses line collections (might actually be from pbrain example)
- clm 

Notes: can use bokeh widgets in the notebook
from bokeh.layouts... widgetbox

CustomJS Callboacks allow you to activate things in a plot but they don't call into python
(just runs in the browser side)


"""
from __future__ import division, print_function, absolute_import
from collections import OrderedDict 
import pprint

import numpy as np
import bokeh.plotting as bplt
from bokeh.models import FuncTickFormatter, Range1d
from bokeh.models.tickers import FixedTicker
from bokeh.io import push_notebook

# ipython related
from IPython.display import display
import ipywidgets # using verseion 7.0 from conda-forge

import eegvis.montageview as montageview

#p = bplt.figure()
#p.line([1,2,3,4,5], [6,7,2,4,5], line_width=2)

#bplt.show(p)

### Note from http://bokeh.pydata.org/en/latest/docs/user_guide/styling.html
# can also define the FuncTickFormat (a javascript function) from python using
# FuncTickFormatter.from_py_func(<python_function>)
# for example::
#
# def ticker():
#     return "{:.0f} + {:.2f}".format(tick, tick % 1)

# p.yaxis.formatter = FuncTickFormatter.from_py_func(ticker)


def stackplot(marray,
              seconds=None,
              start_time=None,
              ylabels=None,
              yscale=1.0,
              topdown=False,
              **kwargs):
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
    return stackplot_t(
        tarray,
        seconds=seconds,
        start_time=start_time,
        ylabels=ylabels,
        yscale=yscale,
        topdown=topdown, 
        **kwargs)


def stackplot_t(tarray,
                seconds=None,
                start_time=None,
                ylabels=None,
                yscale=1.0, topdown=False,
                **kwargs):
    """
    will plot a stack of traces one above the other assuming
    @tarray is an nd-array like object with format
    tarray.shape =  numSamples, numRows

    @seconds = with of plot in seconds for labeling purposes (optional)
    @start_time is start time in seconds for the plot (optional)

    @ylabels a list of labels for each row ("channel") in marray
    @yscale with increase (mutiply) the signals in each row by this amount
    """
    data = tarray
    numSamples, numRows = tarray.shape
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
        # a default width that is wider but can just fit in jupyter
        kwargs['width'] = 950
    fig = bplt.figure(
        tools="pan,box_zoom,reset,previewsave,lasso_select",
        **kwargs)  # subclass of Plot that simplifies plot creation

    ## xlim(*xlm)
    # xticks(np.linspace(xlm, 10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows - 1) * dr + dmax
    ## ylim(y0, y1)

    # wonder if just reverse the order of these if that takes care of topdown issue ???
    ticklocs = [ii * dr for ii in range(numRows)]
    if topdown == True:
        ticklocs.reverse()  #inplace
    # print("ticklocs:", ticklocs)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    ## segs = []
    # note could also duplicate time axis then use p.multi_line
    for ii in range(numRows):
        ## segs.append(np.hstack((t[:, np.newaxis], yscale * data[:, i, np.newaxis])))
        fig.line(t[:], yscale * data[:, ii] +
                 offsets[ii, 1])  # adds line glyphs to figure

        # print("segs[-1].shape:", segs[-1].shape)
        ##ticklocs.append(i * dr)

    ##lines = LineCollection(segs, offsets=offsets,
    #                        transOffset=None,
    #                       )

    ## ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ## ax.set_yticks(ticklocs)
    # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
    if not ylabels:
        ylabels = ["%d" % ii for ii in range(numRows)]
    ylabel_dict = dict(zip(ticklocs, ylabels))
    # print('ylabel_dict:', ylabel_dict)
    fig.yaxis.ticker = FixedTicker(
        ticks=ticklocs)  # can also short cut to give list directly
    fig.yaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick];
    """ % ylabel_dict)
    ## ax.set_yticklabels(ylabels)

    ## xlabel('time (s)')
    return fig


def test_stackplot_t_1():
    NumRows = 2

    NumSamples = 1000

    data = np.zeros((NumSamples, NumRows))
    data[:, 0] = np.random.normal(size=1000)
    data[:, 1] = 3.0 * np.random.normal(size=1000)
    fig = stackplot_t(data)
    return fig
    # bplt.show(p)


def test_stackplot_t_2():
    NumRows = 2

    NumSamples = 1000

    data = np.zeros((NumSamples, NumRows))
    data[:, 0] = np.random.normal(size=1000)
    data[:, 1] = 3.0 * np.random.normal(size=1000)
    fig = stackplot_t(data, seconds=5.0, start_time=47)
    return fig


def test_stackplot_t_3():
    NumRows = 2

    NumSamples = 1000

    data = np.zeros((NumSamples, NumRows))
    data[:, 0] = np.random.normal(size=1000)
    data[:, 1] = 3.0 * np.random.normal(size=1000)
    fig = stackplot_t(data, seconds=5.0, start_time=47, ylabels=['AAA', 'BBB'])
    return fig


def test_stacklineplot():
    numSamples, numRows = 800, 5
    data = np.random.randn(numRows, numSamples)  # test data
    return stackplot(data, 10.0)


def limit_sample_check(x, signals):
    if x < 0:
        return 0
    num_chan, chan_len = signals.shape
    if x > chan_len:
        return chan_len
    return x


def show_epoch_centered(signals,
                        goto_sec,
                        epoch_width_sec,
                        chstart,
                        chstop,
                        fs,
                        ylabels=None,
                        yscale=1.0, topdown=True):
    """
    @signals array-like object with signals[ch_num, sample_num]
    @goto_sec where to go in the signal to show the feature
    @epoch_width_sec length of the window to show in secs
    @chstart   which channel to start
    @chstop    which channel to end
    @labels_by_channel
    @yscale
    @fs sample frequency (num samples per second)
    """

    goto_sample = int(fs * goto_sec)
    hw = half_width_epoch_sample = int(epoch_width_sec * fs / 2)

    # plot epochs of width epoch_width_sec centered on (multiples in DE)
    ch0, ch1 = chstart, chstop

    ptepoch = int(epoch_width_sec * fs)  # pts per epoch
    dp = int(0.5 * ptepoch)
    s0 = limit_sample_check(goto_sample - hw, signals)
    s1 = limit_sample_check(goto_sample + hw, signals)
    duration = (s1 - s0) / fs
    start_time_sec = s0 / fs

    return stackplot(
        signals[ch0:ch1, s0:s1],
        start_time=start_time_sec,
        seconds=duration,
        ylabels=ylabels[ch0:ch1],
        yscale=yscale, topdown=topdown)


def show_montage_centered(signals,
                          goto_sec,
                          epoch_width_sec,
                          chstart,
                          chstop,
                          fs,
                          ylabels=None,
                          yscale=1.0,
                          montage=None,
                          topdown=True):
    """
    @signals array-like object with signals[ch_num, sample_num]
    @goto_sec where to go in the signal to show the feature
    @epoch_width_sec length of the window to show in secs
    @chstart   which channel to start
    @chstop    which channel to end
    @labels_by_channel
    @yscale
    @fs sample frequency (num samples per second)

    @topdown [=True] determines that the first element of the montage is plotted at the top of the plot
    """

    goto_sample = int(fs * goto_sec)
    hw = half_width_epoch_sample = int(epoch_width_sec * fs / 2)

    # plot epochs of width epoch_width_sec centered on (multiples in DE)
    ch0, ch1 = chstart, chstop

    ptepoch = int(epoch_width_sec * fs)  # pts per epoch
    dp = int(0.5 * ptepoch)
    s0 = limit_sample_check(goto_sample - hw, signals)
    s1 = limit_sample_check(goto_sample + hw, signals)
    duration = (s1 - s0) / fs
    start_time_sec = s0 / fs
    # signals[ch0:ch1, s0:s1]
    signal_view = signals[:, s0:s1]
    inmontage_view = np.dot(montage.V.data, signal_view)
    rlabels = montage.montage_labels
    return stackplot(
        inmontage_view,
        start_time=start_time_sec,
        seconds=duration,
        ylabels=rlabels,
        yscale=yscale, topdown=topdown)


class IpyStackplot:
    """
    work in jupyter notebook
    given an hdf @signal array-like object
    allow:
       - scrolling 
       - goto
       ? filtering
       ? montaging (linear combinations)

    use push_notebook(handle=self.bkhandle)
    question: explicit DataSource ? vs use implicit datasource in line or multi_line glyph
       r1 = fig.line(xarr, yarr)
       r.data['x'] = new_xarr

    """

    def __init__(self,
                 signals,
                 page_width_seconds,
                 ylabels,
                 fs,
                 showchannels='all',
                 yscale=3.0,
                 **kwargs):
        """
        showchannels (start,end) given a range of channels might extend later to be some sort of slice
        """
        self.signals = signals
        self.page_width_secs = page_width_seconds
        self.ylabels = ylabels
        self.yscale = yscale
        self.init_kwargs = kwargs
        self.bk_handle = None
        self.fig = None
        self.fs = fs
        self.loc_sec = page_width_seconds / 2.0  # default start for current location
        if showchannels == 'all':
            self.ch_start = 0  # change this to a list of channels for fancy slicing
            self.ch_stop = signals.shape[0]
        else:
            self.ch_start, self.ch_stop = showchannels
            
        self.num_rows, self.num_samples = signals.shape
        self.line_glyphs = []
        self.multi_line_glyph = None

    def plot(self):
        self.fig = self.show_epoch_centered(
            self.signals,
            self.loc_sec,
            page_width_sec=self.page_width_secs,
            chstart=self.ch_start,
            chstop=self.ch_stop,
            fs=self.fs,
            ylabels=self.ylabels,
            yscale=self.yscale)
        self.fig.xaxis.axis_label = 'seconds'

    def show(self):
        self.plot()
        self.register_ui()  # create the buttons
        self.bk_handle = bplt.show(self.fig, notebook_handle=True)

    def update(self, goto_sec=None):



        goto_sample = int(self.fs * self.loc_sec)
        page_width_samples = int(self.page_width_secs * self.fs)
        hw = half_width_epoch_sample = int(page_width_samples / 2)

        s0 = limit_sample_check(goto_sample - hw, self.signals)
        s1 = limit_sample_check(goto_sample + hw, self.signals)
        window_samples = s1 - s0
        duration = (s1 - s0) / self.fs
        start_time_sec = s0 / self.fs

        signal_view = self.signals[:, s0:s1]  # note transposed
        inmontage_view = np.dot(self.current_montage.V.data, signal_view)
        numRows = inmontage_view.shape[0]  # this is a silly way to get this number

        t = self.page_width_secs * np.arange(
            window_samples, dtype=float) / window_samples
        t = t + s0 / self.fs  # t = t + start_time
        # t = t[:s1-s0]
        ## this is not quite right if ch_start is not 0
        xs = [t for ii in range(numRows)]
        ys = [
            self.yscale * inmontage_view[ii, :] + self.ticklocs[ii]
            for ii in range(numRows)
        ]
        # need to decide what to do if number of channels change 

        # for ii in range(len(lg)):
        #     lg[ii].data_source.data["x"] = t
        #     lg[ii].data_source.data['y'] = self.yscale * data[ii,:] + self.ticklocs[ii]
        #     # fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to fig

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        self.data_source.data['xs'] = xs
        self.data_source.data['ys'] = ys
        push_notebook(handle=self.bk_handle)

    def stackplot_t(self,
                    tarray,
                    seconds=None,
                    start_time=None,
                    ylabels=None,
                    yscale=1.0,
                    topdown=True, # true for this one
                    **kwargs):
        """
        will plot a stack of traces one above the other assuming
        @tarray is an nd-array like object with format
        tarray.shape =  numSamples, numRows

        @seconds = with of plot in seconds for labeling purposes (optional)
        @start_time is start time in seconds for the plot (optional)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount
        """
        data = tarray
        numSamples, numRows = tarray.shape
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
        fig = bplt.figure(
            tools="pan,box_zoom,reset,previewsave,lasso_select",
            **kwargs)  # subclass of Plot that simplifies plot creation

        ## xlim(*xlm)
        # xticks(np.linspace(xlm, 10))
        dmin = float(data.min())
        dmax = float(data.max())

        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ## ylim(y0, y1)

        ticklocs = [ii * dr for ii in range(numRows)]
        # print("ticklocs:", ticklocs)
        if topdown == True:
            ticklocs.reverse()  #inplace


        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        self.ticklocs = ticklocs
        self.time = t
        ## segs = []
        # note could also duplicate time axis then use p.multi_line
        # line_glyphs = []
        # for ii in range(numRows):
        #     ## segs.append(np.hstack((t[:, np.newaxis], yscale * data[:, i, np.newaxis])))
        #     line_glyphs.append(
        #         fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to figure
        #     )

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        # self.line_glyphs = line_glyphs

        ## instead build a data_dict and use datasource with multi_line
        xs = [t for ii in range(numRows)]
        ys = [yscale * data[:, ii] + ticklocs[ii] for ii in range(numRows)]

        self.multi_line_glyph = fig.multi_line(
            xs=xs, ys=ys)  # , line_color='firebrick')
        self.data_source = self.multi_line_glyph.data_source

        # set the yticks to use axes coords on the y axis
        ## ax.set_yticks(ticklocs)
        # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
        if not ylabels:
            ylabels = ["%d" % ii for ii in range(numRows)]
        ylabel_dict = dict(zip(ticklocs, ylabels))
        # print('ylabel_dict:', ylabel_dict)
        fig.yaxis.ticker = FixedTicker(
            ticks=ticklocs)  # can also short cut to give list directly
        fig.yaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % ylabel_dict)
        ## ax.set_yticklabels(ylabels)

        ## xlabel('time (s)')
        return fig

    def stackplot(self,
                  marray,
                  seconds=None,
                  start_time=None,
                  ylabels=None,
                  yscale=1.0,
                  topdown=True, #true for this?
                  **kwargs):
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
        return self.stackplot_t(
            tarray,
            seconds=seconds,
            start_time=start_time,
            ylabels=ylabels,
            yscale=yscale,
            **kwargs)

    def show_epoch_centered(self,
                            signals,
                            goto_sec,
                            page_width_sec,
                            chstart,
                            chstop,
                            fs,
                            ylabels=None,
                            yscale=1.0):
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

        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        return self.stackplot(
            signals[ch0:ch1, s0:s1],
            start_time=start_time_sec,
            seconds=duration,
            ylabels=ylabels[ch0:ch1],
            yscale=yscale, topdown=True)

    ## ipython widget callbacks
    def register_ui(self):
        self.buttonf = ipywidgets.Button(description="go forward 10s")
        self.buttonback = ipywidgets.Button(description="go backward 10s")

        # self.floattext_goto = ipywidgets.BoundedFloatText(value=0.0, min=0, max=inr.num_samples/inr.fs, step=1, description='goto time(sec)')

        def go_forward(b):
            self.loc_sec += 10
            self.update()

        def go_backward(b):
            self.loc_sec -= 10
            self.update()

        def go_to_handler(change):
            # print("change:", change)
            if change['name'] == 'value':
                self.loc_sec = change['new']
                self.update()

        self.buttonf.on_click(go_forward)
        self.buttonback.on_click(go_backward)
        display(ipywidgets.HBox([self.buttonback, self.buttonf]))


class IpyEEGPlot:
    """
    work in jupyter notebook
    given an hdf @signal array-like object
    allow:
       - scrolling 
       - goto
       ? filtering
       - montaging (linear combinations)

    use push_notebook(handle=self.bkhandle)
    question: explicit DataSource ? vs use implicit datasource in line or multi_line glyph
       r1 = fig.line(xarr, yarr)
       r.data['x'] = new_xarr

    """

    def __init__(self,
                 signals,
                 page_width_seconds,
                 electrode_labels,
                 fs,
                 showchannels='all', # will depend on montage(s)
                 yscale=3.0,
                 montage=None,
                 **kwargs):
        self.title='' # init 
        self.signals = signals
        self.page_width_secs = page_width_seconds
        self.elabels = montageview.standard2shortname(electrode_labels)
        self.yscale = yscale
        self.init_kwargs = kwargs
        self.bk_handle = None
        self.fig = None
        self.fs = fs
        self.loc_sec = page_width_seconds / 2.0  # default start for current location
        if showchannels=='all':
            self.ch_start = 0  # change this to a list of channels for fancy slicing
            if montage:
                self.ch_stop = montage.shape[0] # all the channels in the montage
            self.ch_stop = signals.shape[0] # all the channels in the original signal 
        else:
            self.ch_start, self.ch_stop = showchannels 
        self.num_rows, self.num_samples = signals.shape
        self.line_glyphs = []
        self.multi_line_glyph = None

        self.all_montages = []
        if not montage:         # now define the default montage 'trace' and add it to the list of montages
            self.current_montage = montageview.MontageView(self.elabels,self.elabels, name='trace')
            self.current_montage.V.data = np.eye(self.num_rows)
            
        else:
            self.current_montage = montage 
        
        self.all_montages.append(self.current_montage)

    def plot(self):

        self.fig = self.show_montage_centered(
            self.signals,
            self.loc_sec,
            page_width_sec=self.page_width_secs,
            chstart=self.ch_start,
            chstop=self.ch_stop,
            fs=self.fs,
            ylabels=self.elabels,
            yscale=self.yscale,
            montage=self.current_montage)
        self.fig.xaxis.axis_label = 'seconds'

    def show(self):
        self.plot()
        self.register_ui()  # create the buttons
        self.bk_handle = bplt.show(self.fig, notebook_handle=True)

    def update(self, goto_sec=None):

        numRows = len(self.ticklocs)  # this is a silly way to get this number

        goto_sample = int(self.fs * self.loc_sec)
        page_width_samples = int(self.page_width_secs * self.fs)
        hw = half_width_epoch_sample = int(page_width_samples / 2)

        s0 = limit_sample_check(goto_sample - hw, self.signals)
        s1 = limit_sample_check(goto_sample + hw, self.signals)
        window_samples = s1 - s0
        signal_view = self.signals[:, s0:s1]
        inmontage_view = np.dot(self.current_montage.V.data, signal_view)


        data = inmontage_view[self.ch_start:self.ch_stop, :]  # note transposed
        t = self.page_width_secs * np.arange(
            window_samples, dtype=float) / window_samples
        t = t + s0 / self.fs  # t = t + start_time
        # t = t[:s1-s0]
        ## this is not quite right if ch_start is not 0
        xs = [t for ii in range(numRows)]
        ys = [
            self.yscale * data[ii, :] + self.ticklocs[ii]
            for ii in range(numRows)
        ]

        # for ii in range(len(lg)):
        #     lg[ii].data_source.data["x"] = t
        #     lg[ii].data_source.data['y'] = self.yscale * data[ii,:] + self.ticklocs[ii]
        #     # fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to fig

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        self.data_source.data['xs'] = xs
        self.data_source.data['ys'] = ys
        push_notebook(handle=self.bk_handle)

    def stackplot_t(self,
                    tarray,
                    seconds=None,
                    start_time=None,
                    ylabels=None,
                    yscale=1.0,
                    topdown=True,
                    **kwargs):
        """
        will plot a stack of traces one above the other assuming
        @tarray is an nd-array like object with format
        tarray.shape =  numSamples, numRows

        @seconds = with of plot in seconds for labeling purposes (optional)
        @start_time is start time in seconds for the plot (optional)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount
        """
        data = tarray
        numSamples, numRows = tarray.shape
        # data = np.random.randn(numSamples,numRows) # test data
        # data.shape = numSamples, numRows
        if seconds:
            t = seconds * np.arange(numSamples, dtype=float) / numSamples

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
            kwargs[
                'width'] = 950  # a default width that is wider but can just fit in jupyter
        fig = bplt.figure(title=self.title,
            tools="pan,box_zoom,reset,previewsave,lasso_select,ywheel_zoom",
            **kwargs)  # subclass of Plot that simplifies plot creation

        ## xlim(*xlm)
        # xticks(np.linspace(xlm, 10))
        dmin = data.min()
        dmax = data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ## ylim(y0, y1)

        ticklocs = [ii * dr for ii in range(numRows)]
        if topdown == True:
            ticklocs.reverse()  #inplace

        # print("ticklocs:", ticklocs)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        self.ticklocs = ticklocs
        self.time = t
        ## segs = []
        # note could also duplicate time axis then use p.multi_line
        # line_glyphs = []
        # for ii in range(numRows):
        #     ## segs.append(np.hstack((t[:, np.newaxis], yscale * data[:, i, np.newaxis])))
        #     line_glyphs.append(
        #         fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to figure
        #     )

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        # self.line_glyphs = line_glyphs

        ## instead build a data_dict and use datasource with multi_line
        xs = [t for ii in range(numRows)]
        ys = [yscale * data[:, ii] + ticklocs[ii] for ii in range(numRows)]

        self.multi_line_glyph = fig.multi_line(
            xs=xs, ys=ys)  # , line_color='firebrick')
        self.data_source = self.multi_line_glyph.data_source

        # set the yticks to use axes coords on the y axis
        ## ax.set_yticks(ticklocs)
        # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
        if not ylabels:
            ylabels = ["%d" % ii for ii in range(numRows)]
        ylabel_dict = dict(zip(ticklocs, ylabels))
        # print('ylabel_dict:', ylabel_dict)
        fig.yaxis.ticker = FixedTicker(
            ticks=ticklocs)  # can also short cut to give list directly
        fig.yaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % ylabel_dict)
        ## ax.set_yticklabels(ylabels)

        ## xlabel('time (s)')
        return fig

    def stackplot(self,
                  marray,
                  seconds=None,
                  start_time=None,
                  ylabels=None,
                  yscale=1.0,
                  topdown=True,
                  **kwargs):
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
        return self.stackplot_t(
            tarray,
            seconds=seconds,
            start_time=start_time,
            ylabels=ylabels,
            yscale=yscale,
            topdown=True,
            **kwargs)

    def show_epoch_centered(self,
                            signals,
                            goto_sec,
                            page_width_sec,
                            chstart,
                            chstop,
                            fs,
                            ylabels=None,
                            yscale=1.0):
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

        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        return self.stackplot(
            signals[ch0:ch1, s0:s1],
            start_time=start_time_sec,
            seconds=duration,
            ylabels=ylabels[ch0:ch1],
            yscale=yscale)

    def show_montage_centered(self,
                              signals,
                              goto_sec,
                              page_width_sec,
                              chstart,
                              chstop,
                              fs,
                              ylabels=None,
                              yscale=1.0,
                              montage=None,
                              topdown=True):
        """
        @signals array-like object with signals[ch_num, sample_num]
        @montage object
        @goto_sec where to go in the signal to show the feature
        @page_width_sec length of the window to show in secs
        @chstart   which channel to start
        @chstop    which channel to end
        @labels_by_channel
        @yscale
        @fs sample frequency (num samples per second)
        """

        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        # signals[ch0:ch1, s0:s1]
        signal_view = signals[:, s0:s1]
        inmontage_view = np.dot(montage.V.data, signal_view)

        rlabels = montage.montage_labels

        return self.stackplot(
            inmontage_view[ch0:ch1,:],
            start_time=start_time_sec,
            seconds=duration,
            ylabels=rlabels,
            yscale=yscale,
            topdown=topdown)

    ## ipython widget callbacks
    def register_ui(self):
        self.buttonf = ipywidgets.Button(description="go forward 10s")
        self.buttonback = ipywidgets.Button(description="go backward 10s")

        # self.floattext_goto = ipywidgets.BoundedFloatText(value=0.0, min=0, max=inr.num_samples/inr.fs, step=1, description='goto time(sec)')

        def go_forward(b):
            self.loc_sec += 10
            self.update()

        def go_backward(b):
            self.loc_sec -= 10
            self.update()

        def go_to_handler(change):
            # print("change:", change)
            if change['name'] == 'value':
                self.loc_sec = change['new']
                self.update()

        self.buttonf.on_click(go_forward)
        self.buttonback.on_click(go_backward)
        display(ipywidgets.HBox([self.buttonback, self.buttonf]))


class IpyHdfEegPlot(IpyEEGPlot):
    """
    take an hdfeeg file and allow for browsing of the EEG signal

    just use the raw hdf file and conventions for now
    """

    def __init__(self,hdf, page_width_seconds, montage=None,**kwargs):
        rec=hdf['record-0']
        self.signals = rec['signals']
        blabels = rec['signal_labels'] # byte labels
        self.electrode_labels = [str(ss,'ascii') for ss in blabels]
        self.ref_labels = montageview.standard2shortname(self.electrode_labels)
        super().__init__(self.signals,page_width_seconds, electrode_labels=self.electrode_labels,fs=rec.attrs['sample_frequency'], montage=montage,**kwargs)
        self.title = "hdf %s - montage: %s" % (hdf.filename, self.current_montage.full_name if  self.current_montage else '')



class IpyHdfEegPlot2:
    """
    take an hdfeeg file and allow for browsing of the EEG signal

    just use the raw hdf file and conventions for now

    """
    # !!! probably should refactor to another file given this depends on a specific file format 
    def __init__(self,eeghdf_file, page_width_seconds, montage_class=None, montage_options={}, start_seconds=-1, **kwargs):
        """
        @eeghdf_file - an eeghdf.Eeeghdf instance
        @page_width_seconds = how big to make the view in seconds
        @montage_class - montageview (class factory) OR a string that identifies a default montage (may want to change this to a factory function 
        @start_seconds - center view on this point in time
        """
        self.eeghdf_file = eeghdf_file
        hdf = eeghdf_file.hdf
        rec=hdf['record-0']
        self.signals = rec['signals']
        blabels = rec['signal_labels'] # byte labels
        # self.electrode_labels = [str(ss,'ascii') for ss in blabels]
        self.electrode_labels = eeghdf_file.electrode_labels
        self.ref_labels = montageview.standard2shortname(self.electrode_labels) # reference labels are used for montages 
        self.fig =None 
        self.page_width_secs = page_width_seconds
        if start_seconds < 0:
            self.loc_sec = page_width_seconds / 2.0  # default location in file by default at start if possible
        else:
            self.loc_sec = start_seconds 
        self.elabels = self.ref_labels
        self.init_kwargs = kwargs

        if 'yscale' in kwargs:
            self.yscale = kwargs['yscale']
        else:
            self.yscale = 3.0

        self.bk_handle = None
        self.fig = None
        self.fs = rec.attrs['sample_frequency']

        # defines self.current_montage 
        if type(montage_class) == str: # then we have some work to do
            if montage_class in montage_options:
                self.current_montage = montage_options[montage]
            else:
                raise Exception('unrecognized montage: %s' % montage_class)
        else:
            if montage_class:
                self.current_montage = montage_class(self.ref_labels)
                montage_options[self.current_montage.name] = montage_class
            else: # use default 
                montage_options = montageview.MONTAGE_BUILTINS
                self.current_montage = montage_options[0](self.ref_labels)
                
        assert self.current_montage
        self.montage_options = montage_options
        self.update_title()

        self.num_rows, self.num_samples = self.signals.shape
        self.line_glyphs = [] # not used?
        self.multi_line_glyph = None


        self.ch_start = 0
        self.ch_stop = self.current_montage.shape[0]


    def update_title(self):
        self.title = "hdf %s - montage: %s" % (self.eeghdf_file.hdf.filename, self.current_montage.full_name if  self.current_montage else '')

#     def __init__(self,
#                  signals,
#                  page_width_seconds,
#                  electrode_labels,
#                  fs,
#                  showchannels='all', # will depend on montage(s)
#                  yscale=3.0,
#                  montage=None,
#                  **kwargs):

#         self.yscale = yscale
#         self.init_kwargs = kwargs
#         self.bk_handle = None
#         self.fig = None
#         self.fs = fs
#         self.loc_sec = page_width_seconds / 2.0  # default start for current location
#         if showchannels=='all':
#             self.ch_start = 0  # change this to a list of channels for fancy slicing
#             if montage:
#                 self.ch_stop = montage.shape[0] # all the channels in the montage
#             self.ch_stop = signals.shape[0] # all the channels in the original signal 
#         else:
#             self.ch_start, self.ch_stop = showchannels 
#         self.num_rows, self.num_samples = signals.shape
#         self.line_glyphs = []
#         self.multi_line_glyph = None


    def plot(self):

        self.fig = self.show_montage_centered(
            self.signals,
            self.loc_sec,
            page_width_sec=self.page_width_secs,
            chstart=0,
            chstop=self.current_montage.shape[0],
            fs=self.fs,
            ylabels=self.current_montage.montage_labels,
            yscale=self.yscale,
            montage=self.current_montage)
        self.fig.xaxis.axis_label = 'seconds'

    def show(self):
        self.plot()
        self.register_ui()  # create the buttons
        self.bk_handle = bplt.show(self.fig, notebook_handle=True)

    def update(self, goto_sec=None):
        """
        updates the data in the plot and does push_notebook
        so that it will show up
        """
        goto_sample = int(self.fs * self.loc_sec)
        page_width_samples = int(self.page_width_secs * self.fs)
        hw = half_width_epoch_sample = int(page_width_samples / 2)
        s0 = limit_sample_check(goto_sample - hw, self.signals)
        s1 = limit_sample_check(goto_sample + hw, self.signals)
        window_samples = s1 - s0
        signal_view = self.signals[:, s0:s1]
        inmontage_view = np.dot(self.current_montage.V.data, signal_view)


        data = inmontage_view[self.ch_start:self.ch_stop, :]  # note transposed
        numRows = inmontage_view.shape[0]
        t = self.page_width_secs * np.arange(
            window_samples, dtype=float) / window_samples
        t = t + s0 / self.fs  # t = t + start_time
        # t = t[:s1-s0]
        ## this is not quite right if ch_start is not 0
        xs = [t for ii in range(numRows)]
        ys = [
            self.yscale * data[ii, :] + self.ticklocs[ii]
            for ii in range(numRows)
        ]
        # print('len(xs):', len(xs), 'len(ys):', len(ys))

        # is this the best way to update the data? should it be done both at once
        # {'xs':xs, 'ys':ys}
        self.data_source.data.update(dict(xs=xs, ys=ys)) # could just use equals?
        # old way 
        #self.data_source.data['xs'] = xs
        #self.data_source.data['ys'] = ys
        
        push_notebook(handle=self.bk_handle)

    def stackplot_t(self,
                    tarray,
                    seconds=None,
                    start_time=None,
                    ylabels=None,
                    yscale=1.0,
                    topdown=True,
                    **kwargs):
        """
        will plot a stack of traces one above the other assuming
        @tarray is an nd-array like object with format
        tarray.shape =  numSamples, numRows

        @seconds = with of plot in seconds for labeling purposes (optional)
        @start_time is start time in seconds for the plot (optional)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount
        """
        data = tarray
        numSamples, numRows = tarray.shape
        # data = np.random.randn(numSamples,numRows) # test data
        # data.shape = numSamples, numRows
        if seconds:
            t = seconds * np.arange(numSamples, dtype=float) / numSamples

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
            kwargs[
                'width'] = 950  # a default width that is wider but can just fit in jupyter
        if not self.fig:
            print('creating figure')
            fig = bplt.figure(title=self.title,
                tools="pan,box_zoom,reset,previewsave,lasso_select,ywheel_zoom",
                **kwargs)  # subclass of Plot that simplifies plot creation
            self.fig = fig
        
        

        ## xlim(*xlm)
        # xticks(np.linspace(xlm, 10))
        dmin = data.min()
        dmax = data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ## ylim(y0, y1)

        ticklocs = [ii * dr for ii in range(numRows)]
        bottom = -dr/0.7 
        top = (numRows-1) * dr + dr/0.7
        self.y_range = Range1d(bottom, top)
        self.fig.y_range = self.y_range

        if topdown == True:
            ticklocs.reverse()  #inplace

        
        # print("ticklocs:", ticklocs)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        self.ticklocs = ticklocs
        self.time = t
        ## segs = []
        # note could also duplicate time axis then use p.multi_line
        # line_glyphs = []
        # for ii in range(numRows):
        #     ## segs.append(np.hstack((t[:, np.newaxis], yscale * data[:, i, np.newaxis])))
        #     line_glyphs.append(
        #         fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to figure
        #     )

        #     # print("segs[-1].shape:", segs[-1].shape)
        #     ##ticklocs.append(i * dr)
        # self.line_glyphs = line_glyphs

        ## instead build a data_dict and use datasource with multi_line
        xs = [t for ii in range(numRows)]
        ys = [yscale * data[:, ii] + ticklocs[ii] for ii in range(numRows)]

        self.multi_line_glyph = self.fig.multi_line(
                xs=xs, ys=ys)  # , line_color='firebrick')
        self.data_source = self.multi_line_glyph.data_source

        # set the yticks to use axes coords on the y axis
        if not ylabels:
            ylabels = ["%d" % ii for ii in range(numRows)]
        ylabel_dict = dict(zip(ticklocs, ylabels))
        # print('ylabel_dict:', ylabel_dict)
        self.fig.yaxis.ticker = FixedTicker(
            ticks=ticklocs)  # can also short cut to give list directly
        self.fig.yaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % ylabel_dict)
        ## ax.set_yticklabels(ylabels)

        ## xlabel('time (s)')


        return self.fig


    def update_plot_after_montage_change(self):
        
        self.fig.title.text = self.title
        goto_sample = int(self.fs * self.loc_sec)
        page_width_samples = int(self.page_width_secs * self.fs)
        
        hw = half_width_epoch_sample = int(page_width_samples / 2)
        s0 = limit_sample_check(goto_sample - hw, self.signals)
        s1 = limit_sample_check(goto_sample + hw, self.signals)

        window_samples = s1 - s0
        signal_view = self.signals[:, s0:s1]
        inmontage_view = np.dot(self.current_montage.V.data, signal_view)
        self.ch_start = 0
        self.ch_stop = inmontage_view.shape[0]
        
        numRows = inmontage_view.shape[0] # ???
        # print('numRows: ', numRows)
        
        data = inmontage_view[self.ch_start:self.ch_stop, :]  # note transposed
        # really just need to reset the labels

        ticklocs = []

        ## xlim(*xlm)
        # xticks(np.linspace(xlm, 10))
        dmin = data.min()
        dmax = data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ## ylim(y0, y1)

        ticklocs = [ii * dr for ii in range(numRows)]
        ticklocs.reverse()  #inplace
        bottom = -dr/0.7 
        top = ( numRows-1) * dr + dr/0.7
        self.y_range.start = bottom
        self.y_range.end = top
        #self.fig.y_range = Range1d(bottom, top)

        # print("ticklocs:", ticklocs)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        self.ticklocs = ticklocs
        # self.time = t


        ylabels = self.current_montage.montage_labels
        ylabel_dict = dict(zip(ticklocs, ylabels))
        # print('ylabel_dict:', ylabel_dict)
        self.fig.yaxis.ticker = FixedTicker(
            ticks=ticklocs)  # can also short cut to give list directly
        self.fig.yaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % ylabel_dict)

        ## experiment with clearing the data source
        # self.data_source.data.clear() # vs .update() ???
        
        


    def stackplot(self,
                  marray,
                  seconds=None,
                  start_time=None,
                  ylabels=None,
                  yscale=1.0,
                  topdown=True,
                  **kwargs):
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
        return self.stackplot_t(
            tarray,
            seconds=seconds,
            start_time=start_time,
            ylabels=ylabels,
            yscale=yscale,
            topdown=True,
            **kwargs)

    def show_epoch_centered(self,
                            signals,
                            goto_sec,
                            page_width_sec,
                            chstart,
                            chstop,
                            fs,
                            ylabels=None,
                            yscale=1.0):
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

        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        return self.stackplot(
            signals[ch0:ch1, s0:s1],
            start_time=start_time_sec,
            seconds=duration,
            ylabels=ylabels[ch0:ch1],
            yscale=yscale)

    def show_montage_centered(self,
                              signals,
                              goto_sec,
                              page_width_sec,
                              chstart,
                              chstop,
                              fs,
                              ylabels=None,
                              yscale=1.0,
                              montage=None,
                              topdown=True):
        """
        @signals array-like object with signals[ch_num, sample_num]
        @montage object
        @goto_sec where to go in the signal to show the feature
        @page_width_sec length of the window to show in secs
        @chstart   which channel to start
        @chstop    which channel to end
        @labels_by_channel
        @yscale
        @fs sample frequency (num samples per second)
        """

        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration = (s1 - s0) / fs
        start_time_sec = s0 / fs

        # signals[ch0:ch1, s0:s1]
        signal_view = signals[:, s0:s1]
        inmontage_view = np.dot(montage.V.data, signal_view)

        rlabels = montage.montage_labels

        return self.stackplot(
            inmontage_view[ch0:ch1,:],
            start_time=start_time_sec,
            seconds=duration,
            ylabels=rlabels,
            yscale=yscale,
            topdown=topdown)



    def register_ui(self):
        self.buttonf = ipywidgets.Button(description="go forward 10s")
        self.buttonback = ipywidgets.Button(description="go backward 10s")
        self.montage_dropdown = ipywidgets.Dropdown(
            #options={'One': 1, 'Two': 2, 'Three': 3},
            options = self.montage_options.keys(), # or .montage_optins.keys() 
            value=self.current_montage.name,
            description='Montage:',
        )                        
        def go_forward(b):
            self.loc_sec += 10
            self.update()

        def go_backward(b):
            self.loc_sec -= 10
            self.update()

        def go_to_handler(change):
            # print("change:", change)
            if change['name'] == 'value':
                self.loc_sec = change['new']
                self.update()

        def on_dropdown_change(change):
            #print('change observed: %s' % pprint.pformat(change))
            if change['name'] == 'value': # the value changed
                if change['new'] != change['old']:
                    # print('*** should change the montage to %s from %s***' % (change['new'], change['old']))
                    self.update_montage(change['new']) # change to the montage keyed by change['new']
                    self.update_plot_after_montage_change()
                    self.update() #                    



        self.buttonf.on_click(go_forward)
        self.buttonback.on_click(go_backward)
        self.montage_dropdown.observe(on_dropdown_change)
        
        display(ipywidgets.HBox([self.buttonback, self.buttonf, self.montage_dropdown]))

    def update_montage(self, montage_name):
        Mv = self.montage_options[montage_name]
        new_montage = Mv(self.elabels)
        self.current_montage = new_montage
        self.ch_start = 0
        self.ch_stop = new_montage.shape[0]
        self.update_title()
        # self.fig = None # this does not work



if __name__ == '__main__':
    # stackplot_t(tarray, seconds=None, start_time=None, ylabels=None, yscale=1.0)
    fig = test_stackplot_t_3()
    bplt.show(fig)


