# -*- coding: utf-8 -*-
# put code to view whole eeg files here
from __future__ import absolute_import, print_function, division, unicode_literals
from collections import OrderedDict 
import pprint

import numpy as np
import ipywidgets

import bokeh.plotting as bplt
from bokeh.models import FuncTickFormatter, Range1d
from bokeh.models.tickers import FixedTicker, SingleIntervalTicker

from . import montageview
from . import stackplot_bokeh
from .stackplot_bokeh import limit_sample_check
from bokeh.io import push_notebook

import eegml_signal.filters as esfilters

# """
# notes on setting ranges for a plot

# from bokeh.models import Range1d

# fig = make_fig()
# left, right, bottom, top = 3, 9, 4, 10
# fig.x_range=Range1d(left, right)
# fig.y_range=Range1d(bottom, top)
# show(fig)
# to update dynamically, then change fig.y_range.start = newbottom; fig.y_range.end = newtop
# """     

# this is not used yet

def setNotebookWidth100():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    np.set_printoptions(linewidth=110) # apply width to output formatting
    # cf jupyter themes https://github.com/dunovank/jupyter-themes
    # jt -t oceans16 -f roboto -fs 12 -cellw 100%
    # https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-ipython-notebook-in-my-browser
    
    
class MinimalEEGRecord:
    """
    a basic Minimal EEG/MEG signal has uniform sampling, continuous time

    essential parts:

    @signals - acts like a numpy ndarray of shape = (number_of_channels, number_of_samples)
    @sample_frequency - float in Hz sampling rate of the signal 

    optional parts: useful if you have them

    @electrode_labels
    @electrode_montage_labels
    @electrode_montage_factories
    @electrode_positions3D
    @start_dtime
    @end_dtime

    """
    def __init__(self,
                 signals,
                 sample_frequency=1.0,
                 electrode_labels=None,
                 electrode_montage_labels=None,
                 electrode_montage_factories=None,
                 electrode_positions3D=None,
                 start_dtime=None,
                 end_dtime=None):

        self.signals = signals
        self.fs = sample_frequency
        self.electrode_labels = electrode_labels
        self.montage_labels = montage_labels
        self.electrode_positions3D = electrode_positions3D
        self.start_dtime = start_dtime
        self.end_dtime = end_dtime
        
                 
                                                 
class EeghdfBrowser:
    """
    take an hdfeeg file and allow for browsing of the EEG signal

    just use the raw hdf file and conventions for now

    """
    def __init__(self, eeghdf_file, page_width_seconds=10.0, start_seconds=-1,
                 montage='trace', montage_options={},
                 yscale=1.0,
                 plot_width=950, plot_height=600):
        """
        @eegfile is an eeghdf.Eeghdf() class instance representing the file
        @montage is either a string in the standard list or a montageview factory
        @eeghdf_file - an eeghdf.Eeeghdf instance
        @page_width_seconds = how big to make the view in seconds
        @montage - montageview (class factory) OR a string that identifies a default montage (may want to change this to a factory function 
        @start_seconds - center view on this point in time

        BTW 'trace' is what NK calls its 'as recorded' montage - might be better to call 'raw'
        """

        self.eeghdf_file = eeghdf_file
        self.update_eeghdf_file(eeghdf_file, montage, montage_options)

        # display related
        self.page_width_seconds = page_width_seconds
        
        ## bokeh related 


        self.page_width_secs = page_width_seconds
        if start_seconds < 0:
            self.loc_sec = page_width_seconds / 2.0  # default location in file by default at start if possible
        else:
            self.loc_sec = start_seconds 

        # self.init_kwargs = kwargs

        # other ones
        self.yscale = yscale
        self.ui_plot_width = plot_width
        self.ui_plot_height = plot_height

        self.bk_handle = None
        self.fig = None


        self.update_title()

        self.num_rows, self.num_samples = self.signals.shape
        self.line_glyphs = [] # not used?
        self.multi_line_glyph = None


        self.ch_start = 0
        self.ch_stop = self.current_montage_instance.shape[0]

        ####### set up filter cache: first try
        self.current_hp_filter = None
        self.current_lp_filter = None
        self._highpass_cache = OrderedDict()

        self._highpass_cache['None'] = None
 
        self._highpass_cache['0.1 Hz'] = esfilters.fir_highpass_firwin_ff(self.fs, cutoff_freq=0.1,
                                                                       numtaps=int(self.fs)) 

        self._highpass_cache['0.3 Hz'] = esfilters.fir_highpass_firwin_ff(self.fs, cutoff_freq=0.3,
                                                                       numtaps=int(self.fs)) 

        #ff = esfilters.fir_highpass_remez_zerolag(fs=self.fs, cutoff_freq=1.0, transition_width=0.5, numtaps=int(2*self.fs))
        ff = esfilters.fir_highpass_firwin_ff(fs=self.fs, cutoff_freq=1.0, numtaps=int(2*self.fs))
        self._highpass_cache['1 Hz'] = ff
        #ff = esfilters.fir_highpass_remez_zerolag(fs=self.fs, cutoff_freq=5.0, transition_width=2.0, numtaps=int(0.2*self.fs))
        ff = esfilters.fir_highpass_firwin_ff(fs=self.fs, cutoff_freq=5.0, numtaps=int(0.2*self.fs))
        self._highpass_cache['5 Hz'] = ff

        firstkey = '0.3 Hz' # list(self._highpass_cache.keys())[0]
        self.current_hp_filter = self._highpass_cache[firstkey]

        
        self._lowpass_cache = OrderedDict()
        self._lowpass_cache['None'] = None
        self._lowpass_cache['15 Hz'] = esfilters.fir_lowpass_firwin_ff(fs=self.fs, cutoff_freq=15.0, numtaps=int(self.fs/2.0))
        self._lowpass_cache['30 Hz'] = esfilters.fir_lowpass_firwin_ff(fs=self.fs, cutoff_freq=30.0, numtaps=int(self.fs/4.0))
        self._lowpass_cache['50 Hz'] = esfilters.fir_lowpass_firwin_ff(fs=self.fs, cutoff_freq=50.0, numtaps=int(self.fs/4.0))
        self._lowpass_cache['70 Hz'] = esfilters.fir_lowpass_firwin_ff(fs=self.fs, cutoff_freq=70.0, numtaps=int(self.fs/4.0))
        
        self._notch_filter = esfilters.notch_filter_iir_ff(notch_freq=60.0, fs=self.fs, Q=60)
        self.current_notch_filter = None


    @property
    def signals(self):
        return self.eeghdf_file.phys_signals

    def update_eeghdf_file(self, eeghdf_file, montage='trace', montage_options={}):
        self.eeghdf_file = eeghdf_file
        hdf = eeghdf_file.hdf
        rec=hdf['record-0']
        self.fs = rec.attrs['sample_frequency']
        # self.signals = rec['signals']
        blabels = rec['signal_labels'] # byte labels
        # self.electrode_labels = [str(ss,'ascii') for ss in blabels]
        self.electrode_labels = eeghdf_file.electrode_labels
        # fill in any missing ones
        if len(self.electrode_labels) < eeghdf_file.phys_signals.shape[0]:
            d =  eeghdf_file.phys_signals.shape[0] - len(self.electrode_labels)
            ll = len(self.electrode_labels)
            suppl = [str(ii) for ii in range(ll, ll+d)]
            self.electrode_labels += suppl
            print('extending electrode lables:', suppl)
            
        # reference labels are used for montages, since this is an eeghdf file, it can provide these 

        self.ref_labels = eeghdf_file.shortcut_elabels

        if not montage_options:
            # then use builtins and/or ones in the file
            montage_options = montageview.MONTAGE_BUILTINS.copy()
            #print('starting build of montage options', montage_options)
                        
            # montage_options = eeghdf_file.get_montages()

        # defines self.current_montage_instance 
        self.current_montage_instance = None
        if type(montage) == str: # then we have some work to do
            if montage in montage_options:

                self.current_montage_instance = montage_options[montage](self.ref_labels)
            else:
                raise Exception('unrecognized montage: %s' % montage)
        else:
            if montage: # is a class 
                self.current_montage_instance = montage(self.ref_labels)
                montage_options[self.current_montage_instance.name] = montage
            else: # use default 

                self.current_montage_instance = montage_options[0](self.ref_labels)
                
        assert self.current_montage_instance
        try: # to update ui display
            self.ui_montage_dropdown.value = self.current_montage_instance.name
        except AttributeError:
            # guess is not yet instantiated
            pass

        self.montage_options = montage_options # save the montage_options for later
        self.update_title()
        # note this does not do any plotting or update the plot 

    def update_title(self):
        self.title = "hdf %s - montage: %s" % (
            self.eeghdf_file.hdf.filename,
            self.current_montage_instance.full_name if self.current_montage_instance else '')

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
            chstop=self.current_montage_instance.shape[0],
            fs=self.fs,
            ylabels=self.current_montage_instance.montage_labels,
            yscale=self.yscale,
            montage=self.current_montage_instance)
        self.fig.xaxis.axis_label = 'seconds'
        # make the xgrid mark every second 
        self.fig.xgrid.ticker =SingleIntervalTicker(interval=1.0) #  bokeh.models.tickers.SingleIntervalTicker

    def show_for_bokeh_app(self):
        """try running intside a bokeh app, so don't need notebook stuff"""
        self.plot()

    def show(self):
        self.plot()
        self.register_top_bar_ui()  # create the buttons
        self.bk_handle = bplt.show(self.fig, notebook_handle=True)
        self.register_bottom_bar_ui()

    def update(self):
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
        inmontage_view = np.dot(self.current_montage_instance.V.data, signal_view)


        data = inmontage_view[self.ch_start:self.ch_stop, :]  # note transposed
        numRows = inmontage_view.shape[0]
        ########## do filtering here ############
        # start primative filtering
        if self.current_notch_filter:
            for ii in range(numRows):
                data[ii, :] = self.current_notch_filter(data[ii, :])
                
        if self.current_hp_filter:
            for ii in range(numRows):
                data[ii,:] = self.current_hp_filter(data[ii,:])
        if self.current_lp_filter:
            for ii in range(numRows):
                data[ii,:] = self.current_lp_filter(data[ii,:])
        
        ## end filtering
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
        
        self.push_notebook()

    def push_notebook(self):
        if self.bk_handle:
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
        if not 'plot_width' in kwargs:
            kwargs['plot_width'] = self.ui_plot_width # 950  # a default width that is wider but can just fit in jupyter, not sure if plot_width is preferred
        if not 'plot_height' in kwargs:
            kwargs['plot_height'] = self.ui_plot_height

        if not self.fig:
            #print('creating figure')
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

        ########## do filtering here ############
        # start primative filtering
        # remember we are in the stackplot_t so channels and samples are flipped -- !!! eliminate this junk
        if self.current_notch_filter:
            for ii in range(numRows):
                data[ii, :] = self.current_notch_filter(data[ii, :])

        if self.current_hp_filter:
            # print("doing filtering")
            for ii in range(numRows):
                data[:,ii] = self.current_hp_filter(data[:,ii])

        if self.current_lp_filter:
            for ii in range(numRows):
                data[ii,:] = self.current_lp_filter(data[ii,:])
                
        ## end filtering


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
        inmontage_view = np.dot(self.current_montage_instance.V.data, signal_view)
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


        ylabels = self.current_montage_instance.montage_labels
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
                              topdown=True,
                              **kwargs):
        """
        plot an eeg segment using current montage, center the plot at @goto_sec
        with @page_width_sec shown

        @signals array-like object with signals[ch_num, sample_num]

        @goto_sec where to go in the signal to show the feature
        @page_width_sec length of the window to show in secs
        @chstart   which channel to start
        @chstop    which channel to end

        @fs sample frequency (num samples per second)

        @ylabels a list of labels for each row ("channel") in marray
        @yscale with increase (mutiply) the signals in each row by this amount
        @montage instance 

        """


        goto_sample = int(fs * goto_sec)
        hw = half_width_epoch_sample = int(page_width_sec * fs / 2)

        # plot epochs of width page_width_sec centered on (multiples in DE)
        ch0, ch1 = chstart, chstop

        ptepoch = int(page_width_sec * fs)

        s0 = limit_sample_check(goto_sample - hw, signals)
        s1 = limit_sample_check(goto_sample + hw, signals)
        duration_sec = (s1 - s0) / fs
        start_time_sec = s0 / fs

        # signals[ch0:ch1, s0:s1]
        signal_view = signals[:, s0:s1]
        inmontage_view = np.dot(montage.V.data, signal_view)
        rlabels = montage.montage_labels
        # topdown = True
        
        # return self.stackplot(
        #     signals[ch0:ch1, s0:s1],
        #     start_time=start_time_sec,
        #     seconds=duration_sec,
        #     ylabels=ylabels[ch0:ch1],
        #     yscale=yscale)
        ### to here start stackplot_t
        # self.stackplot_t(
        #     tarray,
        #     seconds=seconds,
        #     start_time=start_time,
        #     ylabels=ylabels,
        #     yscale=yscale,
        #     topdown=True,
        #     **kwargs)


        data = inmontage_view[chstart:chstop,:]
        numRows, numSamples = data.shape
        # data = np.random.randn(numSamples,numRows) # test data
        # data.shape =  numRows, numSamples

        t = duration_sec * np.arange(numSamples, dtype=float) / numSamples

        t = t + start_time_sec # shift over
        xlm = (start_time_sec, start_time_sec + duration_sec)

        ticklocs = []
        if not 'plot_width' in kwargs:
            kwargs['plot_width'] = self.ui_plot_width # 950  # a default width that is wider but can just fit in jupyter, not sure if plot_width is preferred
        if not 'plot_height' in kwargs:
            kwargs['plot_height'] = self.ui_plot_height

        if not self.fig:
            #print('creating figure')
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

        ########## do filtering here ############
        # start primative filtering
        # remember we are in the stackplot_t so channels and samples are flipped -- !!! eliminate this junk
        if self.current_notch_filter:
            for ii in range(numRows):
                data[ii, :] = self.current_notch_filter(data[ii, :])

        if self.current_hp_filter:
            # print("doing filtering")
            for ii in range(numRows):
                data[ii,:] = self.current_hp_filter(data[ii,:])

        if self.current_lp_filter:
            for ii in range(numRows):
                data[ii,:] = self.current_lp_filter(data[ii,:])
                
        ## end filtering


        ## instead build a data_dict and use datasource with multi_line
        xs = [t for ii in range(numRows)]
        ys = [yscale * data[ii, :] + ticklocs[ii] for ii in range(numRows)]

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
        return self.fig

        # return self.stackplot(
        #     inmontage_view[ch0:ch1,:],
        #     start_time=start_time_sec,
        #     seconds=duration_sec,
        #     ylabels=rlabels,
        #     yscale=yscale,
        #     topdown=topdown)


    def register_top_bar_ui(self):
        mlayout= ipywidgets.Layout()
        mlayout.width = '15em'
        self.ui_montage_dropdown = ipywidgets.Dropdown(
            #options={'One': 1, 'Two': 2, 'Three': 3},
            options = self.montage_options.keys(), # or .montage_optins.keys() 
            value=self.current_montage_instance.name,
            description='Montage:',
            layout = mlayout,
        )       

        def on_dropdown_change(change, parent=self):
            #print('change observed: %s' % pprint.pformat(change))
            if change['name'] == 'value': # the value changed
                if change['new'] != change['old']:
                    # print('*** should change the montage to %s from %s***' % (change['new'], change['old']))
                    parent.update_montage(change['new']) # change to the montage keyed by change['new']
                    parent.update_plot_after_montage_change()
                    parent.update() #                    
        self.ui_montage_dropdown.observe(on_dropdown_change)

        flayout= ipywidgets.Layout()
        flayout.width = '12em'
        self.ui_low_freq_filter_dropdown = ipywidgets.Dropdown(
            #options = ['None', '0.1 Hz', '0.3 Hz', '1 Hz', '5 Hz', '15 Hz', 
            #           '30 Hz', '50 Hz', '100 Hz', '150Hz'],
            options = self._highpass_cache.keys(),
            value = '0.3 Hz',
            description = 'LF',
            layout = flayout
            )

        def lf_dropdown_on_change(change, parent=self):
            #print('change observed: %s' % pprint.pformat(change))
            if change['name'] == 'value': # the value changed
                if change['new'] != change['old']:
                    # print('*** should change the filter to %s from %s***' % (change['new'], change['old']))
                    parent.current_hp_filter = parent._highpass_cache[change['new']]
                    parent.update() #                    
        self.ui_low_freq_filter_dropdown.observe(lf_dropdown_on_change)

        ###

        self.ui_high_freq_filter_dropdown = ipywidgets.Dropdown(
            #options = ['None', '15 Hz', '30 Hz', '50 Hz', '70Hz', '100 Hz', '150Hz', '300 Hz'],
            options = self._lowpass_cache.keys(),
            # value = '70Hz',
            description = 'HF',
            layout = flayout
            )
        
        def hf_dropdown_on_change(change, parent=self):
            if change['name'] == 'value': # the value changed
                if change['new'] != change['old']:
                    # print('*** should change the filter to %s from %s***' % (change['new'], change['old']))
                    self.current_lp_filter = self._lowpass_cache[change['new']]
                    self.update() #                    
        self.ui_high_freq_filter_dropdown.observe(hf_dropdown_on_change)
        
        def go_to_handler(change, parent=self):
            # print("change:", change)
            if change['name'] == 'value':
                self.loc_sec = change['new']
                self.update()

        self.ui_notch_option = ipywidgets.Checkbox(value=False, description='60Hz Notch', disabled=False)
        def notch_change(change):
            if change['name'] == 'value':
                if change['new']:
                    self.current_notch_filter = self._notch_filter
                else:
                    self.current_notch_filter = None
                self.update()
                
        self.ui_notch_option.observe(notch_change)
        
        self.ui_gain_bounded_float = ipywidgets.BoundedFloatText(
            value=1.0,
            min=0.001,
            max=1000.0,
            step=0.1,
            description='gain',
            disabled=False, 
            continuous_update=False, # only trigger when done
            layout=flayout)
        def ui_gain_on_change(change, parent=self):
            if change['name'] == 'value':
                if change['new'] != change['old']:
                    self.yscale = float(change['new'])
                    self.update()
        self.ui_gain_bounded_float.observe(ui_gain_on_change)
        
        display(ipywidgets.HBox([self.ui_montage_dropdown, 
                                 self.ui_low_freq_filter_dropdown,
                                 self.ui_high_freq_filter_dropdown,
                                 self.ui_notch_option,
                                 self.ui_gain_bounded_float
        ]))


    def register_bottom_bar_ui(self):
        self.ui_buttonf = ipywidgets.Button(description="go forward 10s")
        self.ui_buttonback = ipywidgets.Button(description="go backward 10s")
        self.ui_buttonf1 = ipywidgets.Button(description="forward 1 s")
        self.ui_buttonback1 = ipywidgets.Button(description="back 1 s")

        def go_forward(b, parent=self):
            self.loc_sec += 10
            self.update()
        self.ui_buttonf.on_click(go_forward)

        def go_backward(b):
            self.loc_sec -= 10
            self.update()
        self.ui_buttonback.on_click(go_backward)
        
        def go_forward1(b, parent=self):
            self.loc_sec += 1
            self.update()
        self.ui_buttonf1.on_click(go_forward1)
        
        def go_backward1(b, parent=self):
            self.loc_sec -= 1
            self.update()
        self.ui_buttonback1.on_click(go_backward1)

        def go_to_handler(change, parent=self):
            # print("change:", change)
            if change['name'] == 'value':
                self.loc_sec = change['new']
                self.update()

        display(ipywidgets.HBox([self.ui_buttonback, self.ui_buttonf,self.ui_buttonback1,self.ui_buttonf1]))

        
    def update_montage(self, montage_name):
        Mv = self.montage_options[montage_name]
        new_montage = Mv(self.ref_labels)
        self.current_montage_instance = new_montage
        self.ch_start = 0
        self.ch_stop = new_montage.shape[0]
        self.update_title()
        # self.fig = None # this does not work


# class Eegbrowser:
#     def __init__(self, eegfile, page_width_seconds=10.0, start_sec=0,
#                  montage='double banana'):  # may want to change default to 'trace' or 'raw'
#         """
#         @eegfile is an eeghdf.Eeghdf() class instance representing the file
#         @montage is either a string in the standard list or a montageview factory"""

#         self.eeghdf_file = eegfile
#         self.page_width_seconds = page_width_seconds
#         self.start_sec = start_sec #!! not used yet
        
#         if montage in montageview.MONTAGE_BUILTINS:
#             self.cur_montageview_factory = montageview.MONTAGE_BUILTINS[montage]
#             self.montage_options = montageview.MONTAGE_BUILTINS
#         else:
#             self.montage_options = montageview.MONTAGE_BUILTINS.copy() # ordered dict
#             self.montage_options[montage.name] = montage
#             self.cur_montageview_factory = montage

#         shortlabels = eegfile.shortcut_elabels
#         self.current_montage_instance = self.cur_montageview_factory(shortlabels)

#         self.eegplot = stackplot_bokeh.IpyHdfEegPlot2(self.eeghdf_file,
#                                                       page_width_seconds=page_width_seconds,
#                                                       montage_class=self.cur_montageview_factory,
#                                                       montage_options=self.montage_options)
        
#         self.eegplot.show()            
        

class EegBrowser(EeghdfBrowser):
    """
    take an minimal eeg file and allow for browsing of the EEG signal



    """
    def __init__(self,min_eeg, page_width_seconds, montage=None, montage_options=OrderedDict(), start_seconds=-1, **kwargs):
    #def __init__(self, eeghdf_file, page_width_seconds=10.0, start_seconds=-1,
    #             montage='trace', montage_options={}, **kwargs):  
        """
        @min_eeg file is a minimal_eeg_record class instance representing the file

        @montage is either a string in the standard list or a montageview factory
        @eeghdf_file - an eeghdf.Eeeghdf instance
        @page_width_seconds = how big to make the view in seconds
        @montage - montageview (class factory) OR a string that identifies a default montage (may want to change this to a factory function 
        @start_seconds - center view on this point in time

        BTW 'trace' is what NK calls its 'as recorded' montage - might be better to call 'raw'
        """

        self._signals = min_eeg.signals
        num_channels, num_samples = self._signals.shape
        
        # self.electrode_labels = [str(ss,'ascii') for ss in blabels]
        self.electrode_labels = min_eeg.electrode_labels
        if not self.electrode_labels:
            self.electrode_labels = [str(ii) for ii in range(num_channels)]
        self.ref_labels = montageview.standard2shortname(self.electrode_labels) 
            

        # if montage in montageview.MONTAGE_BUILTINS:
        #     self.cur_montageview_factory = montageview.MONTAGE_BUILTINS[montage]
        #     self.montage_options = montageview.MONTAGE_BUILTINS
        # else:
        #     self.montage_options = montageview.MONTAGE_BUILTINS.copy() # ordered dict
        #     self.montage_options[montage.name] = montage
        #     self.cur_montageview_factory = montage

        if min_eeg.montage_options:
            # not sure if update works here
            montage_options.update(min_eeg.montage_options)
        if not montage_options:
            # then use builtins and/or ones in the file
            montage_options = montageview.MONTAGE_BUILTINS.copy()
            #print('starting build of montage options', montage_options)
                        
            # montage_options = eeghdf_file.get_montages()

        # defines self.current_montage_instance 
        if type(montage) == str: # then we have some work to do
            if montage in montage_options:

                self.current_montage_instance = montage_options[montage](self.ref_labels)
            else:
                raise Exception('unrecognized montage: %s' % montage)
        else:
            if montage: # is a class 
                self.current_montage_instance = montage(self.ref_labels)
                montage_options[self.current_montage_instance.name] = montage
            else: # use default 

                self.current_montage_instance = montage_options[0](self.ref_labels)
                
        assert self.current_montage_instance
        self.montage_options = montage_options # save the montage_options for later

        # display related
        self.page_width_seconds = page_width_seconds
        
        ## bokeh related 


        self.page_width_secs = page_width_seconds
        if start_seconds < 0:
            self.loc_sec = page_width_seconds / 2.0  # default location in file by default at start if possible
        else:
            self.loc_sec = start_seconds 

        #self.init_kwargs = kwargs

        if 'yscale' in kwargs:
            self.yscale = kwargs['yscale']
        else:
            self.yscale = 3.0

        self.bk_handle = None
        self.fig = None
        self.fs = rec.attrs['sample_frequency']

        self.update_title()

        self.num_rows, self.num_samples = self.signals.shape
        self.line_glyphs = [] # not used?
        self.multi_line_glyph = None


        self.ch_start = 0
        self.ch_stop = self.current_montage_instance.shape[0]

        ####### set up filter cache: first try
        self.current_hp_filter = None
        self.current_lp_filter = None
        self._highpass_cache = OrderedDict()
        ff = esfilters.fir_highpass_remez_zerolag(fs=self.fs, cutoff_freq=1.0, transition_width=0.5, numtaps=int(2*self.fs))
        self._highpass_cache['None'] = None
        self._highpass_cache['1 Hz'] = ff
        ff = esfilters.fir_highpass_remez_zerolag(fs=self.fs, cutoff_freq=5.0, transition_width=2.0, numtaps=int(0.2*self.fs))
        self._highpass_cache['5 Hz'] = ff

        firstkey = '1 Hz' # list(self._highpass_cache.keys())[0]
        self.current_hp_filter = self._highpass_cache[firstkey]

        
        self._lowpass_cache = OrderedDict()
        





    @property
    def signals(self):
        return self.eeghdf_file.phys_signals

