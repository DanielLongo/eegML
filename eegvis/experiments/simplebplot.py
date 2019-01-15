# -*- coding: utf-8 *-*
import numpy as np
import bokeh.plotting as bplt

#p = bplt.figure()
#p.line([1,2,3,4,5], [6,7,2,4,5], line_width=2)

#bplt.show(p)


def stackplot_t(tarray, seconds=None, start_time=None, ylabels=None, yscale=1.0):
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
    ## ax = subplot(111)
    fig = bplt.figure() # subclass of Plot that simplifies plot creation

    ## xlim(*xlm)
    # xticks(np.linspace(xlm, 10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows - 1) * dr + dmax
    ## ylim(y0, y1)

    ticklocs = [ii*dr for ii in range(numRows)]

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    ## segs = []
    # note could also duplicate time axis then use p.multi_line
    for ii in range(numRows):
        ## segs.append(np.hstack((t[:, np.newaxis], yscale * data[:, i, np.newaxis])))
        fig.line(t[:],yscale * data[:, ii] + offsets[ii, 1] ) # adds line glyphs to figure

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
if __name__ == '__main__':
    # stackplot_t(tarray, seconds=None, start_time=None, ylabels=None, yscale=1.0)
    fig = test_stackplot_t_2()
    bplt.show(fig)