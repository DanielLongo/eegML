# -*- coding: utf-8 -*
# eeg_app2.py first try
"""
"""
from __future__ import print_function, division, unicode_literals
import functools
import os.path as path
from pprint import pprint

import pandas as pd
import numpy as np
import h5py


import bokeh.plotting as bplt
import bokeh.models
import bokeh.models.widgets
import bokeh.layouts as layouts  # column, row, ?grid
# import bokeh.models.widgets as bmw
# import bokeh.models.sources as bms
from bokeh.models import FuncTickFormatter
from bokeh.models.tickers import FixedTicker


# from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

# stuff to define new widget 
from bokeh.models import LayoutDOM
from bokeh.util.compiler import TypeScript
from bokeh.core.properties import Int # String, Instance 

DOC = curdoc() # hold on to an instance of current doc in case need multithreads
SIZING_MODE =  'fixed' # 'scale_width' also an option, 'scale_both', 'scale_width', 'scale_height', 'stretch_both'



#placeholder figure
mainfig = figure(tools="pan,box_zoom,reset,resize,previewsave,lasso_select",
                 width=1200)




desc = bokeh.models.Div(text=open(path.join(path.dirname(__file__), "description.html")).read(), width=800)

### layout ###
#
bBackward10 = bokeh.models.widgets.Button(label='<<') # ,width=1)
bBackward1 = bokeh.models.widgets.Button(label='<') 
bForward10 = bokeh.models.widgets.Button(label='>>')
bForward1 = bokeh.models.widgets.Button(label='>')

toprowctrls = [bBackward10,bBackward1,bForward1, bForward10]
bottomrowctrls = [bokeh.models.widgets.Select(title='montage',value='trace', options=['trace', 'db','tcp']),
                  bokeh.models.widgets.Button(label="spacer")]

#for control in controls:
#    control.on_change('value', lambda attr, old, new: update())



#inputs = widgetbox(*controls[:3], sizing_mode=SIZING_MODE)
wbox = functools.partial(layouts.widgetbox, sizing_mode=SIZING_MODE)
wbox20 = functools.partial(layouts.widgetbox, sizing_mode=SIZING_MODE)
toprow = layouts.row(*map(wbox20, toprowctrls))
# toprow = layouts.row(wbox(layouts.row(children=toprowctrls)))
print(toprow)
bottomrow = layouts.row(*map(wbox, bottomrowctrls))
L = layouts.layout([
    [desc],
    [toprow],
    [mainfig],
    [bottomrow]
    ], sizing_mode=SIZING_MODE)

DOC.add_root(L)
