# -*- encoding: utf-8 -*-
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
#from bokeh.core.properties import Int # String, Instance
import bokeh.core.properties as properties # import Int # String, Instance 

import eeghdf
import eegvis.nb_eegview

ARCHIVEDIR = r'../../eeg-hdfstorage/data/'
#EEGFILE = ARCHIVEDIR + 'spasms.eeghdf'
EEGFILE = ARCHIVEDIR + 'absence_epilepsy.eeghdf'
hf = eeghdf.Eeghdf(EEGFILE)

eegbrow = eegvis.nb_eegview.EeghdfBrowser(hf, montage='double banana', start_seconds=1385, plot_width=1024, plot_height=700)
eegbrow.show_for_bokeh_app()
## set up some synthetic data

# N = 200
# x = np.linspace(0, 4*np.pi, N)
# y = np.sin(x)
# source = bokeh.models.ColumnDataSource(data=dict(x=x, y=y))

class KeyModResponder(LayoutDOM):
    """capture all aspects of keydown/up events"""
    #__implementation__ = TypeScript(KEYBOARDRESPONDERCODE_TS)
    __implementation__ = "keymodresponder.ts"
    # can use # __css__ = '<a css file.css>'
    # can use # __javascript__ = 'katex.min.js' # for additional javascript
    # this should match with javascript/typescript implementation 
    key = properties.String(default="")
    keyCode = properties.Int(default=0) 
    altKey = properties.Bool(default=False)
    ctrlKey = properties.Bool(default=False)
    metaKey = properties.Bool(default=False)
    shiftKey = properties.Bool(default=False)
    
    key_num_presses = properties.Int(default=0)
    
    keypress_callback = properties.Instance(bokeh.models.callbacks.Callback,
                                   help=""" A callback to run in the browser whenever a key is pressed
                                   """)

    # not sure how to do this to make it so could call
    # def __init__(self, parent=None):  # by default will attach to top level document
    #    """km = KeyModResonder(parent=containerdiv)"""  # !! don't know how to implement
    
keyboard = KeyModResponder()
keyboard.css_classes = ['keyboard']
callback_keyboard = bokeh.models.callbacks.CustomJS(
    args=dict(keyboard=keyboard, code="""
    console.log('in callback_keyboard')
    console.log('keyCode:', keyboard.keyCode)

    """))
# keyboard.js_on_event('change:keyCode', callback_keyboard) # no luck
keyboard.js_on_change('keyCode', callback_keyboard)

        



DOC = curdoc() # hold on to an instance of current doc in case need multithreads

SIZING_MODE =  'fixed' # 'scale_width' also an option, 'scale_both', 'scale_width', 'scale_height', 'stretch_both'



#placeholder figure
#mainfig = figure(tools="previewsave",  width=600, height=400)
mainfig = eegbrow.fig

#desc = bokeh.models.Div(text=open(path.join(path.dirname(__file__), "description.html")).read(), width=800)
desc = bokeh.models.Div(text="""Some placeholder text""")

### layout ###
# there are unicode labels which would look better
MVT_BWIDTH = 50
# note am setting button width as same as widget box (wbox50) to make one abut the next
bBackward10 = bokeh.models.widgets.Button(label='<<', width=MVT_BWIDTH)
bBackward1 = bokeh.models.widgets.Button(label='\u25C0', width=MVT_BWIDTH)  # <-
bForward10 = bokeh.models.widgets.Button(label='>>', width=MVT_BWIDTH)
bForward1 = bokeh.models.widgets.Button(label='\u25B6', width=MVT_BWIDTH) # -> or '\u279F'

def forward1():
    eegbrow.loc_sec += 1
    # print('keyCode: ', keyboard.keyCode)
    eegbrow.update()

def forward10():
    eegbrow.loc_sec += 10
    # print('keyCode: ', keyboard.keyCode)
    eegbrow.update()

def backward1():
    eegbrow.loc_sec -= 1
    # print('keyCode: ', keyboard.keyCode)
    eegbrow.update()

def backward10():
    eegbrow.loc_sec -= 10
    # print('keyCode: ', keyboard.keyCode)
    eegbrow.update()
    
bForward1.on_click(forward1)
bBackward1.on_click(backward1)
bForward10.on_click(forward10)
bBackward10.on_click(backward10)

# keyCodes
# {'ArrowRight': 39,
#  'ArrowLeft' : 37,
#  'ArrowUp' : 38,
#  'ArrowDown' : 40}

# the CustomJS args dict maps string names to Bkeh models, any models on python side
# will be avaiable in the javascript code string (@code) cb_obj is also available which represents
# the model triggering the callback 
callback_keydown = bokeh.models.callbacks.CustomJS(
    args=dict(keyboard=keyboard, bForward1=bForward1),
    code="""
    console.log('in keydown_callback, bForward1', bForward1)
    // experiments in trying to use this the passed in type
    // bForward1.clicks += 1//  this is incrementing correctly but it does not trigger the clicked OR does it?
    // bForward1.change.emit() // this sort of works, but can't handle repeats

    // console.log('emit obj:', bForward1.emit) // does not work
    // bForward1.click.emit() // does not work
    // console.log('keyCode:', keyboard.keyCode)
    // console.log('cb_obj:', cb_obj)
    // keyboard.change.emit() // is this needed?
    """)
#keyboard.keypress_callback = callback_keydown # this does not seem to work???
keyboard.js_on_change('keyCode', callback_keydown)


    
def keycallback_print(attr, old, new):
    print('keycallback_print: ', 'keyCode:', keyboard.keyCode, 'key:"%s"' % keyboard.key, 'ctrl/shift/alt:',
          keyboard.ctrlKey, keyboard.shiftKey, keyboard.altKey, attr, old, new)

def keycallback(attr, old, new):
    # print('keycallback: ', attr, old, new, 'keyCode:', keyboard.keyCode)
    keycallback_print(attr,old,new)
    if keyboard.keyCode == 71: # KeyG
        forward10()
    if keyboard.keyCode == 70: # KeyF
        forward1()
    if keyboard.keyCode == 65: # KeyA
        backward10()
    if keyboard.keyCode == 68: # KeyD
        eegbrow.yscale /= 1.5
        eegbrow.update()

    if keyboard.keyCode == 38: # ArrowUp
        # increase gain
        eegbrow.yscale *= 1.5
        eegbrow.update()
    if keyboard.keyCode == 69: # key='e'
        print('loc_sec:', eegbrow.loc_sec)
        eegbrow.yscale *= 1.5
        eegbrow.update()
        return
    if keyboard.keyCode == 83: # KeyS
        backward1()
    if keyboard.keyCode == 39: # ArrowRight
        forward10()
    if keyboard.keyCode == 37: # ArrowLeft
        backward10()
        
    if keyboard.keyCode == 40: # ArrowDown
        # decrease gain
        eegbrow.yscale /= 1.5 
        eegbrow.update()


    
keyboard.on_change('key_num_presses',keycallback)
# keyboard.on_change('keyCode',keycallback_print)
        
bottomrowctrls = [bBackward10,bBackward1,bForward1, bForward10]
toprowctrls = [bokeh.models.widgets.Select(title='Montage',value='trace', options=['trace', 'db','tcp']),
               bokeh.models.widgets.Select(title='Sensitivity',value='7uV/div', options=['1uV/div', '3uV/div','7uV/div','10uV/div']),
               bokeh.models.widgets.Select(title='LF',value='0.3Hz', options=['None', '0.1Hz','0.3Hz','1Hz','5Hz']),
               bokeh.models.widgets.Select(title='HF',value='70Hz', options=['None', '15Hz','30Hz','50Hz','70Hz']),

]
                 

#for control in controls:
#    control.on_change('value', lambda attr, old, new: update())



#inputs = widgetbox(*controls[:3], sizing_mode=SIZING_MODE)
wbox50 = functools.partial(layouts.widgetbox, sizing_mode=SIZING_MODE, width=MVT_BWIDTH)
wbox20 = functools.partial(layouts.widgetbox, sizing_mode=SIZING_MODE, width=150)
toprow = layouts.row(*map(wbox20, toprowctrls))
# toprow = layouts.row(wbox(layouts.row(children=toprowctrls)))
print(toprow)
bottomrow = layouts.row(*map(wbox50, bottomrowctrls))

L = layouts.layout([
    [desc],
    [toprow],
    [mainfig],
    [bottomrow],
    [keyboard],
    ], sizing_mode=SIZING_MODE)

L.js_on_event
DOC.add_root(L)
