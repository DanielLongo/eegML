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

## 
KEYBOARDRESPONDERCODE_TS = """
import {div, empty} from "core/dom"
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

//import {jQuery} from "jquery.min.js"
// import * as $ from "jquery";
var url = "https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js";

// load jQuery by hand
var script = document.createElement('script');
script.src = url;
script.async = false;
script.onreadystatechange = script.onload = function() {jQuery.noConflict(); };
document.querySelector("head").appendChild(script);
var gLastKeyCode = undefined

// create custom Bokeh model based upon example
export class KeyboardResponderView extends LayoutDOMView {

  initialize(options) {
    super.initialize(options)

    console.log("setting up jQuery");
    console.log(jQuery);
    this.model.keyCode = 0
    jQuery('body').keydown( // same as .on('keydown', handler);
      function(ev) {
          console.log("got key", ev.keyCode);
          // this.model.keyCode = ev.keyCode
          gLastKeyCode = ev.keyCode
          // this.render()
          // how to set lastKeyCode = ev.keyCode
	  // jQuery('#output').text(JSON.stringify(ev.keyCode));
	  // jQuery('#which').text(ev.which);
      });

    this.render()

    // Set listener so that when the a change happens
    // event, we can process the new data
    // this.connect(this.model.slider.change, () => this.render())
  }

  render() {
    // Bokehjs Views (like Backbone) create <div> elements by default, accessible as
    // this.el
    // Many Bokeh views ignore this default <div>, and instead do things
    // like draw to the HTML canvas. In this case though, we change the
    // contents of the <div>, based on the current value.
    console.log('trying render', this.el)
    empty(this.el)
    // this.el.appendChild(document.createTextNode(gLastKeyCode))

  }
}

export class KeyboardResponder extends LayoutDOM {

  // If there is an associated view, this is boilerplate.
  default_view = KeyboardResponderView

  // The ``type`` class attribute should generally match exactly the name
  // of the corresponding Python class.
  type = "KeyboardResponder"
}

// The @define block adds corresponding "properties" to the JS model. These
// should basically line up 1-1 with the Python model class. Most property
// types have counterparts, e.g. bokeh.core.properties.String will be
// p.String in the JS implementation. Where the JS type system is not yet
// as rich, you can use p.Any as a "wildcard" property type.
KeyboardResponder.define({
  // text:   [ p.String ],
  // slider: [ p.Any    ],
  keycode : [ p.Int, 0 ]
})

"""

class KeyboardResponder(LayoutDOM):
    #__implementation__ = TypeScript(KEYBOARDRESPONDERCODE_TS)
    __implementation__ = "keyboardresponder.ts"
    # can use # __css__ = '<a css file.css>'
    # can use # __javascript__ = 'katex.min.js'
    keycode = properties.Int(default=0) # this should match with javascript
    key_num_presses = properties.Int(default=0)
    keypress_callback = properties.Instance(bokeh.models.callbacks.Callback,
                                   help=""" A callback to run in the browser whenever a key is pressed
                                   """)
keyboard = KeyboardResponder()
keyboard.css_classes = ['keyboard']
callback_keyboard = bokeh.models.callbacks.CustomJS(
    args=dict(keyboard=keyboard, code="""
    console.log('in callback_keyboard')
    console.log('keycode:', keyboard.keycode)
    /* keyboard.change.emit() */
    """))
# keyboard.js_on_event('change:keycode', callback_keyboard)
keyboard.js_on_change('keycode', callback_keyboard)

        

# the CustomJS args dict maps string names to Bkeh models, any models on python side
# will be avaiable in the javascript code string (@code) cb_obj is also available which represents
# the model triggering the callback 
callback_keydown = bokeh.models.callbacks.CustomJS(
    args=dict(keyboard=keyboard),
    code="""
    console.log('in keydown_callback')
    // console.log('keycode:', keyboard.keycode)
    // console.log('cb_obj:', cb_obj)
    // keyboard.change.emit() // is this needed?
    """)
keyboard.keypress_callback = callback_keydown


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
    # print('keycode: ', keyboard.keycode)
    eegbrow.update()

def forward10():
    eegbrow.loc_sec += 10
    # print('keycode: ', keyboard.keycode)
    eegbrow.update()

def backward1():
    eegbrow.loc_sec -= 1
    # print('keycode: ', keyboard.keycode)
    eegbrow.update()

def backward10():
    eegbrow.loc_sec -= 10
    # print('keycode: ', keyboard.keycode)
    eegbrow.update()
    
bForward1.on_click(forward1)
bBackward1.on_click(backward1)

# keycodes
# {'ArrowRight': 39,
#  'ArrowLeft' : 37,
#  'ArrowUp' : 38,
#  'ArrowDown' : 40}

def keycallback(attr, old, new):
    print('keycallback: ', attr, old, new, 'keycode:', keyboard.keycode)
    if keyboard.keycode == 70: # KeyF
        forward10()
    if keyboard.keycode == 65: # KeyA
        backward10()
    if keyboard.keycode == 68: # KeyD
        forward1()
    if keyboard.keycode == 83: # KeyS
        backward1()
    if keyboard.keycode == 39: # ArrowRight
        forward10()
    if keyboard.keycode == 37: # ArrowLeft
        backward10()
    if keyboard.keycode == 38: # ArrowUp
        pass # increase gain
    if keyboard.keycode == 40: # ArrowDown
        pass # decrease gain
    
def keycallback_print(attr, old, new):
    print('keycallback: ', attr, old, new, 'keycode:', keyboard.keycode)

keyboard.on_change('key_num_presses',keycallback)
# keyboard.on_change('keycode',keycallback_print)
        
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
