# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:26:21 2017

@author: Kevin Anderson
"""
import io # added per note
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Button
from bokeh.io import curdoc

import StringIO
import base64

file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})

def file_callback(attr,old,new):
    print 'filename:', file_source.data['file_name']
    raw_contents = file_source.data['file_contents'][0]
    # remove the prefix that JS adds  
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    #file_io = StringIO.StringIO(file_contents)
    # substitute
    file_io = io.StringIO(bytes.decode(file_contents))

    df = pd.read_excel(file_io)
    print "file contents:"
    print df

file_source.on_change('data', file_callback)

button = Button(label="Upload", button_type="success")
button.callback = CustomJS(args=dict(file_source=file_source), code = """
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
""")

curdoc().add_root(row(button))

# Hello @kevinsa5 ,

# Thank you for the code. I was able to upload .csv files data with a minor modification on my system.

# modification -->file_io = io.StringIO(bytes.decode(file_contents))
# It worked like magic.
