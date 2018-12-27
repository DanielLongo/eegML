# File Open Dialog and upload to python
# https://github.com/bokeh/bokeh/issues/6096
# For anyone searching for help: in the meantime before this is added to bokeh,
# here's how I implemented sending a file from the user's computer to a Bokeh
# server:

# Using templates (Bokeh's, Flask's, whatever you're using), insert an HTML file
# choosing input: <input id="file-input" type="file"> 

# Attach a javascript callback
# to a button that reads the selected file's contents

# If the file is complex like an excel document, you can base64 encode it. If
# it's something like a text file or a csv, you might as well leave it as-is

# Insert it into a dedicated
# ColumnDataSource, treating it as a dictionary to transfer data from browser to
# server On the server side.

# Implement an on_change("data", callback) for the data
# source 

# Base64 decode the file if necessary If you just need the file contents as
# a string, you're done. 
# If you need it as a proper file (for instance, to use with
# pandas.read_excel), you can wrap it in a StringIO object to make it file-like
# It's a bit complicated, but the upcoming namespace model for Bokeh would make it
# a bit easier. Feel free to ping me for code samples.

# Hello @kevinsa5 ,

#could you provide code samples. Thank you
#The following is my code

import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Button
from bokeh.io import curdoc

source = ColumnDataSource(dict())

print(source.data.keys)

def callback1(attr,old,new):
print('data got changed')

button = Button(label="Upload", button_type="success")
button.callback = CustomJS(args=dict(source=source),
code="""
fileSelector = document.createElement('input');
fileSelector.setAttribute('type', 'file');

selectDialogueLink = document.createElement('a');
selectDialogueLink.setAttribute('href', '');

selectDialogueLink.onclick = fileSelector.click();

document.body.appendChild(selectDialogueLink);

if ('files' in fileSelector) {
  if (fileSelector.files.length == 0) {
    alert("Select file.");
  } else {
    for (var i = 0; i < fileSelector.files.length; i++) {
       var file = fileSelector.files[i];
       var reader = new FileReader();
       source.data = reader.readAsArrayBuffer(file);
       if ('name' in file) {
         alert("name: " + file.name);
       }
       if ('size' in file) {
         alert("size: " + file.size);
       }
    }
  }
} 
""") 
