from __future__ import division

from os.path import dirname, join

import numpy as np

from bokeh.driving import count
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Div, Column, Button
import bokeh.layouts as layouts

from newwidget import NewWidget

doc = curdoc()

x = np.arange(0, 300, 20)
y = np.arange(0, 300, 20)
xx, yy = np.meshgrid(x, y)
xx = xx.ravel()
yy = yy.ravel()



def compute(t):
    value = np.sin(xx/50 + t/10) * np.cos(yy/50 + t/10) * 50 + 50
    # print("doing compute..")
    return dict(x=xx, y=yy, z=value, color=value)

b = Button(name='test')
source = ColumnDataSource(data=compute(0))

#content_filename = join(dirname(__file__), "description.html")

description = Div(text="some text to describe this",
                  render_as_text=False, width=600)

# surface = Surface3d(x="x", y="y", z="z", color="color", data_source=source)
nwidget = NewWidget()

curdoc().add_root(layouts.column([description, b, layouts.WidgetBox(nwidget)]))
# doc.add_root(layouts.column([description, b]))

@count()
def update(t):
    source.data = compute(t)

doc.add_periodic_callback(update, 100)
doc.title = "NewWidget"
