from bokeh.core.properties import Any, Dict, Instance, String, Int # List, Angle, Auto, Bool, Byte, Color, Complex, Date, Float, Interval, JSON, MinMaxBounds, Percent, Regex, Size, TimeDelta, Dict, RelativeDelta, Seq, Tuple, 
from bokeh.models import ColumnDataSource
from bokeh.models import LayoutDOM


# This custom extension model will have a DOM view that should layout-able in
# Bokeh layouts, so use ``LayoutDOM`` as the base class. If you wanted to create
# a custom tool, you could inherit from ``Tool``, or from ``Glyph`` if you
# wanted to create a custom glyph, etc.

# this is kind of a lie


class NewWidget(LayoutDOM):

    # The special class attribute ``__implementation__`` should contain a string
    # of JavaScript (or CoffeeScript) code that implements the JavaScript side
    # of the custom extension model.
    __implementation__ = "newwidget.coffee"

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    http://bokeh.pydata.org/en/latest/docs/reference/core.html#bokeh-core-properties

    # This is a Bokeh ColumnDataSource that can be updated in the Bokeh
    # server by Python code
    # data_source = Instance(ColumnDataSource)

    # The vis.js library that we are wrapping expects data for x, y, z, and
    # color. The data will actually be stored in the ColumnDataSource, but
    # these properties let us specify the *name* of the column that should
    # be used for each field.
    # x = String

    #y = String

    # z = String

    # color = String

    # Any of the available vis.js options for Graph3d can be set by changing
    # the contents of this dictionary.
    # options = Dict(String, Any, default=DEFAULTS)

    keycode = Int

