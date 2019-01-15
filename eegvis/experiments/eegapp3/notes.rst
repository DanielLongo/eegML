
source url =  https://bokeh.github.io/blog/2017/7/5/idiomatic_bokeh/

An empty canvas when passed to the bokeh.io.show method::
::
    from bokeh.models import Plot, Range1d

    plot = Plot(x_range=Range1d(), y_range=Range1d(), plot_height=200)


https://bokeh.github.io/blog/2017/6/29/simple_bokeh_server/
Start a bokeh server on port 5000 with a single route to / with single plot
::
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.plotting import figure, ColumnDataSource

    def make_document(doc):
        fig = figure(title='Line plot!', sizing_mode='scale_width')
	fig.line(x=[1, 2, 3], y=[1, 4, 9])

	doc.title = "Hello, world!"
	doc.add_root(fig)

    apps = {'/': Application(FunctionHandler(make_document))}

    server = Server(apps, port=5000)
    server.start()   
