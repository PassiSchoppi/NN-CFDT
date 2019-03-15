from bokeh.plotting import figure, output_file, show
import numpy as np


class Plott:
    title = ""

    def __init__(self, file_name, a_title="", a_x_label="", a_y_label=""):
        output_file(str(file_name))
        self.title = a_title
        self.x_label = a_x_label
        self.y_label = a_y_label
        self.p = figure(title=str(self.title),
                        x_axis_label=str(self.x_label),
                        y_axis_label=str(self.y_label))

    def add_graph(self, y, mode="line",
                  legend="", colour="blue", width=2):
        if mode == "line":
            self.p.line(np.arange(0, len(y), 1), y,
                        legend=str(legend),
                        line_width=int(width),
                        line_color=str(colour))
        elif mode == "circle":
            self.p.circle(np.arange(0, len(y), 1), y,
                          legend=str(legend),
                          fill_color=str(colour),
                          line_color=str(colour),
                          size=int(width))

    def save_plott(self, mode="save"):
        if mode == "save":
            save(self.p)
        elif mode == "show":
            show(self.p)
        else:
            print("not saved plott")
