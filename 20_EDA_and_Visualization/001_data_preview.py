from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from data import Data

columns = Data.columns
students = Data.students

figure_width = 14
figure_height = 7
font_size = 8
column_width = 0.95
row_height = 1.35
visible_rows = 15
max_start_row = len(students) - visible_rows

fig, ax = plt.subplots(figsize=(figure_width, figure_height))
plt.subplots_adjust(right=0.88)


def draw_table(start_row):
    ax.clear()
    end_row = start_row + visible_rows
    visible_students = students[start_row:end_row]

    table = ax.table(cellText=visible_students,
                     colLabels=columns,
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(column_width, row_height)

    ax.axis("off")
    ax.set_title(f"001 Data Preview: Rows {start_row + 1} - {end_row}")


draw_table(start_row=0)

slider_axis = plt.axes([0.91, 0.15, 0.03, 0.7])
row_slider = Slider(ax=slider_axis,
                    label="Row",
                    valmin=0,
                    valmax=max_start_row,
                    valinit=max_start_row,
                    valstep=1,
                    orientation="vertical")


def update_table(value):
    start_row = max_start_row - int(value)
    draw_table(start_row=start_row)
    fig.canvas.draw_idle()


row_slider.on_changed(update_table)

plt.show()
