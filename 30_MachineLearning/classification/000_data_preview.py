import csv
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "classification_training.csv"

figure_width = 11
figure_height = 7
font_size = 8
row_height = 1.35
visible_rows = 15
column_widths = [0.22, 0.20, 0.16, 0.34]
SKILL_LEVEL_NAMES = {
    "0": "basic",
    "1": "intermediate",
    "2": "advanced",
}


# ============================================================
# 2. 데이터 파일 확인
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing data file: {DATA_PATH}")


# ============================================================
# 3. 데이터 불러오기
# ============================================================
with open(DATA_PATH, newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)

columns = rows[0]
skill_level_index = columns.index("skill_level")
columns[skill_level_index] = "skill_level"

samples = []
for row in rows[1:]:
    preview_row = row.copy()
    skill_level = preview_row[skill_level_index]
    skill_name = SKILL_LEVEL_NAMES[skill_level]
    preview_row[skill_level_index] = f"{skill_level} ({skill_name})"
    samples.append(preview_row)

max_start_row = len(samples) - visible_rows

if max_start_row < 0:
    max_start_row = 0
    visible_rows = len(samples)


# ============================================================
# 4. 데이터 표 시각화
# ============================================================
fig, ax = plt.subplots(figsize=(figure_width, figure_height))
plt.subplots_adjust(left=0.05, right=0.80)


def draw_table(start_row):
    ax.clear()
    end_row = start_row + visible_rows
    visible_samples = samples[start_row:end_row]

    table = ax.table(
        cellText=visible_samples,
        colLabels=columns,
        colWidths=column_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, row_height)

    ax.axis("off")
    ax.set_title(f"000 Training Data Preview: Rows {start_row + 1} - {end_row}")


draw_table(start_row=0)

slider_axis = plt.axes([0.92, 0.15, 0.025, 0.7])
row_slider = Slider(
    ax=slider_axis,
    label="Row",
    valmin=0,
    valmax=max_start_row,
    valinit=max_start_row,
    valstep=1,
    orientation="vertical",
)


def update_table(value):
    start_row = max_start_row - int(value)
    draw_table(start_row=start_row)
    fig.canvas.draw_idle()


row_slider.on_changed(update_table)

plt.show()
