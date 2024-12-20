import sys
import numpy as np
import matplotlib
from PyQt5.QtCore import QLocale

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QDoubleValidator
from matplotlib.lines import Line2D

class EMFieldPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.epsilon1 = None
        self.epsilon2 = None
        self.E0 = None
        self.theta_deg = None
        self.limit_x = None
        self.limit_y = None

        self.fig = Figure()
        super(EMFieldPlot, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.fig.tight_layout()

        self.press = None

        self.mpl_connect('scroll_event', self.zoom)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_move)

    def compute_fields(self):
        if None in (self.epsilon1, self.epsilon2, self.E0, self.theta_deg, self.limit_x, self.limit_y):
            return

        self.x = np.linspace(-self.limit_x, self.limit_x, 400)
        self.y = np.linspace(-self.limit_y, self.limit_y, 400)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        theta_rad = np.deg2rad(self.theta_deg)
        E0x = self.E0 * np.cos(theta_rad)
        E0y = self.E0 * np.sin(theta_rad)

        E1x = E0x
        E1y = E0y
        D1x = self.epsilon1 * E1x
        D1y = self.epsilon1 * E1y

        E2x = E0x
        E2y = (self.epsilon1 / self.epsilon2) * E0y
        D2x = self.epsilon2 * E2x
        D2y = self.epsilon2 * E2y

        self.Ex = np.where(self.Y >= 0, E2x, E1x)
        self.Ey = np.where(self.Y >= 0, E2y, E1y)
        self.Dx = np.where(self.Y >= 0, D2x, D1x)
        self.Dy = np.where(self.Y >= 0, D2y, D1y)

    def plot_fields(self):
        if None in (self.epsilon1, self.epsilon2, self.E0, self.theta_deg, self.limit_x, self.limit_y):
            self.ax.clear()
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title('Граничные условия двух диэлектриков')
            self.draw()
            return

        self.ax.clear()
        self.ax.fill_between(self.x, 0, self.limit_y, color='lightgray', alpha=0.5, label='ε₂')
        self.ax.fill_between(self.x, -self.limit_y, 0, color='white', alpha=0.5, label='ε₁')

        self.ax.axhline(0, color='k', linewidth=2)

        density = 1.5

        self.ax.streamplot(
            self.X, self.Y, self.Ex, self.Ey,
            color='blue', linewidth=1, density=density, arrowsize=1
        )

        self.ax.streamplot(
            self.X, self.Y, self.Dx, self.Dy,
            color='red', linewidth=1, density=density, arrowsize=1
        )

        custom_lines = [
            Line2D([0], [0], color='blue', lw=2),
            Line2D([0], [0], color='red', lw=2)
        ]
        self.ax.legend(custom_lines, ['Напряженность E', 'Смещение D'])

        self.ax.set_xlim(-self.limit_x, self.limit_x)
        self.ax.set_ylim(-self.limit_y, self.limit_y)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Граничные условия двух диэлектриков\nСиние линии: Напряженность E, Красные линии: Смещение D')
        self.draw()

    def zoom(self, event):
        if None in (self.limit_x, self.limit_y):
            return

        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        new_xmin = xdata - new_width * (1 - relx)
        new_xmax = xdata + new_width * relx
        new_ymin = ydata - new_height * (1 - rely)
        new_ymax = ydata + new_height * rely

        new_xmin = max(-self.limit_x, new_xmin)
        new_xmax = min(self.limit_x, new_xmax)
        new_ymin = max(-self.limit_y, new_ymin)
        new_ymax = min(self.limit_y, new_ymax)

        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.draw()

    def on_press(self, event):
        if event.button == 1:
            self.press = event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim()

    def on_release(self, event):
        self.press = None
        self.draw()

    def on_move(self, event):
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        xpress, ypress, xlim, ylim = self.press
        dx = event.x - xpress
        dy = event.y - ypress
        dx_scaled = -dx * (xlim[1] - xlim[0]) / self.fig.bbox.width
        dy_scaled = dy * (ylim[1] - ylim[0]) / self.fig.bbox.height
        new_xlim = [xlim[0] + dx_scaled, xlim[1] + dx_scaled]
        new_ylim = [ylim[0] + dy_scaled, ylim[1] + dy_scaled]

        new_xlim[0] = max(-self.limit_x, new_xlim[0])
        new_xlim[1] = min(self.limit_x, new_xlim[1])
        new_ylim[0] = max(-self.limit_y, new_ylim[0])
        new_ylim[1] = min(self.limit_y, new_ylim[1])

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw()

    def update_parameters(self, epsilon1, epsilon2, E0, theta_deg, limit_x, limit_y):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.E0 = E0
        self.theta_deg = theta_deg
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.compute_fields()
        self.plot_fields()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Визуализация граничных условий двух диэлектриков')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        vbox = QVBoxLayout()
        central_widget.setLayout(vbox)

        self.plot = EMFieldPlot(self)
        vbox.addWidget(self.plot)

        input_layout1 = QHBoxLayout()

        double_validator = QDoubleValidator(bottom=0.0)
        double_validator.setLocale(QLocale(QLocale.C))

        input_layout1.addWidget(QLabel('Диэлектрическая проницаемость ε₁:'))
        self.epsilon1_input = QLineEdit()
        self.epsilon1_input.setPlaceholderText('Введите ε₁')
        self.epsilon1_input.setValidator(double_validator)
        input_layout1.addWidget(self.epsilon1_input)

        input_layout1.addWidget(QLabel('Диэлектрическая проницаемость ε₂:'))
        self.epsilon2_input = QLineEdit()
        self.epsilon2_input.setPlaceholderText('Введите ε₂')
        self.epsilon2_input.setValidator(double_validator)
        input_layout1.addWidget(self.epsilon2_input)

        input_layout1.addWidget(QLabel('Модуль напряженности E₀:'))
        self.E0_input = QLineEdit()
        self.E0_input.setPlaceholderText('Введите E₀')
        self.E0_input.setValidator(double_validator)
        input_layout1.addWidget(self.E0_input)

        input_layout1.addWidget(QLabel('Направление θ (градусы):'))
        self.theta_input = QLineEdit()
        self.theta_input.setPlaceholderText('Введите θ')
        self.theta_input.setValidator(QDoubleValidator(0.0, 90.0, 2))
        input_layout1.addWidget(self.theta_input)

        vbox.addLayout(input_layout1)

        input_layout2 = QHBoxLayout()

        input_layout2.addWidget(QLabel('Лимит X:'))
        self.limit_x_input = QLineEdit()
        self.limit_x_input.setPlaceholderText('Введите лимит X')
        self.limit_x_input.setValidator(double_validator)
        input_layout2.addWidget(self.limit_x_input)

        input_layout2.addWidget(QLabel('Лимит Y:'))
        self.limit_y_input = QLineEdit()
        self.limit_y_input.setPlaceholderText('Введите лимит Y')
        self.limit_y_input.setValidator(double_validator)
        input_layout2.addWidget(self.limit_y_input)

        self.update_button = QPushButton('Обновить')
        self.update_button.clicked.connect(self.update_plot)
        input_layout2.addWidget(self.update_button)

        vbox.addLayout(input_layout2)

        # self.epsilon1_input.setText('1.0')
        # self.epsilon2_input.setText('2.0')
        # self.E0_input.setText('1.0')
        # self.theta_input.setText('45.0')
        self.limit_x_input.setText('10')
        self.limit_y_input.setText('10')

    def update_plot(self):
        try:
            epsilon1 = float(self.epsilon1_input.text())
            epsilon2 = float(self.epsilon2_input.text())
            E0 = float(self.E0_input.text())
            theta_deg = float(self.theta_input.text())
            limit_x = float(self.limit_x_input.text())
            limit_y = float(self.limit_y_input.text())

            if epsilon1 <= 0 or epsilon2 <= 0 or E0 <= 0 or limit_x <= 0 or limit_y <= 0:
                raise ValueError("Диэлектрические проницаемости, модуль напряженности и лимиты должны быть положительными.")
            if not (0 <= theta_deg <= 90):
                raise ValueError("Угол θ должен быть в диапазоне от 0° до 90°.")

            self.plot.update_parameters(epsilon1, epsilon2, E0, theta_deg, limit_x, limit_y)
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка ввода", f"Пожалуйста, введите корректные числовые значения.\n{e}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
