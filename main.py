import sys
import numpy as np
import matplotlib
from PyQt5.QtCore import QLocale
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
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
        self.is_panning = False
        self.press = None
        self.full_reflection = False
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.fig.tight_layout()
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
        alpha = np.deg2rad(self.theta_deg)
        n1 = np.sqrt(self.epsilon1)
        n2 = np.sqrt(self.epsilon2)
        sin_val = n1 * np.sin(alpha) / n2
        if abs(sin_val) <= 1:
            self.full_reflection = False
            alpha2 = np.arcsin(sin_val)
            Ex2 = self.E0 * np.cos(alpha2)
            Ey2 = self.E0 * np.sin(alpha2)
        else:
            self.full_reflection = True
            Ex2 = 0.0
            Ey2 = 0.0
        Ex1 = self.E0 * np.cos(alpha)
        Ey1 = self.E0 * np.sin(alpha)
        D1x = self.epsilon1 * Ex1
        D1y = self.epsilon1 * Ey1
        D2x = self.epsilon2 * Ex2
        D2y = self.epsilon2 * Ey2
        Ex_bottom = Ex1
        Ey_bottom = Ey1
        Dx_bottom = D1x
        Dy_bottom = D1y
        Ex_top = Ex2
        Ey_top = Ey2
        Dx_top = D2x
        Dy_top = D2y
        self.Ex = np.where(self.Y >= 0, Ex_top, Ex_bottom)
        self.Ey = np.where(self.Y >= 0, Ey_top, Ey_bottom)
        self.Dx = np.where(self.Y >= 0, Dx_top, Dx_bottom)
        self.Dy = np.where(self.Y >= 0, Dy_top, Dy_bottom)

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
        if self.full_reflection:
            self.ax.text(0, self.limit_y / 2, 'Полное отражение', color='magenta', fontsize=12, ha='center')
        density = 1.5
        self.ax.streamplot(self.X, self.Y, self.Ex, self.Ey, color='blue', linewidth=1, density=density, arrowsize=1)
        self.ax.streamplot(self.X, self.Y, self.Dx, self.Dy, color='red', linewidth=1, density=density, arrowsize=1)
        custom_lines = [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='red', lw=2)]
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
            self.is_panning = True
            self.press = event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim()

    def on_release(self, event):
        if event.button == 1:
            self.is_panning = False
            self.press = None
        self.draw()

    def on_move(self, event):
        if not self.is_panning:
            return
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
        w = QWidget()
        self.setCentralWidget(w)
        v = QVBoxLayout()
        w.setLayout(v)
        self.plot = EMFieldPlot(self)
        v.addWidget(self.plot)
        l1 = QHBoxLayout()
        dv = QDoubleValidator(bottom=0.0)
        dv.setLocale(QLocale(QLocale.C))
        l1.addWidget(QLabel('ε₁:'))
        self.eps1_input = QLineEdit()
        self.eps1_input.setValidator(dv)
        l1.addWidget(self.eps1_input)
        l1.addWidget(QLabel('ε₂:'))
        self.eps2_input = QLineEdit()
        self.eps2_input.setValidator(dv)
        l1.addWidget(self.eps2_input)
        l1.addWidget(QLabel('E₀:'))
        self.E0_input = QLineEdit()
        self.E0_input.setValidator(dv)
        l1.addWidget(self.E0_input)
        l1.addWidget(QLabel('θ (градусы):'))
        self.theta_input = QLineEdit()
        self.theta_input.setValidator(QDoubleValidator(0.0, 90.0, 2))
        l1.addWidget(self.theta_input)
        v.addLayout(l1)
        l2 = QHBoxLayout()
        l2.addWidget(QLabel('Лимит X:'))
        self.limit_x_input = QLineEdit()
        self.limit_x_input.setValidator(dv)
        l2.addWidget(self.limit_x_input)
        l2.addWidget(QLabel('Лимит Y:'))
        self.limit_y_input = QLineEdit()
        self.limit_y_input.setValidator(dv)
        l2.addWidget(self.limit_y_input)
        self.update_button = QPushButton('Обновить')
        self.update_button.clicked.connect(self.update_plot)
        l2.addWidget(self.update_button)
        v.addLayout(l2)
        self.limit_x_input.setText('10')
        self.limit_y_input.setText('10')

    def update_plot(self):
        try:
            epsilon1 = float(self.eps1_input.text())
            epsilon2 = float(self.eps2_input.text())
            E0 = float(self.E0_input.text())
            theta_deg = float(self.theta_input.text())
            limit_x = float(self.limit_x_input.text())
            limit_y = float(self.limit_y_input.text())
            if epsilon1 <= 0 or epsilon2 <= 0 or E0 <= 0 or limit_x <= 0 or limit_y <= 0:
                raise ValueError
            if not (0 <= theta_deg <= 90):
                raise ValueError
            self.plot.update_parameters(epsilon1, epsilon2, E0, theta_deg, limit_x, limit_y)
        except ValueError:
            QMessageBox.warning(self, 'Ошибка ввода', 'Введите корректные положительные числа.')

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
