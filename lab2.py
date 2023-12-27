import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def rotate_rod_with_spiral_spring(angle, rotation_angle, spiral_thetta):
    rod_length = 1.0
    disk_radius = 0.2

    rod_x = np.array([0, rod_length * np.cos(angle)])
    rod_y = np.array([0, rod_length * np.sin(angle)])

    disk_x = rod_x[1]
    disk_y = rod_y[1]

    rotated_A_x = disk_x + disk_radius * np.cos(rotation_angle)
    rotated_A_y = disk_y + disk_radius * np.sin(rotation_angle)

    A_x = rotated_A_x
    A_y = rotated_A_y

    R1 = 0  # Внутренний радиус спирали
    R2 = 0.1  # Внешний радиус спирали

    # Вращение координат спиральной пружины относительно центра вращения
    X_Spiral_Spring = disk_x - (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.sin(spiral_thetta + angle + np.pi/2)
    Y_Spiral_Spring = disk_y + (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.cos(spiral_thetta + angle + np.pi/2)

    return rod_x, rod_y, disk_x, disk_y, disk_radius, A_x, A_y, X_Spiral_Spring, Y_Spiral_Spring


fig, ax = plt.subplots(1, 1)
ax.axis('equal')

initial_angle = 0.0
initial_rotation_angle = 0.0

# Задаем параметры спиральной пружины
Nv = 4
spiral_thetta = np.linspace(0, Nv * 2 * math.pi, 500)

# Функция анимации
def update(frame):
    ax.clear()
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])

    angle = initial_angle + 0.05 * frame  # Увеличиваем угол поворота стержня
    rotation_angle = initial_rotation_angle + 0.1 * frame  # Увеличиваем угол вращения диска

    rod_x, rod_y, disk_x, disk_y, disk_radius, A_x, A_y, X_Spiral_Spring, Y_Spiral_Spring = rotate_rod_with_spiral_spring(angle, rotation_angle, spiral_thetta)

    # Рисуем спиральную пружину
    ax.plot(X_Spiral_Spring, Y_Spiral_Spring, color='black', linestyle='-', linewidth=2)

    # Рисуем стержень
    ax.plot(rod_x, rod_y, color='blue', linewidth=2)

    # Рисуем точку O
    ax.scatter(rod_x[0], rod_y[0], color='red')

    # Рисуем диск
    disk = plt.Circle((disk_x, disk_y), disk_radius, color='gray', fill=True)
    ax.add_patch(disk)

    # Отмечаем точку A
    ax.scatter(A_x, A_y, color='orange')

# Создаем анимацию
animation = FuncAnimation(fig, update, frames=500, interval=10, blit=False)

plt.show()
