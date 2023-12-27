import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

Steps = 500
t_fin = 20
t = np.linspace(0, t_fin, Steps)


def odesys(y, t, m1, l, m2, r, m3, g, c):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]
    #тут все верно написано, надо править аргументы
    a11 = np.cos(y[0]+y[1])*r
    a12 = (m1/3 + m2 + m3)*l
    a21 = (m2/2 + m3) * r
    a22 = m3*l*np.cos(y[0]+y[1])

    b1 = (m1/2 + m2 + m3)*g*np.sin(y[1]) - (c/l)*(y[0]+y[1])
    b2 = m3*g*np.sin(y[0]) - (c/r)*(y[0]+y[1])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

m1 = 1
m2 = 2
m3 = 3
c = 5
l = 4
r = 1
g = 9.81

psi0 = 0
#psi0 = 0
phi0 = 0
dpsi0 = 0
dphi0 = 0
y0 = [psi0, phi0, dpsi0, dphi0]

Y = odeint(odesys, y0, t, (m1, l, m2, r, m3, g, c))

#что это такое
fig = plt.figure(figsize=[20, 20])
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal', adjustable='datalim')  # Устанавливаем соотношение сторон
ax.set_xlim([0, 20])  # Установите пределы оси X
ax.set_ylim([0, 20])  # Установите пределы оси Y


psi = Y[:, 0]
phi = Y[:, 1]
dpsi = Y[:, 2]
dphi = Y[:, 3]

ddpsi, ddphi = [odesys(y, t, m1, l, m2, r, m3, g, c)[2] for y, t in zip(Y, t)], [odesys(y, t, m1, l, m2, r, m3, g, c)[3] for y, t in zip(Y, t)]



Nx = (m2+m3)*(g - l*(ddphi*np.sin(phi) + dphi*dphi*np.cos(phi)) + m3*r*(ddpsi*np.sin(psi) + dpsi*dpsi * np.cos(psi)))
Ny = (m2+m3)*l*((ddphi*np.cos(phi) - dphi*dphi*np.sin(phi)) + m3*r*(ddpsi*np.cos(psi) - dpsi*dpsi * np.sin(psi)))

#графики
fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, psi, color='Blue')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, Nx, color='Red')
ax_for_graphs.set_title("Nx(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, Ny, color='Green')
ax_for_graphs.set_title("Ny(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

X_O = 10
Y_O = 10

X_C = X_O + l * np.sin(Y[:, 1])
Y_C = Y_O + l * np.cos(Y[:, 1])

X_A = X_C + r * np.sin(Y[:, 0] + Y[:, 1])
Y_A = Y_C - r * np.cos(Y[:, 0] + Y[:, 1])

# Создайте графические объекты для стержня OC, точек O, C и A
Drawed_OC, = ax.plot([X_O, X_C[0]], [Y_O, Y_C[0]], '-o', color='blue')
Point_O, = ax.plot(X_O, Y_O, marker='o', color='red')
Point_C, = ax.plot(X_C[0], Y_C[0], marker='o', markersize=5, color='green')
Point_A, = ax.plot(X_A[0], Y_A[0], marker='o', markersize=5, color='purple')

# Создайте объект для диска
disk = plt.Circle((X_C[0], Y_C[0]), r, color='gray', fill=True)
ax.add_patch(disk)

Nv = 3
R1 = r/10
R2 = r/2
spiral_thetta = np.linspace(0, Nv*2*math.pi-phi[0], 500)
X_Spiral_Spring = X_C - (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.sin(spiral_thetta)
Y_Spiral_Spring = Y_C + (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.cos(spiral_thetta)

Drawed_Spiral_Spring, = ax.plot([], [], color='black', linestyle='-', linewidth=2)

def anima(i):
    X_C = X_O + l * np.sin(Y[i, 1])
    Y_C = Y_O + l * np.cos(Y[i, 1])
    
    X_A = X_C + r * np.sin(Y[i, 0] + Y[i, 1])
    Y_A = Y_C - r * np.cos(Y[i, 0] + Y[i, 1])

    # Обновите данные на графике
    Drawed_OC.set_data([X_O, X_C], [Y_O, Y_C])
    Point_C.set_data(X_C, Y_C)
    Point_A.set_data(X_A, Y_A)

    # Обновите координаты диска
    disk.set_center((X_C, Y_C))
    spiral_thetta = np.linspace(0, Nv*2*math.pi-phi[i], 500)
    X_Spiral_Spring = X_C - (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.sin(spiral_thetta + np.pi)
    Y_Spiral_Spring = Y_C + (R1 + spiral_thetta * (R2 - R1) / spiral_thetta[-1]) * np.cos(spiral_thetta + np.pi)
    Drawed_Spiral_Spring.set_data(X_Spiral_Spring, Y_Spiral_Spring)

    return Drawed_OC, Point_O, Point_C, Point_A, disk, Drawed_Spiral_Spring

# Создайте анимацию
anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()