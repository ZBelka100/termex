import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY


Pi = math.acos(-1)
t = sp.Symbol('t')
T = np.linspace(0, 10, 1000)

# Заданный закон
r = 2 + sp.sin(8*t)
phi = t + 0.2*sp.cos(6*t)


x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
w = sp.diff(phi, t)  # угловая скорость для радиуса кривизны

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
W = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    W[i] = sp.Subs(w, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-15, 15], ylim=[-15, 15])
ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o') # Первый элемент кортежа

VectorX = np.array([-0.2, 0, -0.2])
VectorY = np.array([0.1, 0, -0.1])

#Радиус вектор, скорость и ускорение соответвенно
RVLine, = ax1.plot([0, X[0]], [0, Y[0]], 'blue')  
RVVectorX, RVVectorY = Rot2D(VectorX, VectorY, math.atan2(Y[0], X[0]))
RVVector, = ax1.plot(RVVectorX+X[0], RVVectorY+Y[0], 'blue')


VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'red')
VVectorX, VVectorY = Rot2D(VectorX, VectorY, math.atan2(VY[0], VX[0]))
VVector, = ax1.plot(VVectorX+X[0]+VX[0], VVectorY+Y[0]+VY[0], 'red')


ALine, = ax1.plot([X[0], X[0]+AX[0]], [Y[0], Y[0]+AY[0]], 'yellow')
AVectorX, AVectorY = Rot2D(VectorX, VectorY, math.atan2(AY[0], AX[0]))
AVector, = ax1.plot(AVectorX+X[0]+AX[0], AVectorY+Y[0]+AY[0], 'yellow')

#Радиус кризизны через VX[0]/W[0]
RX, RY = Rot2D(X[0]+VX[0]/W[0], Y[0]+VY[0]/W[0], Pi/2)
RLine, = ax1.plot([X[0], RX], [Y[0], RY], 'black')

RVectorX, RVectorY = Rot2D(VectorX, VectorY, math.atan2(RY, RX))
RVector, = ax1.plot(RVectorX+RX, RVectorY+RY, 'black')

def anima(i):
    P.set_data(X[i], Y[i])
    RVLine.set_data([0, X[i]], [0, Y[i]])
    RVVectorX, RVVectorY = Rot2D(VectorX, VectorY, math.atan2(Y[i], X[i]))
    RVVector.set_data(RVVectorX+X[i], RVVectorY+Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    VVectorX, VVectorY = Rot2D(VectorX, VectorY, math.atan2(VY[i], VX[i]))
    VVector.set_data(VVectorX+X[i]+VX[i], VVectorY+Y[i]+VY[i])
    ALine.set_data([X[i], X[i]+AX[i]], [Y[i], Y[i]+AY[i]])
    AVectorX, AVectorY = Rot2D(VectorX, VectorY, math.atan2(AY[i], AX[i]))
    AVector.set_data(AVectorX+X[i]+AX[i], AVectorY+Y[i]+AY[i])
    RX, RY = Rot2D(VX[i] / W[i], VY[i] / W[i], Pi / 2)
    RLine.set_data([X[i], X[i]+RX], [Y[i], Y[i]+RY])
    RVectorX, RVectorY = Rot2D(VectorX, VectorY, math.atan2(RY, RX))
    RVector.set_data(RVectorX+X[i]+RX, RVectorY+Y[i]+RY)
    return P, VLine, VVector, ALine, AVector, RLine, RVector

anim = FuncAnimation(fig, anima, frames=1000, interval=100, repeat=False)

plt.show()