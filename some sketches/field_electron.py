import numpy as np
import matplotlib.pyplot as plt
import pyperclip
pi = np.pi

R=1

dr = 0.005
dtheta = 0.005

def field(h):
    sum = 0
    for theta in np.linspace(0, 2*pi, num=int(2*pi/dtheta)):
        for r in np.linspace(0, R, num=int(R/dr)):
            sum += r*(r*np.sin(theta)+h) / ((r*np.cos(theta))**2 + (r*np.sin(theta)+h)**2) * dr * dtheta

    return sum

h = np.linspace(-10, 10, num=160)
F = field(h)

plt.title("")
plt.xlabel("Height of electron above the field $h$")
plt.ylabel("Electric force felt at the center of the field $F$")
plt.plot(h, F)
plt.grid()
plt.show()

message = ""
for i in range(len(h)):
    message += "(" + str(h[i]) + "," + str(F[i]) + "),"

pyperclip.copy(message)