import numpy as np
import matplotlib.pyplot as plt
import pyperclip
pi = np.pi

R=1

dr = 0.005
dtheta = 0.005

def field(h, x):
    sum = 0
    for theta in np.linspace(0, 2*pi, num=int(2*pi/dtheta)):
        for r in np.linspace(0, R, num=int(R/dr)):
            sum += r*(r*np.sin(theta)+h) / ((r*np.cos(theta) - x)**2 + (r*np.sin(theta)+h)**2) * dr * dtheta

    return sum

h = np.linspace(-5, 5, num=100)
x = np.linspace(-5, 5, num=100)

H, X = np.meshgrid(h, x)

F = field(H, X)

# plt.title("")
# plt.xlabel("Height of electron above the field $h$")
# plt.ylabel("Electric force felt at the center of the field $F$")
# plt.plot(h, F)
# plt.grid()
# plt.show()

message = ""
for i in range(len(h)):
    message += "(" + str(h[i]) + "," + str(F[i]) + "),"

pyperclip.copy(message)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, H, F, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()