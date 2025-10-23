tank_size = 1, 1
size = 1000, 1000

import taichi as ti
import subprocess
import numpy as np

ti.init(arch=ti.gpu)

# setting up ffmpeg, rendered frames are going to get passed in here to make a video
ffmpeg = subprocess.Popen(
    [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{size[1]}x{size[0]}',
        '-r', '60',
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        "-pix_fmt", "yuv420p",
        '-preset', 'ultrafast',
        "-crf", "26",
        "light-sim.mp4"
    ],
    stdin=subprocess.PIPE
)

simulation_time = 35 #in seconds

c = ti.math.sqrt(0.25 * 9.81)
dx = tank_size[0] / size[0]

# dynamic dt based on stability condition
Courant = 0.4
dt = Courant * dx / c

f_emit = 5
omega_emit = 2*ti.math.pi * f_emit


# the electrons in the laser cavity
P = [(100, 100), (100, 400), (400, 100), (400, 400)]

electronPositions = ti.Vector.field(2, dtype=int, shape=(len(P)))
for i in range(len(P)):
    electronPositions[i] = ti.Vector([P[i][0], P[i][1]])

electronOscillations = ti.field(dtype=float, shape=(len(P), 3)) # stores the oscillation values for the last 3 time steps


# the pixels forming the image to display
pixels = ti.Vector.field(3, dtype=float, shape=size)

# the electric field (all polarization is the same, so it's 1-D)
F = ti.field(dtype=float, shape=(size[0], size[1], 3)) # stores the field values for the last 3 time steps


@ti.kernel
def reset():
    for i, j in pixels:
        F[i, j, 0] = 0
        F[i, j, 1] = 0
        F[i, j, 2] = 0
        pixels[i, j] = ti.Vector([0, 0, 0])

    for i in electronPositions:
        electronOscillations[i, 0] = 0
        electronOscillations[i, 1] = 0
        electronOscillations[i, 2] = 0


@ti.func
def sample_border(field, x, y, z):
    output = float(0)
    if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and 0 <= z < field.shape[2]:
        output = field[x, y, z]

    return output


@ti.kernel
def update(currentI: ti.i32):
    beforeLastI = int((currentI + 1) % 3)
    lastI = int((currentI + 2) % 3)

    K = (c*dt/dx)**2

    for i, j in pixels:
        # updating the electric field
        A = 2 * sample_border(F, i, j, lastI) - sample_border(F, i, j, beforeLastI)
        B = (sample_border(F, i+1, j, lastI) - 2*sample_border(F, i, j, lastI) + sample_border(F, i-1, j, lastI))
        C = (sample_border(F, i, j+1, lastI) - 2*sample_border(F, i, j, lastI) + sample_border(F, i, j-1, lastI))

        F[i, j, currentI] = A + K*(B + C)

    for i in electronPositions:
        # getting the acceleration
        xL = electronOscillations[i, lastI]
        xB = electronOscillations[i, beforeLastI]

        dxdt = (xL - xB) / dt
        gamma = -0.2 + 1000 * xL**2
        omega2_x = omega_emit**2 * xL
        E = 0.3 * F[electronPositions[i][0], electronPositions[i][1], lastI]
        R = 0

        dx2dt2 = -dxdt * gamma - omega2_x + E + R

        # updating the electric field from this electron
        F[electronPositions[i][0], electronPositions[i][1], currentI] += 0.03 * dx2dt2

        # updating the electron's position
        electronOscillations[i, currentI] = dx2dt2 * dt**2 + 2 * xL - xB

@ti.kernel
def draw(currentI: ti.i32):
    # for i, j in pixels:
    #     # later i'll make this prettier
    #     v = 10 * F[i, j, currentI]

    #     vp = ti.math.clamp(v, 0, 1)
    #     vn = ti.math.clamp(-v, 0, 1)

    #     pixels[i, j] = ti.Vector([vp, 0, vn])

    for i, j in pixels:
        lastI = int((currentI + 2) % 3)

        x = F[i, j, currentI]
        v = (x - F[i, j, lastI]) / dt

        c = x**2 + (v/omega_emit)**2

        pixels[i, j] = ti.Vector([c, c, c])



I = 0
t = 0

reset()

for frame in range(int(simulation_time/dt)):

    # updating the emitter
    if (t < 1/f_emit):
        F[250, 250, I] = 1*ti.math.sin(omega_emit * t)

    I = (I + 1) % 3
    t += dt

    update(currentI=I)
    draw(currentI=I)

    if (t % (1/60)) < ((t - dt) % (1/60)):
        img = (pixels.to_numpy() * 255).astype(np.uint8)
        ffmpeg.stdin.write(img.tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()