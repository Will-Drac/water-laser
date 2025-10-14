simulation_time = 5 #in seconds

import taichi as ti
import subprocess
import numpy as np

ti.init(arch=ti.gpu)

size = 1000, 1000

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
        "-crf", "30",
        "light-sim.mp4"
    ],
    stdin=subprocess.PIPE
)

c = 0.700357
dx = 0.001

# dynamic dt based on stability condition
Courant = 0.4
dt = Courant * dx / c

f_emit = 1

pixels = ti.Vector.field(3, dtype=float, shape=size)

# the electric field (all polarization is the same, so it's 1-D)
F = ti.field(dtype=float, shape=(size[0], size[1], 3))


@ti.kernel
def reset():
    for i, j in pixels:
        F[i, j, 0] = 0
        F[i, j, 1] = 0
        F[i, j, 2] = 0
        pixels[i, j] = ti.Vector([0, 0, 0])


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
        A = 2 * sample_border(F, i, j, lastI) - sample_border(F, i, j, beforeLastI)
        B = (sample_border(F, i+1, j, lastI) - 2*sample_border(F, i, j, lastI) + sample_border(F, i-1, j, lastI))
        C = (sample_border(F, i, j+1, lastI) - 2*sample_border(F, i, j, lastI) + sample_border(F, i, j-1, lastI))

        F[i, j, currentI] = A + K*(B + C)


@ti.kernel
def draw(currentI: ti.i32):
    for i, j in pixels:
        # later i'll make this prettier
        v = 10 * F[i, j, currentI]

        vp = ti.math.clamp(v, 0, 1)
        vn = ti.math.clamp(-v, 0, 1)

        pixels[i, j] = ti.Vector([vp, 0, vn])


I = 0
t = 0

reset()

for frame in range(int(simulation_time/dt)):

    # updating the emitter
    F[500, 500, I] = ti.math.sin(2*ti.math.pi * f_emit * t)

    I = (I + 1) % 3
    t += dt

    update(currentI=I)
    draw(currentI=I)

    if (t % (1/60)) < ((t - dt) % (1/60)):
        img = (pixels.to_numpy() * 255).astype(np.uint8)
        ffmpeg.stdin.write(img.tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()