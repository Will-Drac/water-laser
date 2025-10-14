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
        'water-sim.mp4'
    ],
    stdin=subprocess.PIPE
)

dx = 0.001
dy = dx
g = 9.81
H0 = 0.05

# dynamic dt based on stability condition
Courant = 0.4
dt = Courant * dx / (g * H0) ** 0.5

n_manning = 0.009  # acrylic-like bottom roughness

A_emit = 0.01   # amplitude (meters)
f_emit = 1     # frequency (Hz)
emit_x, emit_y = 500, 500


Q = ti.Vector.field(3, ti.f32, shape=size)   # [h, hu, hv]
Q_new = ti.Vector.field(3, ti.f32, shape=size)

@ti.kernel
def set_reflective_bc():
    # left/right (i=0 and i=nx-1 mirror i=1 and i=nx-2)
    for j in range(size[1]):
        # left wall mirrors cell 1
        Q[0, j][0] = Q[1, j][0]
        Q[0, j][1] = -Q[1, j][1]     # flip normal (hu) at vertical wall
        Q[0, j][2] = Q[1, j][2]
        # right wall mirrors cell nx-2
        Q[size[0] - 1, j][0] = Q[size[0] - 2, j][0]
        Q[size[0] - 1, j][1] = -Q[size[0] - 2, j][1]
        Q[size[0] - 1, j][2] = Q[size[0] - 2, j][2]

    # bottom/top (j=0 and j=ny-1 mirror j=1 and j=ny-2)
    for i in range(size[0]):
        Q[i, 0][0] = Q[i, 1][0]
        Q[i, 0][1] = Q[i, 1][1]
        Q[i, 0][2] = -Q[i, 1][2]     # flip normal (hv) at horizontal wall
        Q[i, size[1] - 1][0] = Q[i, size[1] - 2][0]
        Q[i, size[1] - 1][1] = Q[i, size[1] - 2][1]
        Q[i, size[1] - 1][2] = -Q[i, size[1] - 2][2]

@ti.func
def flux_x(Qcell):
    h, hu, hv = Qcell[0], Qcell[1], Qcell[2]
    u = 0.0 if h <= 0 else hu / h
    return ti.Vector([hu, hu * u + 0.5 * g * h * h, hu * (0.0 if h <= 0 else hv / h)])

@ti.func
def flux_y(Qcell):
    h, hu, hv = Qcell[0], Qcell[1], Qcell[2]
    v = 0.0 if h <= 0 else hv / h
    return ti.Vector([hv, hv * (0.0 if h <= 0 else hu / h), hv * v + 0.5 * g * h * h])

@ti.func
def rusanov_flux(QL, QR, normal):
    FL = ti.Vector([0.0, 0.0, 0.0])
    FR = ti.Vector([0.0, 0.0, 0.0])
    smax = 0.0
    if normal == 0:
        FL = flux_x(QL); FR = flux_x(QR)
        uL = 0.0 if QL[0] <= 0 else QL[1] / QL[0]
        uR = 0.0 if QR[0] <= 0 else QR[1] / QR[0]
        aL = ti.sqrt(g * QL[0]); aR = ti.sqrt(g * QR[0])
        smax = ti.max(ti.abs(uL) + aL, ti.abs(uR) + aR)
    else:
        FL = flux_y(QL); FR = flux_y(QR)
        vL = 0.0 if QL[0] <= 0 else QL[2] / QL[0]
        vR = 0.0 if QR[0] <= 0 else QR[2] / QR[0]
        aL = ti.sqrt(g * QL[0]); aR = ti.sqrt(g * QR[0])
        smax = ti.max(ti.abs(vL) + aL, ti.abs(vR) + aR)
    return 0.5 * (FL + FR) - 0.5 * smax * (QR - QL)


@ti.kernel
def step():
    for i, j in ti.ndrange((1, size[0] - 1), (1, size[1] - 1)):
        FxL = rusanov_flux(Q[i - 1, j], Q[i, j], 0)
        FxR = rusanov_flux(Q[i, j], Q[i + 1, j], 0)
        FyB = rusanov_flux(Q[i, j - 1], Q[i, j], 1)
        FyT = rusanov_flux(Q[i, j], Q[i, j + 1], 1)

        dQ = -dt / dx * (FxR - FxL) - dt / dy * (FyT - FyB)
        Q_new[i, j] = Q[i, j] + dQ

        # --- bottom friction ---
        h = Q[i, j][0]
        if h > 1e-6:
            u = Q[i, j][1] / h
            v = Q[i, j][2] / h
            s = ti.sqrt(u * u + v * v)
            cf = g * n_manning * n_manning * s / (h ** (4.0 / 3.0))
            Q_new[i, j][1] -= dt * cf * u * h
            Q_new[i, j][2] -= dt * cf * v * h

    for i, j in ti.ndrange((1, size[0] - 1), (1, size[1] - 1)):
        Q[i, j] = Q_new[i, j]


@ti.kernel
def init():
    for i, j in Q:
        x = (i - size[0] / 2) * dx
        y = (j - size[1] / 2) * dy
        # Q[i, j][0] = H0 + 0.02 * ti.exp(-200 * (x * x + y * y))  # h
        Q[i, j][0] = H0
        Q[i, j][1] = 0.0  # hu
        Q[i, j][2] = 0.0  # hv

@ti.kernel
def apply_emitter(t: ti.f32):
    for i, j in ti.ndrange((emit_x - 1, emit_x + 2), (emit_y - 1, emit_y + 2)):
        Q[i, j][0] = H0 + A_emit * ti.sin(2 * ti.math.pi * f_emit * t)
        Q[i, j][1] = 0.0
        Q[i, j][2] = 0.0


# init
init()
set_reflective_bc()

pixels = ti.Vector.field(3, ti.f32, shape=(size[0], size[1]))
@ti.kernel
def draw():
    for i, j in pixels:
        v = (Q[i, j][0] - H0) / H0 * 100

        vp = ti.math.clamp(v, 0, 1)
        vn = ti.math.clamp(-v, 0, 1)

        pixels[i, j] = ti.Vector([vp, 0, vn])

t = 0

for frame in range(int(simulation_time/dt)):
    set_reflective_bc()  # enforce BCs on ghost cells
    apply_emitter(t)
    step()
    draw()

    t += dt

    if (t % (1/60)) < ((t - dt) % (1/60)):
        img = (pixels.to_numpy() * 255).astype(np.uint8)
        ffmpeg.stdin.write(img.tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()