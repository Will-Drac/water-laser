import taichi as ti
import subprocess
import numpy as np

ti.init(arch=ti.gpu)

tank_size = 0.5 # in meters
H = 0.025  # original tank height before waves

size = 1000
view_height = int(2*H*size / tank_size)

pixels = ti.Vector.field(3, ti.f32, shape=(view_height, size))

# setting up ffmpeg, rendered frames are going to get passed in here to make a video
ffmpeg = subprocess.Popen(
    [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{size}x{view_height}',
        '-r', '60',
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        "-pix_fmt", "yuv420p",
        '-preset', 'ultrafast',
        "-crf", "20",
        "simulation/1d water/sim_working_somehow.mp4"
    ],
    stdin=subprocess.PIPE
)

simulation_time = 25  # in seconds

dx = tank_size/size

g = 9.81

# dynamic dt based on stability condition
Courant = 0.1
wave_speed = (g * H) ** 0.5
print(wave_speed)
dt = Courant * dx / wave_speed

viscous_drag = 1  # drag against the bottom of the tank
# ignoring viscosity
density = 1 #should be ~1000 but this is nicer for the numbers

# 3 entries for (oldest, old, new), although they change jobs cyclicly
h = ti.Vector.field(3, ti.f32, shape=size)
u = ti.Vector.field(3, ti.f32, shape=size)


@ti.func
def reflect_sample(field: ti.template(), slice: ti.i32, pos: ti.i32, max_size: ti.i32) -> ti.f32:
    reflect_pos = 0

    if pos < 0:
        reflect_pos = -pos
    elif pos >= max_size:
        reflect_pos = 2 * max_size - 2 - pos
    else:
        reflect_pos = pos

    return field[reflect_pos][slice]


@ti.kernel
def setup():
    for i in h:
        h[i] = ti.Vector([0, 0, 0])
        u[i] = ti.Vector([0, 0, 0])

# @ti.func
# # applies a gaussian beam of pressure at position `center` (in tank space), with `spread` (1 standard deviation in tank space), normalized with a multiple `strength`, then the gradient is calculated at point `sample`
# def get_pressure_gradient(sample: ti.f32, center: ti.f32, spread: ti.f32, strength: ti.f32):


@ti.kernel
def update(this_slice: ti.i32, time: ti.f32):
    last_slice = (this_slice - 1) % 3
    next_slice = (this_slice + 1) % 3
    before_last_slice = next_slice

    oscillator_velocity = (oscillator_pos[this_slice] - oscillator_pos[last_slice]) / dt

    for i in h:
        h_ll = reflect_sample(h, this_slice, i-2, size)
        h_l = reflect_sample(h, this_slice, i-1, size)
        h_r = reflect_sample(h, this_slice, i+1, size)
        h_rr = reflect_sample(h, this_slice, i+2, size)

        u_ll = reflect_sample(u, this_slice, i-2, size)
        u_l = reflect_sample(u, this_slice, i-1, size)
        u_r = reflect_sample(u, this_slice, i+1, size)
        u_rr = reflect_sample(u, this_slice, i+2, size)

        s_f32 = ti.cast(size, ti.f32)
        i_f32 = ti.cast(i, ti.f32)

        starting = ti.math.min(time, 1)
        pressure_oscillator = (10000000 * oscillator_velocity+1000*starting) * ti.math.exp(-10/s_f32 * (i_f32-s_f32/2)**2)
        del_pressure_oscillator_del_x = -2/size * (i-size/2) * pressure_oscillator

        pressure_external_pos = 400
        pressure_external_strength = 10000 * ti.max(-5*ti.abs(time-2)+1, 0)
        pressure_external = pressure_external_strength * ti.math.exp(-10/s_f32 * (i_f32-pressure_external_pos)**2)
        del_pressure_external_del_x = -2/size * (i-pressure_external_pos) * pressure_external
        del_pressure_external_del_x = 0

        del_pressure_del_x = del_pressure_oscillator_del_x + del_pressure_external_del_x

        del_h_del_x = -h_ll - 2*h_l + 2*h_r + h_rr
        del_h_del_x /= 8 * dx

        del_u_del_x = -u_ll - 2*u_l + 2*u_r + u_rr
        del_u_del_x /= 8 * dx

        u[i][next_slice] = 4*dt/3 * (-g*del_h_del_x - u[i][this_slice]*del_u_del_x - viscous_drag*u[i][this_slice] - del_pressure_del_x/density) + 2/3*u[i][last_slice] + 1/3*u[i][before_last_slice]

        del_hHu_del_x = -(H+h_ll)*u_ll - 2*(H+h_l)*u_l + 2*(H+h_r)*u_r + (H+h_rr)*u_rr
        del_hHu_del_x /= 8 * dx

        h[i][next_slice] = 4*dt/3 * (-del_hHu_del_x) + 2/3*h[i][this_slice] + 1/3*h[i][before_last_slice]

# the water should stop moving at the boundaries of the tank
@ti.kernel
def damp_edge_velocity(this_slice: ti.i32):
    damping_width = 10
    for i in u:
        w = 1.0
        if i < damping_width:
            s = ti.cast(i, ti.f32) / ti.cast(damping_width, ti.f32)
            w = s
        elif i > size - 1 - damping_width:
            s = ti.cast(size - 1 - i, ti.f32) / ti.cast(damping_width, ti.f32)
            w = s
        u[i][this_slice] *= w


@ti.kernel
def draw(current_slice: ti.i32):
    for i, j in pixels:
        water_height = -h[j][current_slice] + H
        water_height_pixels = water_height * view_height/(2*H)
        if (i < water_height_pixels):
            pixels[i, j] = ti.Vector([0, 0, 0])
        else:
            pixels[i, j] = ti.Vector([0, 0.2, 1]) + 15*abs(u[j][current_slice])

        pixels[i, j] = ti.math.clamp(pixels[i, j], 0, 1)

setup()

oscillator_pos = ti.field(dtype=ti.f32, shape=(3))
oscillator_mass = 1
oscillator_damping = -50
oscillator_natural_frequency = wave_speed/(60/1000)

@ti.kernel
def update_oscillator(current_slice: ti.i32, oscillator_damping: ti.f32):
    next_slice = (current_slice + 1) % 3
    last_slice = (current_slice - 1) % 3

    loc = ti.cast(size/2, ti.i32)

    local_F = h[loc][current_slice]

    oscillator_pos[next_slice] = (
        dt**2 * local_F / oscillator_mass -
        dt*oscillator_damping/oscillator_mass*(oscillator_pos[current_slice] - oscillator_pos[last_slice]) -
        dt**2 * (oscillator_natural_frequency**2 + oscillator_damping**2/4)*oscillator_pos[current_slice] +
        2*oscillator_pos[current_slice] -
        oscillator_pos[last_slice]
    )

t = 0
current_slice = 0
for frame in range(int(simulation_time/dt)):

    update(current_slice, t)
    damp_edge_velocity(current_slice)
    damping = -35
    if (t > 15): damping = 35
    update_oscillator(current_slice, damping)

    draw(current_slice)

    t += dt
    current_slice = (current_slice + 1) % 3

    if (t % (1/60)) < ((t - dt) % (1/60)):
        img = (pixels.to_numpy() * 255).astype(np.uint8)
        ffmpeg.stdin.write(img.tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()
