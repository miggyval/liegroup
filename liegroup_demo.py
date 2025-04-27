import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import quaternion

from liegroup import SO2, SE2, SO3, U1, UnitQuat

# --- Helper functions ---

def animate_SO2():
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid()
    arrow, = ax.plot([], [], 'r-', lw=3)

    def init():
        arrow.set_data([], [])
        return (arrow,)

    def update(frame):
        theta = frame
        R = SO2.Exp(np.array([theta]))
        vec = R @ np.array([1, 0])
        arrow.set_data([0, vec[0]], [0, vec[1]])
        return (arrow,)

    frames = np.linspace(0, 2*np.pi, 120)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)
    plt.title("SO(2) rotation")
    plt.show()

def animate_SE2():
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid()

    frame_lines, = ax.plot([], [], 'b-', lw=2)
    arrow, = ax.plot([], [], 'r-', lw=2)

    def init():
        frame_lines.set_data([], [])
        arrow.set_data([], [])
        return (frame_lines, arrow)

    def update(frame):
        theta = frame
        T = SE2.Exp(np.array([theta, theta / 2, 1]))
        R = T[:2, :2]
        p = T[:2, 2]
        X_axis = p + R @ np.array([0.3, 0])
        Y_axis = p + R @ np.array([0, 0.3])
        frame_lines.set_data([p[0], X_axis[0], p[0], Y_axis[0]],
                             [p[1], X_axis[1], p[1], Y_axis[1]])
        arrow.set_data([p[0]], [p[1]])
        return (frame_lines, arrow)

    frames = np.linspace(0, 2*np.pi, 200)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)
    plt.title("SE(2) moving frame")
    plt.show()

def animate_U1():
    fig, ax = plt.subplots()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid()
    circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax.add_artist(circle)
    dot, = ax.plot([], [], 'ro')

    def init():
        dot.set_data([], [])
        return (dot,)

    def update(frame):
        phase = frame
        z = U1.Exp(np.array([phase]))
        dot.set_data([z.real], [z.imag])
        return (dot,)

    frames = np.linspace(0, 2*np.pi, 200)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)
    plt.title("U(1) phase circle")
    plt.show()

def animate_SO3():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = [ax.plot([], [], [], lw=2)[0] for _ in range(3)]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        theta = frame
        R = SO3.Exp(np.array([0.5*np.sin(theta), 0.5*np.cos(theta), 0.5*np.sin(2*theta)]))
        origin = np.zeros(3)
        axes = np.eye(3)
        for i in range(3):
            vec = R @ axes[:, i]
            lines[i].set_data([origin[0], vec[0]], [origin[1], vec[1]])
            lines[i].set_3d_properties([origin[2], vec[2]])
        colors = ['r', 'g', 'b']
        for line, c in zip(lines, colors):
            line.set_color(c)
        return lines

    frames = np.linspace(0, 2*np.pi, 200)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=30)
    plt.title("SO(3) rotating frame")
    plt.show()

def animate_UnitQuat():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = [ax.plot([], [], [], lw=2)[0] for _ in range(3)]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        theta = frame
        axis = np.array([np.cos(theta), np.sin(theta), np.cos(2*theta)])
        axis = axis / np.linalg.norm(axis)
        q = UnitQuat.Exp(axis * theta)
        R = quaternion.as_rotation_matrix(q)
        origin = np.zeros(3)
        axes = np.eye(3)
        for i in range(3):
            vec = R @ axes[:, i]
            lines[i].set_data([origin[0], vec[0]], [origin[1], vec[1]])
            lines[i].set_3d_properties([origin[2], vec[2]])
        colors = ['r', 'g', 'b']
        for line, c in zip(lines, colors):
            line.set_color(c)
        return lines

    frames = np.linspace(0, 2*np.pi, 200)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=30)
    plt.title("Unit Quaternion rotation")
    plt.show()

# --- Main launcher ---

def main():
    animate_SO2()
    animate_SE2()
    animate_U1()
    animate_SO3()
    animate_UnitQuat()

if __name__ == "__main__":
    main()
