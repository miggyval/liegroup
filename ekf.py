from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from liegroup import *

class EKF:
    def __init__(self, lg: LieGroup):
        self.lg = lg

    def run(self, X0=None, S0=None, T=10000, dt=0.01, control_fn=None):
        N = self.lg.N
        
        # Initialize states
        X = self.lg.identity() if X0 is None else X0
        S = np.zeros(N) if S0 is None else S0
        P = 1e-6 * np.eye(N)
        
        # Noise and integration settings
        M = np.eye(N)
        Q = np.diag(0.01 * np.ones(N))
        R = np.diag(0.01 * np.ones(N))
        
        # Storage
        t_arr = np.zeros(T)
        V_arr = np.zeros((T, N))
        X_arr = np.zeros((T, N))  # Store Log(X)
        
        V = np.zeros(N)  # initial body velocity

        for k in range(T):
            t = k * dt
            t_arr[k] = t
            
            # Control input
            if control_fn is not None:
                u = control_fn(t)
            else:
                u = np.random.randn(N)  # random control if none provided

            tau = np.random.multivariate_normal(np.zeros(N), Q)
            sigma = np.random.multivariate_normal(np.zeros(N), R)
            
            Vdot = np.linalg.pinv(M) @ u + tau
            V_new = V + Vdot * dt

            # Prediction
            X_prior = self.lg.oplus(X, V * dt)
            A = self.lg.adjoint(self.lg.Exp(V * dt))
            P_prior = A @ P @ A.T + Q

            # Observation (Z)
            Z = self.lg.oplus(X, sigma)
            y = self.lg.ominus(Z, X_prior)

            # Correction
            H = np.eye(N)
            S = H @ P_prior @ H.T + R
            K = P_prior @ H.T @ np.linalg.pinv(S)
            X = self.lg.oplus(X_prior, K @ y)
            P = (np.eye(N) - K @ H) @ P_prior

            # Store
            V_arr[k, :] = V
            X_arr[k, :] = self.lg.Log(X)  # store group element as a vector
            V = V_new

        return t_arr, V_arr, X_arr

def plot_trajectory(lg: LieGroup, t_arr: np.ndarray, X_arr: np.ndarray):
    N = lg.N
    
    plt.figure()
    
    if isinstance(lg, SO2) or isinstance(lg, U1):
        # SO(2) and U(1) are just a single angle
        plt.plot(t_arr, np.unwrap(X_arr[:, 0]))
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [rad]')
        plt.title(f'Trajectory on {lg.__class__.__name__}')
        plt.grid(True)

    elif isinstance(lg, SO3) or isinstance(lg, UnitQuat):
        # SO(3) and Unit quaternions: plot 3 components of Log
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))
        labels = ['ω₁', 'ω₂', 'ω₃']
        for i in range(3):
            axs[i].plot(t_arr, X_arr[:, i])
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)
        axs[-1].set_xlabel('Time [s]')
        fig.suptitle(f'Rotation components on {lg.__class__.__name__}')
        plt.tight_layout()

    elif isinstance(lg, SE2):
        # SE(2): trajectory in x, y plane
        x = X_arr[:, 1]  # X and Y stored in [θ, x, y]
        y = X_arr[:, 2]
        plt.plot(x, y)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trajectory in SE(2)')
        plt.axis('equal')
        plt.grid(True)

    elif isinstance(lg, SE3):
        # SE(3): trajectory in x, y, z
        x = X_arr[:, 3]
        y = X_arr[:, 4]
        z = X_arr[:, 5]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title('Trajectory in SE(3)')

    else:
        # fallback: plot all components vs time
        for i in range(N):
            plt.plot(t_arr, X_arr[:, i], label=f'Component {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Components')
        plt.title(f'Trajectory on {lg.__class__.__name__}')
        plt.legend()
        plt.grid(True)

def ekf_demo():
    seed = 1234  # fixed seed for reproducibility
    
    
    np.random.seed(seed)
    lg = SO2()  # Change this to SE2, SE3, SO3, U1, UnitQuat, etc.
    ekf = EKF(lg)
    t_arr, V_arr, X_arr = ekf.run()
    plot_trajectory(lg, t_arr, X_arr)
    
    np.random.seed(seed)
    lg = U1()  # Change this to SE2, SE3, SO3, U1, UnitQuat, etc.
    ekf = EKF(lg)
    t_arr, V_arr, X_arr = ekf.run()
    plot_trajectory(lg, t_arr, X_arr)
    plt.show()


if __name__ == "__main__":
    ekf_demo()
