# LieGroup

A lightweight and modular Python framework for Lie groups and Lie algebras, designed for robotics, control theory, estimation, and optimization research.

This library provides a clean, extensible base class for Lie groups, and implementations for common groups such as:

- **SE(2)** — Planar rigid transformations
- **SE(3)** — 3D rigid transformations
- **SO(2)** — 2D rotations
- **SO(3)** — 3D rotations
- **U(1)** — The unit circle group (complex numbers of magnitude 1)
- **UnitQuaternions** — S³ rotation group (quaternion unit sphere)
- **ℝ Additive** — Real numbers under addition
- **ℝⁿ Additive** — Vectors under addition (factory generated)
- **ℝ⁺ Multiplicative** — Positive reals under multiplication

Full support for exponential/log maps, Jacobians, adjoints, random sampling, and Lie brackets.

Full test suite using **pytest**, checking group axioms, inverses, Lie algebra consistency, adjoint properties, and Jacobian behavior.

---

## Features

- Abstract base class `LieGroup` with type generics
- Group operations: `compose`, `inverse`, `identity`
- Algebra operations: `exp`, `log`, `hat`, `vee`
- Adjoint representations and left Jacobians
- Random sampling (`randn`, `randu`)
- Distance functions on manifolds
- Full test coverage with multiple random trials
- Extended Kalman Filter (EKF) example on general Lie groups
- Trajectory visualization tools for different groups

---

## Installation

```bash
git clone https://github.com/miggyval/liegroups.git
cd liegroup
pip install -r requirements.txt
```

---

## Usage Example

```python
from liegroup import SE2, SO3

# Create a random element
X = SE2.Exp(np.random.randn(3))

# Compose two elements
Y = SE2.Exp(np.random.randn(3))
Z = SE2.compose(X, Y)

# Logarithmic map
v = SE2.Log(Z)

# Exponential map
Z_back = SE2.Exp(v)

# Distance
d = SE2.distance(X, Y)

print(f"Distance between X and Y: {d}")
```

---

## Testing

All operations are tested over random samples using `pytest`.

To run all tests:

```bash
pytest tests/
```

Tests include:
- Group axioms: identity, associativity, inverses
- Exponential and logarithm consistency
- Adjoint actions and Lie bracket properties (Jacobi identity, antisymmetry)
- Jacobian inverses and perturbation consistency

---

## EKF Example

An Extended Kalman Filter is provided for any Lie group:

```python
from liegroup import SO2, EKF

ekf = EKF(SO2)
t_arr, V_arr, X_arr = ekf.run()
```

Trajectory plotting included for 2D, 3D, and rotational groups.

---

## Structure

| Module | Description |
|:---|:---|
| `liegroup.py` | Base class and group implementations |
| `tests/test_liegroup.py` | Full pytest suite |
| `ekf_demo.py` | EKF simulation and trajectory visualization |

---

## Supported Lie Groups

- **SE(2)**: Rigid body transformations in 2D
- **SE(3)**: Rigid body transformations in 3D
- **SO(2)**: Rotations in 2D
- **SO(3)**: Rotations in 3D
- **U(1)**: Unit complex numbers
- **UnitQuaternions**: 3D rotations using quaternions
- **ℝ**: Real numbers under addition
- **ℝⁿ**: Vectors under addition
- **ℝ⁺**: Positive real numbers under multiplication

---

## Future Plans

- Add support for **Torus groups** (S¹ × S¹)
- Add **SE(n)** and **SO(n)** for arbitrary dimensions
- Stochastic differential equation (SDE) integration on manifolds
- Batch optimization and factor graph support

---

## License

MIT License — free for academic, research, and commercial use.

---
