import numpy as np
from liegroup import (
    product_groups_factory,
    SO3,
    SO2,
    RnAdd_factory,
)

# --- Build additive ℝⁿ groups ---
R3 = RnAdd_factory(3)
R2 = RnAdd_factory(2)

# --- Build product classes ---
SO3_R3 = product_groups_factory(SO3, R3)
SO2_R2 = product_groups_factory(SO2, R2)

print(SO3_R3.__name__)  # → Product_SO3_R3
print(SO2_R2.__name__)  # → Product_SO2_R2

# --- Identity elements ---
I_so3r3 = SO3_R3.identity()
I_so2r2 = SO2_R2.identity()
print("SO3×R3 identity:", I_so3r3)
print("SO2×R2 identity:", I_so2r2)

# --- Sample random elements ---
X3 = SO3_R3.randu()
X2 = SO2_R2.randu()
print("Random SO3×R3 element:", X3)
print("Random SO2×R2 element:", X2)

# --- Exponential map from a tangent vector ---
# Flat vectors for tangent spaces
vec_so3r3 = np.array([0.1, -0.2, 0.3, 1.0, 2.0, 3.0])  # (ω, v)
vec_so2r2 = np.array([0.4, 10.0, -5.0])                # (θ, v)

elem_so3r3 = SO3_R3.Exp(vec_so3r3)
elem_so2r2 = SO2_R2.Exp(vec_so2r2)

print("SO3×R3 Exp:", elem_so3r3)
print("SO2×R2 Exp:", elem_so2r2)

# --- Compose two SO3×R3 elements ---
delta_vec3 = np.array([0.05, -0.1, 0.0, -1.0, 0.5, 0.2])
Y3 = SO3_R3.Exp(delta_vec3)
Z3 = SO3_R3.compose(elem_so3r3, Y3)
print("Compose SO3×R3:", Z3)

# --- Inverse + distance for SO3×R3 ---
inv_Z3 = SO3_R3.inverse(Z3)
dist3 = SO3_R3.distance(elem_so3r3, Y3)
print("Inverse of Z3:", inv_Z3)
print("Distance between elements in SO3×R3:", dist3)

# --- Compose two SO2×R2 elements ---
delta_vec2 = np.array([0.2, 1.5, -3.0])
Y2 = SO2_R2.Exp(delta_vec2)
Z2 = SO2_R2.compose(elem_so2r2, Y2)
print("Compose SO2×R2:", Z2)

# --- Inverse + distance for SO2×R2 ---
inv_Z2 = SO2_R2.inverse(Z2)
dist2 = SO2_R2.distance(elem_so2r2, Y2)
print("Inverse of Z2:", inv_Z2)
print("Distance between elements in SO2×R2:", dist2)

# --- Log maps back to vector space ---
vec_back_so3r3 = SO3_R3.Log(elem_so3r3)
vec_back_so2r2 = SO2_R2.Log(elem_so2r2)
print("SO3×R3 Log:", vec_back_so3r3)
print("SO2×R2 Log:", vec_back_so2r2)

# --- Verify exp-log roundtrip ---
print("SO3×R3 exp-log error:", np.linalg.norm(vec_so3r3 - vec_back_so3r3))
print("SO2×R2 exp-log error:", np.linalg.norm(vec_so2r2 - vec_back_so2r2))
