from abc import ABC, abstractmethod
import numpy as np
import quaternion
from scipy.linalg import block_diag
from typing import Generic, TypeVar, get_origin, get_args, Optional, Type, Tuple, Any
from numpy.typing import NDArray

g = TypeVar('g')
G = TypeVar('G')
F = NDArray[np.floating[Any]]


EPS = 1e-6

def extract_datatypes(group_cls: type) -> tuple[type, type]:
    """
    Given a subclass of LieGroup[G, g], return (G, g).
    Falls back to group_cls.GroupElement and F if it can't find them.
    """
    # look for a base of the form LieGroup[SomeType, OtherType]
    for base in getattr(group_cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is LieGroup:
            args = get_args(base)
            if len(args) == 2:
                return args  # this is (G_type, g_type)
    # fallback: use the GroupElement alias (or F) and default g to F
    G_type = getattr(group_cls, "GroupElement", F)
    return (G_type, F)

class LieGroup(ABC, Generic[G, g]):
    N: int
    
    GroupElement: type = F
    
    @classmethod
    def _check_hat_input(cls, v: F) -> None:
        assert isinstance(v, np.ndarray), f"hat input must be F, got {type(v)}"
        assert v.shape == (cls.N,), f"hat expects shape ({cls.N},), got {v.shape}"

    @classmethod
    def _check_vee_output(cls, out: F) -> None:
        assert isinstance(out, np.ndarray), f"vee output must be F, got {type(out)}"
        assert out.shape == (cls.N,), f"vee must return shape ({cls.N},), got {out.shape}"
    
    @classmethod
    @abstractmethod
    def identity(cls) -> G:
        pass

    @classmethod
    @abstractmethod
    def compose(cls, X1: G, X2: G) -> G:
        pass

    @classmethod
    @abstractmethod
    def inverse(cls, X: G) -> G:
        pass

    @classmethod
    @abstractmethod
    def log(cls, X: G) -> g:
        pass

    @classmethod
    @abstractmethod
    def exp(cls, v: g) -> G:
        pass

    @classmethod
    @abstractmethod
    def hat(cls, vx: F) -> g:
        pass

    @classmethod
    @abstractmethod
    def vee(cls, v: g) -> F:
        pass

    @classmethod
    @abstractmethod
    def adjoint(cls, X: G) -> F:
        pass
    
    @classmethod
    @abstractmethod
    def left_jacobian(cls, v: F) -> F:
        pass

    @classmethod
    @abstractmethod
    def left_jacobian_inverse(cls, v: F) -> F:
        pass
    
    @classmethod
    @abstractmethod
    def lie_bracket(cls, v: F, w: F) -> F:
        pass
    
    @classmethod
    @abstractmethod
    def randu(cls) -> G:
        pass

    @classmethod
    def Log(cls, X: G) -> F:
        return cls.vee(cls.log(X))

    @classmethod
    def Exp(cls, v: F) -> G:
        return cls.exp(cls.hat(v))

    @classmethod
    def oplus(cls, X: G, v: F) -> G:
        return cls.compose(X, cls.Exp(v))

    @classmethod
    def ominus(cls, X: G, Y: G) -> F:
        return cls.Log(cls.compose(cls.inverse(X), Y))
    
    @classmethod
    def randn_G(cls, *, scale: float=1.0, cov: Optional[F]=None) -> G:
        vx = cls.randn_g(scale=scale, cov=cov)
        return cls.normalize(cls.exp(vx))
    
    @classmethod
    def randn_g(cls, *, scale: float=1.0, cov: Optional[F]=None) -> g:        
        if cov is not None:
            assert isinstance(cov, np.ndarray) and np.issubdtype(cov.dtype, np.floating)
            assert cov.shape == (cls.N, cls.N)
            v = np.random.multivariate_normal(np.zeros((cls.N,)), cov)
        else:
            v = np.random.randn(cls.N) * scale
        return cls.hat(v)
    
    
    @classmethod
    def normalize(cls, X: G) -> G:
        """Default: no normalization needed."""
        return X
    
    @classmethod
    def distance(cls, X1: G, X2: G) -> np.floating[Any]:
        return np.linalg.norm(cls.ominus(X2, X1))

    @classmethod
    def isclose_G(cls, X1: G, X2: G, atol: float=1e-5) -> np.bool:
        return cls.distance(X1, X2) < atol
    
    @classmethod
    def isclose_g(cls, v1: g, v2: g, atol: float=1e-5) -> np.bool:
        return np.linalg.norm(cls.vee(v1) - cls.vee(v2)) < atol
    
# Factory function for product Lie groups
def product_groups_factory(*groups: Type[LieGroup[Any, Any]]) -> Type[LieGroup[Tuple[G], Tuple[g]]]:
    """
    Factory function that creates a Lie group class for the Cartesian product
    of multiple Lie groups provided as arguments (e.g. G1 x G2 x G3 ...).
    
    Arguments:
    *groups -- Any number of Lie group classes.
    
    Returns:
    A new class representing the Cartesian product of the input groups.
    """
    
    # Extract types for each group (G_i, g_i) using the extract_datatypes helper
    type_pairs = [extract_datatypes(Gi) for Gi in groups]
    
    # Create a custom class name based on the names of the groups
    class_name = "x".join(Gi.__name__ for Gi in groups)
    group_dims = [Gi.N for Gi in groups]
    cumdims = np.cumsum([0] + group_dims)
    
    # Define the ProductGroup class that represents the Cartesian product
    class ProductGroup(LieGroup[Tuple[G], Tuple[g]]):
        """Lie group representing the Cartesian product of the input groups."""
        
        # Total dimension: sum of the dimensions of each group
        N = sum(Gi.N for Gi in groups)
        
        # Store the extracted types for each group's elements and algebra
        ElementTypes: list[type] = [G for G, _ in type_pairs]
        AlgebraTypes: list[type] = [g for _, g in type_pairs]
        
        @classmethod
        def identity(cls) -> tuple[G]:
            """Return the identity element of the product group."""
            return tuple(g.identity() for g in groups)
        
        @classmethod
        def compose(cls, X1: tuple[G], X2: tuple[G]) -> tuple[G]:
            """Compose two elements (X1, X2, ..., Xn) of the product group."""
            return tuple(g.compose(x1, x2) for g, (x1, x2) in zip(groups, zip(X1, X2)))
        
        @classmethod
        def inverse(cls, X: tuple[G]) -> tuple[G]:
            """Inverse of an element (X1, X2, ..., Xn) in the product group."""
            return tuple(g.inverse(x) for g, x in zip(groups, X))
        
        @classmethod
        def exp(cls, v: tuple[g]) -> tuple[G]:
            """Exponential map for the product group: apply exp to each component."""
            return tuple(g.exp(vi) for g, vi in zip(groups, v))
        
        @classmethod
        def log(cls, X: tuple[G]) -> tuple[g]:
            """Logarithmic map for the product group: apply log to each component."""
            return tuple(g.log(x) for g, x in zip(groups, X))
        
        @classmethod
        def hat(cls, v: F) -> tuple[g]:
            """Apply hat map to each component of v."""
            parts = [v[cumdims[i]:cumdims[i+1]] for i in range(len(groups))]
            return tuple(g.hat(part) for g, part in zip(groups, parts))
        
        @classmethod
        def vee(cls, v: tuple[g]) -> F:
            """Apply vee map to each component of v."""
            parts = [g.vee(vi) for g, vi in zip(groups, v)]
            return np.concatenate(parts)
        
        @classmethod
        def adjoint(cls, X: tuple[G]) -> F:
            """Adjoint map for the product group: apply adjoint to each component."""
            blocks = [g.adjoint(x) for g, x in zip(groups, X)]
            return block_diag(*blocks).astype(np.float64)
        

        @classmethod
        def left_jacobian(cls, v: F) -> F:
            parts = [v[cumdims[i]:cumdims[i+1]] for i in range(len(groups))]
            blocks = [g.left_jacobian(v) for g, v in zip(groups, parts)]
            return block_diag(*blocks)
            

        @classmethod
        def left_jacobian_inverse(cls, v: F) -> F:
            parts = [v[cumdims[i]:cumdims[i+1]] for i in range(len(groups))]
            blocks = [g.left_jacobian_inverse(v) for g, v in zip(groups, parts)]
            return block_diag(*blocks)
        
        @classmethod
        def lie_bracket(cls, v: F, w: F) -> F:
                parts_v = [v[cumdims[i]:cumdims[i+1]] for i in range(len(groups))]
                parts_w = [w[cumdims[i]:cumdims[i+1]] for i in range(len(groups))]
                parts_bracket = [g.lie_bracket(vi, wi) for g, vi, wi in zip(groups, parts_v, parts_w)]
                return np.concatenate(parts_bracket)
        
        @classmethod
        def randu(cls) -> tuple[G]:
            """Uniform random sampling for the product Lie group."""
            # Generate random elements by calling randu for each group in the product
            return tuple(g.randu() for g in groups)
        
        @classmethod
        def normalize(cls, X: tuple[G]) -> tuple[G]:
            """Normalize each element of the product group separately."""
            return tuple(g.normalize(x) for g, x in zip(groups, X))

    # Set the custom name for the class
    ProductGroup.__name__ = class_name
    return ProductGroup


def robot_factor_se2(types: str):
    class Joint():
        @abstractmethod
        def get_exp(self, theta):
            pass
        
    class RevoluteJoint(Joint):
        def __init__(self):
            self.theta = 0.0
            self.S = np.array([1.0, 0.0, 0.0])
        def get_exp(self, theta):
            return SE2.Exp(theta * self.S)
        

    class PrismaticJoint():
        def __init__(self, vx, vy):
            norm = np.sqrt(vx, vy)
            vx /= norm
            vy /= norm
            self.theta = 0.0
            self.S = np.array([0.0, vx, vy])
        def get_exp(self, theta):
            return SE2.Exp(theta * self.S)
    
    class Robot():
        for c in types:
            if c == ord('R'):
                joint = RevoluteJoint()
            elif c == ord('P'):
                joint = PrismaticJoint()
                

class RAdd(LieGroup[np.float64, np.float64]):
    N: int = 1
    
    @classmethod
    def identity(cls) -> np.float64:
        return np.float64(0.0)

    @classmethod
    def compose(cls, x1: np.float64, x2: np.float64) -> np.float64:
        return x1 + x2

    @classmethod
    def inverse(cls, x: np.float64) -> np.float64:
        return -x

    @classmethod
    def hat(cls, v: F) -> np.float64:
        return np.float64(v[0])

    @classmethod
    def vee(cls, vx: np.float64) -> F:
        return np.array([vx])

    @classmethod
    def exp(cls, vx: np.float64) -> np.float64:
        return vx

    @classmethod
    def log(cls, x: np.float64) -> np.float64:
        return x

    @classmethod
    def adjoint(cls, x: np.float64) -> F:
        return np.eye(1)

    @classmethod
    def left_jacobian(cls, v: F) -> F:
        return np.eye(1)

    @classmethod
    def left_jacobian_inverse(cls, v: F) -> F:
        return np.eye(1)

    @classmethod
    def lie_bracket(cls, v1: F, v2: F) -> F:
        return np.array([0])

    @classmethod
    def randu(cls) -> np.float64:
        return np.float64(np.random.uniform(0, 1))

    @classmethod
    def normalize(cls, x: np.float64) -> np.float64:
        return x


class RplusMul(LieGroup[np.float64, np.float64]):
    N: int = 1

    @classmethod
    def identity(cls) -> np.float64:
        return np.float64(1.0)

    @classmethod
    def compose(cls, x1: np.float64, x2: np.float64) -> np.float64:
        return x1 * x2

    @classmethod
    def inverse(cls, x: np.float64) -> np.float64:
        return 1 / x

    @classmethod
    def hat(cls, v: F) -> np.float64:
        return np.float64(v[0])

    @classmethod
    def vee(cls, vx: np.float64) -> F:
        return np.array([vx])

    @classmethod
    def exp(cls, vx: np.float64) -> np.float64:
        return np.float64(np.exp(vx))

    @classmethod
    def log(cls, x: np.float64) -> np.float64:
        return np.float64(np.log(x))

    @classmethod
    def adjoint(cls, x: np.float64) -> F:
        return np.eye(1)

    @classmethod
    def left_jacobian(cls, v: F) -> F:
        return np.array([[np.exp(v[0])]])

    @classmethod
    def left_jacobian_inverse(cls, v: F) -> F:
        return np.array([[np.exp(-v[0])]])

    @classmethod
    def lie_bracket(cls, v1: F, v2: F) -> F:
        return np.zeros(1)

    @classmethod
    def randu(cls) -> np.float64:
        return np.float64(np.random.uniform(0, 1))

    @classmethod
    def normalize(cls, x: np.float64) -> np.float64:
        return x
    


    
def RnAdd_factory(n: int) -> Type[LieGroup[F, F]]:
    assert n >= 1
    class RnAdd(LieGroup[F, F]):
        N: int = n
        
        @classmethod
        def identity(cls) -> F:
            return np.zeros(cls.N)

        @classmethod
        def compose(cls, x1: F, x2: F) -> F:
            return x1 + x2

        @classmethod
        def inverse(cls, x: F) -> F:
            return -x

        @classmethod
        def hat(cls, v: F) -> F:
            cls._check_hat_input(v)
            return v

        @classmethod
        def vee(cls, vx: F) -> F:
            v = vx
            cls._check_vee_output(v)
            return v

        @classmethod
        def exp(cls, vx: F) -> F:
            return vx

        @classmethod
        def log(cls, x: F) -> F:
            return x

        @classmethod
        def adjoint(cls, x: F) -> F:
            return np.eye(cls.N)

        @classmethod
        def left_jacobian(cls, v: F) -> F:
            return np.eye(cls.N)

        @classmethod
        def left_jacobian_inverse(cls, v: F) -> F:
            return np.eye(cls.N)

        @classmethod
        def lie_bracket(cls, v1: F, v2: F) -> F:
            return np.zeros(cls.N)

        @classmethod
        def randu(cls) -> F:
            return np.random.uniform(0, 1, size=(cls.N,))

        @classmethod
        def normalize(cls, x: F) -> F:
            return x
    RnAdd.__name__ = f"R{n}Add"
    return RnAdd
    
    
class UnitQuat(LieGroup[quaternion.quaternion, quaternion.quaternion]):
    
    N: int = 3

    @classmethod
    def identity(cls) -> quaternion.quaternion:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def compose(cls, q1: quaternion.quaternion, q2: quaternion.quaternion) -> quaternion.quaternion:
        return q1 * q2

    @classmethod
    def inverse(cls, q: quaternion.quaternion) -> quaternion.quaternion:
        return q.conj()

    @classmethod
    def hat(cls, v: F) -> quaternion.quaternion:
        cls._check_hat_input(v)
        return quaternion.quaternion(0.0, v[0], v[1], v[2])
        
    @classmethod
    def vee(cls, vx: quaternion.quaternion) -> F:
        v = np.array([vx.x, vx.y, vx.z])
        cls._check_vee_output(v)
        return v

    @classmethod
    def exp(cls, qx: quaternion.quaternion) -> quaternion.quaternion:
        v = np.array([qx.x, qx.y, qx.z])
        theta = np.linalg.norm(v)
        if theta < 1e-10:
            return cls.identity()
        axis = v / theta
        return quaternion.quaternion(np.cos(theta), *(np.sin(theta) * axis))

    @classmethod
    def log(cls, q: quaternion.quaternion) -> quaternion.quaternion:
        v = np.array([q.x, q.y, q.z])
        norm_v = np.linalg.norm(v)
        w = q.w
        if norm_v < 1e-10:
            return quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
        theta = np.arccos(np.clip(w, -1.0, 1.0))
        return quaternion.quaternion(0.0, *(theta * v / norm_v))

    @classmethod
    def adjoint(cls, X: quaternion.quaternion) -> F:
        return quaternion.as_rotation_matrix(X)
    
    @classmethod
    def left_jacobian(cls, v: F) -> F:
        theta = np.linalg.norm(v)
        vx = SO3.hat(v)
        if theta < EPS:
            return (np.eye(3) + 0.5 * vx + (1.0/6.0) * (vx @ vx)).astype(np.float64)
        return (np.eye(3) + ((1 - np.cos(theta)) / (theta ** 2)) * vx + ((theta - np.sin(theta)) / (theta ** 3)) * (vx @ vx)).astype(np.float64)

    @classmethod
    def left_jacobian_inverse(cls, v: F) -> F:
        theta = np.linalg.norm(v)
        vx = SO3.hat(v)
        if theta < EPS:
            return (np.eye(3) - 0.5 * vx + (1.0 / 12.0) * (vx @ vx)).astype(np.float64)
        return (np.eye(3) - 0.5 * vx + (1.0 / (theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * (vx @ vx)).astype(np.float64)

    @classmethod
    def lie_bracket(cls, v1: F, v2: F) -> F:
        return np.cross(v1, v2)
    
    @classmethod
    def randu(cls) -> quaternion.quaternion:
        u1, u2, u3 = np.random.uniform(0, 1, (3,))
        qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qx = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qy = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qz = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        return quaternion.quaternion(qw, qx, qy, qz)
    
    @classmethod
    def normalize(cls, q: quaternion.quaternion) -> quaternion.quaternion:
        return q / np.abs(q)



    
class U1(LieGroup[np.complex128, np.complex128]):
    N: int = 1

    @classmethod
    def identity(cls) -> np.complex128:
        return np.complex128(1.0 + 0.0j)

    @classmethod
    def compose(cls, z1: np.complex128, z2: np.complex128) -> np.complex128:
        return np.complex128(z1 * z2)

    @classmethod
    def inverse(cls, z: np.complex128) -> np.complex128:
        return np.complex128(np.conj(z))

    @classmethod
    def hat(cls, w: F) -> np.complex128:
        cls._check_hat_input(w)
        return np.complex128(1j * w[0])

    @classmethod
    def vee(cls, wx: np.complex128) -> F:
        v = np.array([np.imag(wx)])
        cls._check_vee_output(v)
        return v

    @classmethod
    def exp(cls, wx: np.complex128) -> np.complex128:
        return np.complex128(np.exp(wx))

    @classmethod
    def log(cls, z: np.complex128) -> np.complex128:
        return np.complex128(np.angle(z) * 1j)

    @classmethod
    def adjoint(cls, z: np.complex128) -> F:
        return np.eye(1)

    @classmethod
    def left_jacobian(cls, w: F) -> F:
        return np.eye(1)

    @classmethod
    def left_jacobian_inverse(cls, w: F) -> F:
        return np.eye(1)

    @classmethod
    def lie_bracket(cls, w1: F, w2: F) -> F:
        return np.zeros(1)
    
    @classmethod
    def randu(cls) -> np.complex128:
        angle = np.random.uniform(-np.pi, np.pi)
        return np.complex128(np.exp(1j * angle))
    @classmethod
    
    def normalize(cls, z: np.complex128) -> np.complex128:
        return np.complex128(z / np.abs(z))


class SE2(LieGroup[F, F]):
    N: int = 3
    
    @classmethod
    def identity(cls) -> F:
        return np.eye(3)

    @classmethod
    def compose(cls, T1: F, T2: F) -> F:
        return T1 @ T2

    @classmethod
    def inverse(cls, T: F) -> F:
        return np.vstack([
            np.hstack([T[:2, :2].T, -T[:2, :2].T @ T[:2, 2:3]]),
            np.array([[0, 0, 1]])
        ])

    @classmethod
    def hat(cls, V: F) -> F:
        cls._check_hat_input(V)
        w = V[:1]
        v = V[1:]
        return np.array([
            [   0, -w[0], v[0]],
            [w[0],     0, v[1]],
            [0,        0,    0]
        ])

    @classmethod
    def vee(cls, Vx: F) -> F:
        V = np.array([Vx[1, 0], Vx[0, 2], Vx[1, 2]])
        cls._check_vee_output(V)
        return V

    @classmethod
    def log(cls, T: F) -> F:
        R = T[:2, :2]
        p = T[:2, 2]
        theta = np.arctan2(R[1, 0], R[0, 0])
        if np.abs(theta) < 1e-10:
            V_inv = np.eye(2)
        else:
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / theta
            V_inv = (1 / (A ** 2 + B ** 2)) * np.array([[A, B], [-B, A]])
        v = V_inv @ p
        return cls.hat(np.array([theta, *v]))

    @classmethod
    def exp(cls, Vx: F) -> F:
        w, vx, vy = cls.vee(Vx)
        v = np.array([[vx, vy]]).T
        if np.abs(w) > EPS:
            theta = w
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / theta
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            G = np.array([
                [A, -B],
                [B, A]
            ])
            return np.vstack([
                np.hstack([R, G @ v]),
                np.array([0, 0, 1])
            ])
        else:
            return np.vstack([
                np.hstack([np.eye(2), v]),
                np.array([0, 0, 1])
            ])

    @classmethod
    def adjoint(cls, T: F) -> F:
        return np.vstack([
            np.array([[1, 0, 0]]),
            np.hstack([np.array([[T[1, 2], -T[0, 2]]]).T, T[:2, :2]])
        ])
        
    
    @classmethod
    def left_jacobian(cls, V: F) -> F:
        theta = V[0]
        if np.abs(theta) < EPS:
            return np.eye(3)
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta
        return np.array([
            [A, -B, 0],
            [B,  A, 0],
            [0,  0, 1]
        ])

    @classmethod
    def left_jacobian_inverse(cls, V: F) -> F:
        theta = V[0]
        if np.abs(theta) < EPS:
            return np.eye(3)
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta
        denom = A**2 + B**2
        V_inv = np.array([[A, B], [-B, A]]) / denom
        return np.array([
            [V_inv[0,0], V_inv[0,1], 0],
            [V_inv[1,0], V_inv[1,1], 0],
            [0,          0,          1]
        ])
        
    @classmethod
    def lie_bracket(cls, V1: F, V2: F) -> F:
        V1x = cls.hat(V1)
        V2x = cls.hat(V2)
        return cls.vee(V1x @ V2x - V2x @ V1x)
    
    @classmethod
    def normalize(cls, T: F) -> F:
        U, _, Vt = np.linalg.svd(T[:2, :2])
        R = U @ Vt
        T[:2, :2] = R
        return T
    
    @classmethod
    def randu(cls, bounds: tuple[tuple[np.float64, np.float64], tuple[np.float64, np.float64]]=((np.float64(0.0), np.float64(1.0)), (np.float64(0.0), np.float64(1.0)))) -> F:
        theta = np.random.uniform(-np.pi, np.pi)
        if (not isinstance(bounds, tuple) or len(bounds) != 2 or not all(isinstance(b, tuple) and len(b) == 2 for b in bounds)):
            raise TypeError(
                f"Bounds must be a tuple of two (min, max) tuples, e.g., ((xmin, xmax), (ymin, ymax)), got {bounds}"
            )
        (xmin, xmax), (ymin, ymax) = bounds
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        p = np.array([[x], [y]])

        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2:3] = p
        return T
    
class SE3(LieGroup[F, F]):
    N: int = 6
    
    @classmethod
    def identity(cls) -> F:
        return np.eye(4)

    @classmethod
    def compose(cls, T1: F, T2: F) -> F:
        return T1 @ T2

    @classmethod
    def inverse(cls, T: F) -> F:
        T_inv = np.vstack([
            np.hstack([T[:3, :3].T, -T[:3, :3].T @ T[:3, 3:4]]),
            np.array([[0, 0, 0, 1]])
        ])
        return T_inv

    @classmethod
    def hat(cls, V: F) -> F:
        cls._check_hat_input(V)
        w = V[:3]
        v = V[3:]
        wx = SO3.hat(w)
        Vx = np.vstack([
            np.hstack([wx, np.array([v]).T]),
            np.array([0, 0, 0, 0])
        ])
        return Vx

    @classmethod
    def vee(cls, Vx: F) -> F:
        V = np.array([Vx[2, 1], Vx[0, 2], Vx[1, 0], Vx[0, 3], Vx[1, 3], Vx[2, 3]])
        cls._check_vee_output(V)
        return V
    
    @classmethod
    def exp(cls, Vx: F) -> F:
        w = np.array([
            Vx[2, 1],
            Vx[0, 2],
            Vx[1, 0]
        ])
        v = Vx[:3, 3]

        theta = np.linalg.norm(w)
        if theta < EPS:
            R = np.eye(3)
            V = np.eye(3)
        else:
            wx = np.array([
                [    0, -w[2],  w[1]],
                [ w[2],     0, -w[0]],
                [-w[1],  w[0],     0]
            ])
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / (theta ** 2)
            C = (1 - A) / (theta ** 2)
            R = np.eye(3) + A * wx + B * (wx @ wx)
            V = np.eye(3) + B * wx + C * (wx @ wx)

        p = V @ v

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    @classmethod
    def log(cls, T: F) -> F:
        R = T[:3, :3]
        p = T[:3, 3]
        
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if np.abs(theta) < EPS:
            w = np.zeros(3)
            V_inv = np.eye(3)
        else:
            wx = (theta / (2 * np.sin(theta))) * (R - R.T)
            w = np.array([wx[2, 1], wx[0, 2], wx[1, 0]])
            w_norm_sq = np.dot(w, w)
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / (theta**2)
            V_inv = (
                np.eye(3)
                - 0.5 * wx
                + (1 - (A / (2 * B))) / w_norm_sq * (wx @ wx)
            )

        v = V_inv @ p

        V = np.zeros((4, 4))
        V[:3, :3] = SO3.hat(w)
        V[:3, 3] = v
        return V

    @classmethod
    def left_jacobian(cls, V: F) -> F:
        w = V[:3]
        v = V[3:]
        theta = np.linalg.norm(w)

        wx = SO3.hat(w)
        vx = SO3.hat(v)

        if theta < EPS:
            Jw_approx = np.eye(3) - 0.5 * wx
            JV = np.zeros((6, 6))
            JV[:3, :3] = Jw_approx
            JV[3:, :3] = 0.5 * wx
            JV[3:, 3:] = Jw_approx
            return JV

        B = (1 - np.cos(theta)) / (theta**2)
        C = (theta - np.sin(theta)) / (theta**3)

        Jw = np.eye(3) + B * wx + C * (wx @ wx)
        Qwv = (1 / 2) * vx + C * (wx @ vx + vx @ wx) + B * (wx @ vx - vx @ wx)

        JV = np.zeros((6, 6))
        JV[:3, :3] = Jw
        JV[3:, :3] = Qwv
        JV[3:, 3:] = Jw
        return JV

    @classmethod
    def left_jacobian_inverse(cls, V: F) -> F:
        w = V[:3]
        v = V[3:]
        theta = np.linalg.norm(w)
        
        wx = SO3.hat(w)
        vx = SO3.hat(v)

        if theta < EPS:
            Jw_inv = np.eye(3) + 0.5 * wx
            J_inv = np.eye(6)
            J_inv[:3, :3] = Jw_inv
            J_inv[3:, :3] = -vx / 2
            J_inv[3:, 3:] = Jw_inv
            return J_inv
        
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta ** 2)
        
        Jw_inv = np.eye(3) \
            - wx / 2 \
            + (1 / (theta ** 2)) * (1 - (A / (2 * B))) * (wx @ wx)
        Qwv = vx / 2 \
            + (theta - np.sin(theta)) / (theta ** 3) * (wx @ vx + vx @ wx) \
            + (1 - np.cos(theta)) / (theta ** 2) * (wx @ vx - vx @ wx)
        
        JV_inv = np.zeros((6, 6))
        JV_inv[:3, :3] = Jw_inv
        JV_inv[3:, :3] = -Jw_inv @ Qwv @ Jw_inv
        JV_inv[3:, 3:] = Jw_inv
        
        return JV_inv
        
    @classmethod
    def lie_bracket(cls, V1: F, V2: F) -> F:
        V1_w = V1[:3]
        V1_v = V1[3:]

        V2_w = V2[:3]
        V2_v = V2[3:]

        w_cross = np.cross(V1_w, V2_w)
        v_cross = np.cross(V1_w, V2_v) - np.cross(V2_w, V1_v)

        return np.hstack([w_cross, v_cross])

    @classmethod
    def normalize(cls, T: F) -> F:
        U, _, Vt = np.linalg.svd(T[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R *= -1
        T[:3, :3] = R
        return T

    @classmethod
    def randu(cls, bounds: tuple[tuple[np.float64, np.float64], tuple[np.float64, np.float64], tuple[np.float64, np.float64]]=((np.float64(0.0), np.float64(1.0)), (np.float64(0.0), np.float64(1.0)), (np.float64(0.0), np.float64(1.0)))) -> F:
        theta = np.random.uniform(-np.pi, np.pi)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        w = axis * theta
        if (not isinstance(bounds, tuple) or 
            len(bounds) != 3 or 
            not all(isinstance(b, tuple) and len(b) == 2 for b in bounds)):
            raise TypeError(
                f"bounds must be a tuple of three (min, max) tuples, e.g., ((xmin, xmax), (ymin, ymax), (zmin, zmax)), got {bounds}"
            )
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        z = np.random.uniform(zmin, zmax)

        wx = SO3.hat(w)
        if theta < EPS:
            R = np.eye(3)
        else:
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / (theta**2)
            R = np.eye(3) + A * wx + B * (wx @ wx)

        p = np.array([x, y, z])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T
    
    @classmethod
    def adjoint(cls, T: F) -> F:
        R = T[:3, :3]
        p = T[:3, 3]

        px = SO3.hat(p)

        Adj = np.zeros((6, 6))
        Adj[:3, :3] = R
        Adj[3:, :3] = px @ R
        Adj[3:, 3:] = R
        return Adj



class SO2(LieGroup[F, F]):
    N: int = 1

    @classmethod
    def identity(cls) -> F:
        return np.eye(2)

    @classmethod
    def compose(cls, R1: F, R2: F) -> F:
        return R1 @ R2

    @classmethod
    def inverse(cls, R: F) -> F:
        return R.T

    @classmethod
    def hat(cls, w: F) -> F:
        cls._check_hat_input(w)
        return np.array([
            [   0, -w[0]],
            [w[0],     0],
        ])

    @classmethod
    def vee(cls, wx: F) -> F:
        w = np.array([wx[1, 0]])
        cls._check_vee_output(w)
        return w

    @classmethod
    def log(cls, R: F) -> F:
        theta = np.arctan2(R[1, 0], R[0, 0])
        return cls.hat(np.array([theta]))

    @classmethod
    def exp(cls, wx: F) -> F:
        w = cls.vee(wx)
        theta = w[0]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return R

    @classmethod
    def adjoint(cls, R: F) -> F:
        return np.eye(1)
    
    @classmethod
    def left_jacobian(cls, w: F) -> F:
        theta = w[0]
        if np.abs(theta) < EPS:
            return np.eye(1)
        return np.array([[np.sin(theta) / theta]])

    @classmethod
    def left_jacobian_inverse(cls, w: F) -> F:
        theta = w[0]
        if np.abs(theta) < EPS:
            return np.eye(1)
        return np.array([[theta / np.sin(theta)]])
    
    @classmethod
    def lie_bracket(cls, w1: F, w2: F) -> F:
        return np.zeros(1)
    
    @classmethod
    def randu(cls) -> F:
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])



class SO3(LieGroup[F, F]):
    N: int = 3
    
    @classmethod
    def identity(cls) -> F:
        return np.eye(3)

    @classmethod
    def compose(cls, R1: F, R2: F) -> F:
        return R1 @ R2

    @classmethod
    def inverse(cls, R: F) -> F:
        return R.T

    @classmethod
    def hat(cls, w: F) -> F:
        cls._check_hat_input(w)
        return np.array([
            [    0, -w[2],  w[1]],
            [ w[2],     0, -w[0]],
            [-w[1],  w[0],     0]
        ])

    @classmethod
    def vee(cls, wx: F) -> F:
        w = np.array([wx[2, 1], wx[0, 2], wx[1, 0]])
        cls._check_vee_output(w)
        return w

    @classmethod
    def log(cls, R: F) -> F:
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if np.abs(theta) < 1e-10:
            return np.zeros((3, 3))
        wx = (theta / (2 * np.sin(theta))) * (R - R.T)
        return wx

    @classmethod
    def exp(cls, wx: F) -> F:
        w = cls.vee(wx)
        theta = np.linalg.norm(w)
        if theta < 1e-10:
            return np.eye(3).astype(np.float64)
        else:
            wn = w / theta
            wxn = cls.hat(wn)
            return (np.eye(3) + np.sin(theta) * wxn + (1 - np.cos(theta)) * (wxn @ wxn)).astype(np.float64)

    @classmethod
    def adjoint(cls, R: F) -> F:
        return R
    
    @classmethod
    def left_jacobian(cls, w: F) -> F:
        theta = np.linalg.norm(w)
        wx = cls.hat(w)
        if theta < EPS:
            return np.eye(3) + 0.5 * wx + (1.0 / 6.0) * (wx @ wx)
        return (np.eye(3) + ((1 - np.cos(theta)) / (theta ** 2)) * wx + ((theta - np.sin(theta)) / (theta ** 3)) * (wx @ wx)).astype(np.float64)

    @classmethod
    def left_jacobian_inverse(cls, w: F) -> F:
        theta = np.linalg.norm(w)
        wx = cls.hat(w)
        if theta < EPS:
            return np.eye(3) - 0.5 * wx + (1.0 / 12.0) * (wx @ wx)
        return (np.eye(3) - 0.5 * wx + (1.0 / (theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * (wx @ wx)).astype(np.float64)
    
    @classmethod
    def lie_bracket(cls, w1: F, w2: F) -> F:
        w1x = cls.hat(w1)
        w2x = cls.hat(w2)
        return cls.vee(w1x @ w2x - w2x @ w1x)
    
    
    @classmethod
    def randu(cls) -> F:
        u1, u2, u3 = np.random.uniform(0, 1, (3,))
        qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qx = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qy = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qz = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        q = quaternion.quaternion(qw, qx, qy, qz)
        return quaternion.as_rotation_matrix(q)
        
    @classmethod
    def normalize(cls, R: F) -> F:
        U, _, Vt = np.linalg.svd(R)
        return (U @ Vt).astype(np.float64)