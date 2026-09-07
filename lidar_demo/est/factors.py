"""The factor that lets the graph see the sensor mounting angle.

Registration compares two clouds in *sensor* coordinates and returns the
relative sensor pose between them.  The graph, meanwhile, estimates *body*
poses.  Converting between the two needs the extrinsic, and the extrinsic is
exactly the unknown.  So instead of converting the measurement once with a
number that is wrong, the conversion is written into the residual and the
extrinsic is left as a variable::

    T_pred(X_i, X_j, E) = E^-1 X_i^-1 X_j E
    r = Log( T_meas^-1 T_pred )

with ``E`` a single ``Pose3`` shared by every factor in the whole flight, not
one per pose.  That sharing is the entire mechanism.  A per-pose extrinsic could
absorb any measurement and would mean nothing; one global extrinsic has to
satisfy every pair at once.

Why the graph can see it and a filter cannot: the extrinsic is simply not in a
filter's state vector, so no measurement it processes has any path to it.  Here
it is a variable like any other and the scan-match residuals have a nonzero
derivative with respect to it.

What makes it observable, precisely
-----------------------------------

These factors have an exact gauge freedom.  Rotate every body pose on the right
by a constant ``A = (R_A, 0)`` and counter-rotate the extrinsic,
``X_k -> X_k A`` and ``E -> A^-1 E``, and every predicted measurement is
unchanged, because ``(A^-1 E)^-1 (A^-1 D A) (A^-1 E) = E^-1 D E``.  Along that
direction the scan-match factors say nothing at all.

The important thing about that gauge is that it does **not** move the aircraft:
``X_k A`` has exactly the same translation as ``X_k``.  So position measurements
cannot break it, no matter how many there are or how precise they are.  GNSS
alone leaves the mounting angle exactly as uncertain as its prior, and the tests
assert that.

What breaks it is absolute *attitude*, and that is what the IMU supplies: the
accelerometer sees the direction of gravity, which fixes roll and pitch outright,
and yaw follows from comparing the integrated heading against the direction the
aircraft is seen to travel.  With attitude anchored, the mounting angle is
observable.

Flying adjacent passes in opposite directions is then a conditioning matter
rather than an on/off one.  In the toy survey the tests build, the reversal
roughly halves the recovered mounting angle's uncertainty.  Its larger role is
in the picture: the same mounting error tilts an outbound swath one way and a
return swath the other, so neighbouring strips disagree and the terrain comes
out corrugated.  Flown all one way the map would be just as wrong and would look
smooth, which is a far harder thing to show anyone.

Jacobians are the chain rule over GTSAM's own derivatives.  Each step of
``between``, ``compose``, ``inverse`` and ``Logmap`` will fill a ``(6, 6)``
Fortran-ordered array, and multiplying those out is both exact and cheap.  The
closed forms are

    dr/dX_i = -J_l(r) Ad(E^-1 D^-1)
    dr/dX_j =  J_l(r) Ad(E^-1)
    dr/dE   =  J_l(r) (I - Ad(B^-1))

with ``D = X_i^-1 X_j`` and ``B = E^-1 D E``.  Those are checked against the
chain-rule result and against central differences in the tests, because a sign
error here would not crash anything: it would just quietly return the nominal
mounting angle and the demo would have nothing to show.
"""

from __future__ import annotations

import numpy as np

try:
    import gtsam
    from gtsam import Pose3
except ImportError:  # pragma: no cover - the estimator half needs WSL
    gtsam = None
    Pose3 = None


def _J(n: int = 6) -> np.ndarray:
    """A Jacobian slot GTSAM will write into: (n, n), Fortran order, zeroed."""
    return np.zeros((n, n), order="F")


def scan_match_error(measured, this, values, H):
    """Residual of one E-dependent relative-sensor-pose measurement.

    ``this.keys()`` is ``[X(i), X(j), E(0)]``.  Filling ``H`` is optional: GTSAM
    passes ``None`` when it only wants the error.
    """
    ki, kj, ke = this.keys()
    Ti = values.atPose3(ki)
    Tj = values.atPose3(kj)
    Ee = values.atPose3(ke)

    if H is None:
        pred = Ee.inverse().compose(Ti.between(Tj)).compose(Ee)
        return Pose3.Logmap(measured.between(pred))

    Hb1, Hb2 = _J(), _J()
    D = Ti.between(Tj, Hb1, Hb2)                 # D = Ti^-1 Tj

    HEi = _J()
    Einv = Ee.inverse(HEi)

    Hc1, Hc2 = _J(), _J()
    A = Einv.compose(D, Hc1, Hc2)                # A = E^-1 D

    Hd1, Hd2 = _J(), _J()
    B = A.compose(Ee, Hd1, Hd2)                  # B = E^-1 D E

    He1, He2 = _J(), _J()
    C = measured.between(B, He1, He2)            # C = T_meas^-1 B

    Hl = _J()
    r = Pose3.Logmap(C, Hl)

    HB = Hl @ He2
    H[0] = HB @ Hd1 @ Hc2 @ Hb1
    H[1] = HB @ Hd1 @ Hc2 @ Hb2
    H[2] = HB @ (Hd1 @ Hc1 @ HEi + Hd2)
    return r


def scan_match_factor(key_i, key_j, key_e, measured, noise):
    """A ``CustomFactor`` over ``[X(i), X(j), E]`` for one registration result."""
    if gtsam is None:
        raise ImportError("gtsam is required for the factor graph; run under WSL")

    def err(this, values, H):
        return scan_match_error(measured, this, values, H)

    return gtsam.CustomFactor(noise, [key_i, key_j, key_e], err)


# ---------------------------------------------------------------------------
# closed forms and numeric differentiation, used by the tests
# ---------------------------------------------------------------------------


def scan_match_jacobians_closed_form(measured, Ti, Tj, Ee):
    """The adjoint expressions for the three Jacobian blocks.

    Derivation, perturbing on the right as GTSAM does.  With ``D = X_i^-1 X_j``
    and ``B = E^-1 D E``:

    * ``X_i -> X_i exp(d)`` makes ``D -> exp(-d) D``, so ``B -> exp(-Ad(E^-1) d) B``
      and ``T_meas^-1 B -> C exp(-Ad(E^-1 D^-1) d)``.
    * ``X_j -> X_j exp(d)`` makes ``D -> D exp(d)``, so ``C -> C exp(Ad(E^-1) d)``.
    * ``E -> E exp(d)`` makes ``B -> exp(-d) B exp(d)``, so
      ``C -> C exp((I - Ad(B^-1)) d)``.

    Kept alongside the chain-rule version because two independent derivations
    agreeing is what makes a sign error in either one visible.
    """
    D = Ti.between(Tj)
    B = Ee.inverse().compose(D).compose(Ee)
    r = Pose3.Logmap(measured.between(B))
    Jlog = Pose3.LogmapDerivative(r)
    I6 = np.eye(6)
    Hi = -Jlog @ Ee.inverse().compose(D.inverse()).AdjointMap()
    Hj = Jlog @ Ee.inverse().AdjointMap()
    HE = Jlog @ (I6 - B.inverse().AdjointMap())
    return Hi, Hj, HE


def numeric_jacobians(measured, Ti, Tj, Ee, eps: float = 1e-6):
    """Central differences on GTSAM's own retraction, for the Jacobian tests."""

    def residual(a, b, e):
        pred = e.inverse().compose(a.between(b)).compose(e)
        return Pose3.Logmap(measured.between(pred))

    out = []
    for slot in range(3):
        Hn = np.zeros((6, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            args_p = [Ti, Tj, Ee]
            args_m = [Ti, Tj, Ee]
            args_p[slot] = args_p[slot].retract(d)
            args_m[slot] = args_m[slot].retract(-d)
            Hn[:, k] = (residual(*args_p) - residual(*args_m)) / (2 * eps)
        out.append(Hn)
    return out
