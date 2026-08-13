"""
Fluctuation-dissipation-corrected rollout sampler for E3DimerCVAE ("Fix-A").

The trained E3 decoder reproduces the correct per-step auxiliary-noise magnitude
but only ~72-75% of the benchmark noise-velocity anticorrelation -- the
memory-friction term that dissipates relative and centre-of-mass kinetic energy.
The reduced dynamics therefore run hot (velocity marginal over-dispersed).

This sampler wraps the base ``E3DimerRolloutSampler`` and adds the missing
friction back as an equivariant, velocity-dependent correction to the sampled
noise:

    e       = bond unit vector (from q1, q2)
    dv      = v1 - v2                        (relative velocity)
    dvx     = dv . e                         (axial relative velocity)
    dv_perp = dv - dvx e                     (perpendicular relative velocity)
    sv      = v1 + v2                        (centre-of-mass velocity, x2)

    d_rel   = g * ( gamma_par * dvx * e + gamma_perp * dv_perp )
    d_com   = g * ( gamma_com * sv )
    r1     += 0.5 * (d_rel + d_com)
    r2     += 0.5 * (d_com - d_rel)

so that Cov(r1 - r2, dv) and Cov(r1 + r2, sv) are pushed toward the benchmark.
The three ``gamma_*`` gains are the per-channel friction deficits calibrated by
``calibrate_gains.py``. The scalar gain ``g`` (default 1.0) sweeps 0 (baseline)
-> 1 (full calibrated correction); it is selected on the velocity marginal by
``gain_sweep.py`` and is orthogonal to the calibration.
"""
import json

import numpy as np

from deepRD.noiseSampler.e3cvae.diagnostics import E3DimerRolloutSampler


def _minimal_image(rel, boxsize):
    return rel - boxsize * np.round(rel / boxsize)


class E3DimerRolloutSamplerFD(E3DimerRolloutSampler):
    """Base E3 rollout sampler + additive fluctuation-dissipation correction.

    Parameters
    ----------
    model, normalizer, boxsize, device, Tr, Tz
        Passed straight through to ``E3DimerRolloutSampler``.
    gains_path : str or Path, optional
        JSON file written by ``calibrate_gains.py`` holding ``gamma_par``,
        ``gamma_perp``, ``gamma_com``. Ignored if ``gains`` is given.
    gains : dict, optional
        In-memory gains dict (same keys); takes precedence over ``gains_path``.
    gain : float
        Scalar amplitude of the correction (0 = baseline, 1 = full calibration).
    """

    def __init__(self, model, normalizer, boxsize, device, Tr=1.0, Tz=1.0,
                 gains_path=None, gain=1.0, gains=None):
        super().__init__(model, normalizer, boxsize, device, Tr=Tr, Tz=Tz)
        if gains is None:
            if gains_path is None:
                raise ValueError("provide either `gains` or `gains_path`")
            with open(gains_path, "r") as f:
                gains = json.load(f)
        self.gamma_par = float(gains["gamma_par"])
        self.gamma_perp = float(gains["gamma_perp"])
        self.gamma_com = float(gains["gamma_com"])
        self.gain = float(gain)

    def sample(self, conditioned_vars):
        r = super().sample(conditioned_vars)  # physical 6D [r1(3), r2(3)]
        if self.gain == 0.0:
            return r
        c = np.asarray(conditioned_vars, dtype=np.float64).reshape(30)
        q1, q2 = c[0:3], c[3:6]
        v1, v2 = c[6:9], c[9:12]
        rel = _minimal_image(q2 - q1, self.boxsize)
        e = rel / max(np.linalg.norm(rel), 1e-12)

        dv = v1 - v2
        dvx = float(dv @ e)
        dv_perp = dv - dvx * e
        sv = v1 + v2

        d_rel = self.gain * (self.gamma_par * dvx * e + self.gamma_perp * dv_perp)
        d_com = self.gain * (self.gamma_com * sv)
        r = r.copy()
        r[0:3] += 0.5 * (d_rel + d_com)
        r[3:6] += 0.5 * (d_com - d_rel)
        return r
