import json
from pathlib import Path

import torch


VELOCITY_KEYS = ("v1_n", "v2_n", "v1_nm1", "v2_nm1")
AUXILIARY_KEYS = ("r1_n", "r2_n", "r1_nm1", "r2_nm1", "r1_next", "r2_next")


def _rms_scale(tensors, eps=1e-12):
    """
    Scalar RMS over all components in a list of vector tensors.
    """
    sq_sum = None
    count = 0
    for tensor in tensors:
        value = tensor.detach().float().pow(2).sum()
        sq_sum = value if sq_sum is None else sq_sum + value
        count += tensor.numel()
    return torch.sqrt(sq_sum / max(count, 1)).clamp_min(eps).item()


class E3VectorNormalizer:
    """
    Equivariance-preserving normalization for structured dimer E3 tensors.

    This does not subtract means and does not scale x/y/z separately. It divides
    each vector family by one scalar RMS factor:
        velocities: v1_n, v2_n, v1_nm1, v2_nm1
        auxiliary:  r1_n, r2_n, r1_nm1, r2_nm1, r1_next, r2_next

    q1/q2 are left in physical units so edge vectors and radial embeddings keep
    the physical dimer separation scale.
    """
    def __init__(self, velocity_scale=1.0, auxiliary_scale=1.0):
        self.velocity_scale = float(velocity_scale)
        self.auxiliary_scale = float(auxiliary_scale)

    @classmethod
    def fit(cls, structured, eps=1e-12):
        velocity_scale = _rms_scale([structured[key] for key in VELOCITY_KEYS], eps=eps)
        auxiliary_scale = _rms_scale([structured[key] for key in AUXILIARY_KEYS], eps=eps)
        return cls(velocity_scale=velocity_scale, auxiliary_scale=auxiliary_scale)

    def transform(self, structured):
        out = {}
        for key, value in structured.items():
            if key in VELOCITY_KEYS:
                out[key] = value / self.velocity_scale
            elif key in AUXILIARY_KEYS:
                out[key] = value / self.auxiliary_scale
            else:
                out[key] = value
        return out

    def inverse_transform_aux(self, value):
        """
        Convert normalized auxiliary vectors back to physical units.
        """
        return value * self.auxiliary_scale

    def state_dict(self):
        return {
            "type": "E3VectorNormalizer",
            "velocity_scale": self.velocity_scale,
            "auxiliary_scale": self.auxiliary_scale,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            velocity_scale=state["velocity_scale"],
            auxiliary_scale=state["auxiliary_scale"],
        )

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.state_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path):
        with Path(path).open("r") as f:
            return cls.from_state_dict(json.load(f))
