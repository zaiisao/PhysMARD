"""Foot support-point geometry from the SMPL body mesh.

Foot frame: origin at the ankle joint, `along` = unit(toe joint - ankle), `normal` = world up made orthogonal to
`along` (negative = below the ankle->toe line). Units: foot lengths (ankle->toe-joint distance).
"""
import json
from pathlib import Path

import numpy as np
import torch

GEOMETRY_CACHE = Path(__file__).resolve().parent / 'support_geometry.json'
FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}
SIDE_NAMES = tuple(FOOT_JOINTS)
POINTS = ('heel', 'pad', 'tip')

# ----------------------------------------------------------------------------------------
# Support-point geometry from the SMPL body mesh
# ----------------------------------------------------------------------------------------
def _points_in_foot_frame(vertices, joints, weights, ankle, toe):
    """Heel, pad and tip of one foot in its ankle->toe frame, in foot lengths.

    along  = unit(toe joint - ankle); normal = world up made orthogonal to `along` (negative = below).
    heel: lowest foot vertex behind the ankle. pad: lowest foot vertex within 0.25 foot lengths of the
    toe joint (SMPL's toe joint sits at the ball of the foot). tip: lowest vertex within 0.1 foot
    lengths of the most forward vertex, i.e. where a toe touches the ground.
    """
    a, t = joints[ankle], joints[toe]
    length = np.linalg.norm(t - a)
    along = (t - a) / length
    normal = np.array([0., 1., 0.]) - along * along[1]
    normal /= np.linalg.norm(normal)
    on_foot = (weights[:, ankle] + weights[:, toe]) > 0.5
    rel = vertices[on_foot] - a
    x, y = rel @ along / length, rel @ normal / length

    def lowest(mask):
        i = np.argmin(np.where(mask, y, np.inf))
        return [float(x[i]), float(y[i])]

    return dict(heel=lowest(x < 0), pad=lowest(np.abs(x - 1.0) < 0.25), tip=lowest(x > x.max() - 0.1),
                length_m=float(length))


def derive_support_geometry(model_dir, num_shapes=1000, seed=0, cache=GEOMETRY_CACHE):
    """Per foot side: SMPL-neutral points, and mean / covariance of the 6 coordinates over random body shapes.

    Shapes are betas ~ N(0, I) for the 10-coefficient neutral SMPL model in its rest pose.
    The result is cached to JSON so the VAE does not need SMPL at construction time.
    """
    import smplx
    body = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl', num_betas=10).eval()
    weights = body.lbs_weights.detach().numpy()
    betas = np.concatenate([np.zeros((1, 10)), np.random.default_rng(seed).standard_normal((num_shapes, 10))])
    per_shape = []
    with torch.no_grad():
        for i in range(0, len(betas), 100):
            b = torch.tensor(betas[i:i + 100], dtype=torch.float32)
            out = body(betas=b, global_orient=torch.zeros(len(b), 3), body_pose=torch.zeros(len(b), 69))
            for v, j in zip(out.vertices.numpy(), out.joints.numpy()):
                per_shape.append({side: _points_in_foot_frame(v, j, weights, *FOOT_JOINTS[side]) for side in FOOT_JOINTS})
    geometry = {}
    for side in FOOT_JOINTS:
        values = np.array([[shape[side][p] for p in POINTS] for shape in per_shape[1:]]).reshape(num_shapes, 6)
        geometry[side] = dict(neutral=[per_shape[0][side][p] for p in POINTS],
                              mean=values.mean(0).tolist(),           # prior mean of (heel, pad, tip) x (along, normal)
                              covariance=np.cov(values, rowvar=False).tolist(),
                              length_m=per_shape[0][side]['length_m'])
    if cache is not None:
        Path(cache).write_text(json.dumps(dict(source=str(model_dir), num_shapes=num_shapes, seed=seed,
                                               points=POINTS, geometry=geometry), indent=2))
    return geometry


def load_support_geometry(cache=GEOMETRY_CACHE):
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"Support geometry cache not found: {cache}. "
            "Generate it with derive_support_geometry(model_dir=...) and a local SMPL model directory."
        )
    return json.loads(cache.read_text())['geometry']
