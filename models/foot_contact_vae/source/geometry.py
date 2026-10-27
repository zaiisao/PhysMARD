"""Foot support-point geometry from the SMPL body mesh.

The local frame starts at the ankle. Its ankle-to-toe direction is unit(toe - ankle); its upward-perpendicular
direction is world up projected perpendicular to ankle-to-toe. Coordinates are in ankle-to-toe foot lengths.
"""
import json
from pathlib import Path

import numpy as np
import torch

GEOMETRY_CACHE = Path(__file__).resolve().parent / 'support_geometry.json'
FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}
SIDE_NAMES = tuple(FOOT_JOINTS)
POINTS = ('heel', 'pad', 'tip')
SMPL_BODY_POSE_DIM = 69

# ----------------------------------------------------------------------------------------
# Support-point geometry from the SMPL body mesh
# ----------------------------------------------------------------------------------------
def _points_in_foot_frame(vertices, joints, weights, ankle, toe):
    """Heel, pad and tip in ankle-to-toe and upward-perpendicular coordinates, in foot lengths.

    The upward-perpendicular direction is world up made orthogonal to the ankle-to-toe direction; negative means below
    the ankle-to-toe line.
    heel: lowest foot vertex behind the ankle. pad: lowest foot vertex within 0.25 foot lengths of the
    toe joint (SMPL's toe joint sits at the ball of the foot). tip: lowest vertex within 0.1 foot
    lengths of the most forward vertex, i.e. where a toe touches the ground.
    """
    ankle_position = joints[ankle]
    toe_position = joints[toe]
    foot_length = np.linalg.norm(toe_position - ankle_position)

    ankle_to_toe_direction = (toe_position - ankle_position) / foot_length
    world_up = np.array([0., 1., 0.])
    world_up_projection_onto_foot = (
        np.dot(world_up, ankle_to_toe_direction) * ankle_to_toe_direction
    )
    world_up_perpendicular_to_foot = world_up - world_up_projection_onto_foot
    upward_perpendicular_length = np.linalg.norm(world_up_perpendicular_to_foot)
    upward_perpendicular_direction = world_up_perpendicular_to_foot / upward_perpendicular_length

    # Approximate the foot region using vertices influenced by the ankle and toe joints
    ankle_toe_vertex_mask = (weights[:, ankle] + weights[:, toe]) > 0.5
    foot_vertex_offsets = vertices[ankle_toe_vertex_mask] - ankle_position
    ankle_to_toe_coordinate = foot_vertex_offsets @ ankle_to_toe_direction / foot_length
    upward_perpendicular_coordinate = foot_vertex_offsets @ upward_perpendicular_direction / foot_length

    heel_vertex = np.argmin(np.where(
        ankle_to_toe_coordinate < 0, upward_perpendicular_coordinate, np.inf
    ))
    pad_vertex = np.argmin(np.where(
        np.abs(ankle_to_toe_coordinate - 1.0) < 0.25, upward_perpendicular_coordinate, np.inf
    ))
    tip_vertex = np.argmin(np.where(
        ankle_to_toe_coordinate > ankle_to_toe_coordinate.max() - 0.1,
        upward_perpendicular_coordinate,
        np.inf,
    ))

    points = {
        'heel': {
            'lengthwise': float(ankle_to_toe_coordinate[heel_vertex]),
            'upward': float(upward_perpendicular_coordinate[heel_vertex]),
        },
        'pad': {
            'lengthwise': float(ankle_to_toe_coordinate[pad_vertex]),
            'upward': float(upward_perpendicular_coordinate[pad_vertex]),
        },
        'tip': {
            'lengthwise': float(ankle_to_toe_coordinate[tip_vertex]),
            'upward': float(upward_perpendicular_coordinate[tip_vertex]),
        },
    }

    return {
        'points': points,
        'foot_length': float(foot_length),
    }


def derive_support_geometry(model_dir, num_shapes=1000, seed=0, cache=GEOMETRY_CACHE):
    """Per foot side: SMPL-neutral points and mean / covariance of six coordinates over random body shapes.

    Shapes are betas ~ N(0, I) for the SMPL model in its rest pose.
    Coordinates are ordered heel, pad, tip, each as (ankle-to-toe, upward-perpendicular).
    The result is cached to JSON so the VAE does not need SMPL at construction time.
    """
    import smplx
    body = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl').eval()
    weights = body.lbs_weights.detach().numpy()

    neutral_betas = np.zeros((1, body.num_betas))
    shape_betas = np.random.default_rng(seed).standard_normal((num_shapes, body.num_betas))
    betas = np.concatenate((neutral_betas, shape_betas))
    with torch.no_grad():
        betas = torch.tensor(betas, dtype=torch.float32)
        out = body(
            betas=betas,
            global_orient=torch.zeros(len(betas), 3),
            body_pose=torch.zeros(len(betas), SMPL_BODY_POSE_DIM),
        )

    per_shape = []
    for vertices, joints in zip(out.vertices.numpy(), out.joints.numpy()):
        foot_points = {}
        for side, (ankle, toe) in FOOT_JOINTS.items():
            foot_points[side] = _points_in_foot_frame(vertices, joints, weights, ankle, toe)
        per_shape.append(foot_points)

    geometry = {}
    for side in FOOT_JOINTS:
        shape_points = []
        for shape in per_shape[1:]:
            foot_points = shape[side]['points']
            point_coordinates = []
            for point in POINTS:
                coordinates = foot_points[point]
                point_coordinates.append((coordinates['lengthwise'], coordinates['upward']))
            shape_points.append(point_coordinates)

        values = np.array(shape_points).reshape(num_shapes, 6)
        neutral_shape = per_shape[0][side]
        neutral_points = neutral_shape['points']
        geometry[side] = {
            "neutral": [
                [neutral_points[point]['lengthwise'], neutral_points[point]['upward']]
                for point in POINTS
            ],
            "mean": values.mean(0).tolist(),
            "covariance": np.cov(values, rowvar=False).tolist(),
            "length_m": neutral_shape['foot_length'],
        }

    if cache is not None:
        metadata = {
            "source": str(model_dir),
            "num_shapes": num_shapes,
            "seed": seed,
            "points": POINTS,
            "geometry": geometry,
        }
        metadata_json = json.dumps(metadata, indent=2)
        Path(cache).write_text(metadata_json)

    return geometry


def load_support_geometry(cache=GEOMETRY_CACHE):
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"Support geometry cache not found: {cache}. "
            "Generate it with derive_support_geometry(model_dir=...) and a local SMPL model directory."
        )
    return json.loads(cache.read_text())['geometry']
