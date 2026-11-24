import json
from pathlib import Path

import numpy as np


PROFILE = 'task_delta_6d'
HORIZON = 30
BODY_JOINTS = ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1']
BASE_LINK = 'CHEST_JOINT0_Link'
TIP_LINKS = ['RARM_JOINT5_Link', 'LARM_JOINT5_Link']
POSE_COMPONENTS = ['x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21']
FEATURE_NAMES = [
    f'{tip}.{component}' for tip in TIP_LINKS for component in POSE_COMPONENTS
] + BODY_JOINTS


def is_task_dataset(features, metadata=None):
    names = [features.get(key, {}).get('names') for key in ('observation.state', 'action')]
    if metadata is not None and (metadata != profile_metadata() or FEATURE_NAMES not in names):
        raise ValueError('Unsupported dataset action profile or feature layout')
    if FEATURE_NAMES not in names:
        return False
    if any(value != FEATURE_NAMES for value in names):
        raise ValueError('Task space state and action features must match')
    return True


def checkpoint_profile(directory):
    path = Path(directory) / 'action_profile.json'
    if not path.exists():
        return 'joint'
    if json.loads(path.read_text()) != profile_metadata():
        raise ValueError(f'Unsupported action profile in {path}')
    return PROFILE


def profile_metadata():
    return {'profile': PROFILE, 'horizon': HORIZON, 'base_link': BASE_LINK,
            'features': FEATURE_NAMES, 'rotation': 'first_two_columns',
            'stored_actions': 'absolute', 'delta': 'inverse_anchor_times_target'}


def rotation_matrix(six):
    first, second = np.split(np.asarray(six, dtype=np.float64), 2, axis=-1)
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm) & (norm > 1e-6)):
        raise ValueError('Invalid first 6D rotation axis')
    first = first / norm
    orthogonal = second - (first * second).sum(axis=-1, keepdims=True) * first
    norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm) & (norm > 1e-6 * np.maximum(
            1, np.linalg.norm(second, axis=-1, keepdims=True)))):
        raise ValueError('Invalid second 6D rotation axis')
    second = orthogonal / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def pose_vector(position, rotation):
    return np.concatenate((position, rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def transform_actions(anchor, actions, *, to_absolute=False):
    """Apply T_anchor^-1 T_target or reconstruct T_anchor T_delta for each wrist."""
    anchor = np.asarray(anchor, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    for values in (actions, anchor):
        if values.shape[-1:] != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError('Expected finite NEXTAGE task space features')
    result = []
    for offset in (0, 9):
        pos = slice(offset, offset + 3)
        rot = slice(offset + 3, offset + 9)
        basis = rotation_matrix(anchor[..., rot])
        target = rotation_matrix(actions[..., rot])
        if to_absolute:
            translation = anchor[..., pos] + (basis @ actions[..., pos, None])[..., 0]
        else:
            basis = basis.swapaxes(-1, -2)
            translation = (basis @ (actions[..., pos] - anchor[..., pos])[..., None])[..., 0]
        result.append(pose_vector(translation, basis @ target))
    body = (np.add if to_absolute else np.subtract)(actions[..., 18:], anchor[..., 18:])
    return np.concatenate((*result, body), axis=-1).astype(np.float32)
