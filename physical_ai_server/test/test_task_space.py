import json
from types import SimpleNamespace

from datasets import Dataset
from lerobot.policies.act.configuration_act import ACTConfig
import numpy as np
import pytest
import torch

from physical_ai_server.data_processing.task_space import (
    FEATURE_NAMES, HORIZON, PROFILE, checkpoint_profile, is_task_dataset, pose_vector,
    profile_metadata, rotation_matrix, transform_actions,
)
from physical_ai_server.inference.inference_manager import InferenceManager
from physical_ai_server.training.task_space import prepare_task_dataset, relative_training_actions


def frame(x, rotation):
    right = pose_vector([x, 1., 2.], rotation)
    left = pose_vector([-x, 2., 1.], rotation.T)
    return np.r_[right, left, [x, 0., -x]].astype(np.float32)


def test_chunk_in_wrist_frame():
    rz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    start, goal = frame(1, rz), frame(2, rz @ rx)
    states = torch.tensor(np.stack([[goal, start]]))
    actions = torch.tensor(np.stack([[start, goal]]))
    labels = relative_training_actions({'observation.state': states, 'action': actions})[0].numpy()
    np.testing.assert_array_equal(labels[0], np.r_[
        [0, 0, 0, 1, 0, 0, 0, 1, 0] * 2, [0, 0, 0]])
    np.testing.assert_array_equal(labels[1, :3], [0, -1, 0])
    np.testing.assert_array_equal(rotation_matrix(labels[1, 3:9]), rx)
    np.testing.assert_allclose(transform_actions(start, labels, to_absolute=True), actions[0])


@pytest.mark.parametrize('six', [[0]*6, [1, 0, 0, 2, 0, 0], [1, 0, 0, 2, 1e-12, 0]])
def test_reject_degenerate_rotation(six):
    with pytest.raises(ValueError):
        rotation_matrix(six)


def test_checkpoint_and_dataset_profile(tmp_path):
    assert checkpoint_profile(tmp_path) == 'joint'
    assert not is_task_dataset({'action': {'names': None}})
    features = {'observation.state': {'names': FEATURE_NAMES}, 'action': {'names': FEATURE_NAMES}}
    assert is_task_dataset(features)
    with pytest.raises(ValueError):
        is_task_dataset({**features, 'action': {'names': None}})
    path = tmp_path / 'action_profile.json'
    path.write_text(json.dumps(profile_metadata()))
    assert checkpoint_profile(tmp_path) == PROFILE
    path.write_text('{}')
    with pytest.raises(ValueError):
        checkpoint_profile(tmp_path)


def test_inference_anchor_and_reset():
    identity = frame(0, np.eye(3))
    stationary = transform_actions(identity, identity)

    class Policy:

        def select_action(self, observation):
            return torch.from_numpy(stationary[None])

        def reset(self):
            pass

    manager = InferenceManager(device='cpu')
    manager.action_profile = PROFILE
    manager.policy = Policy()
    for step in range(HORIZON + 1):
        measured = frame(step, np.eye(3))
        result = manager.predict({}, measured)
        expected = identity if step < HORIZON else measured
        np.testing.assert_allclose(result, expected)
    manager.reset()
    np.testing.assert_allclose(manager.predict({}, identity), identity)
    manager.action_profile = 'joint'
    np.testing.assert_array_equal(manager.predict({}, identity), stationary)


def test_delta_statistics_stop_at_episode_boundary():
    positions = [0, 1, 2, 100, 102]
    table = Dataset.from_dict({
        'observation.state': [frame(x, np.eye(3)) for x in positions],
        'action': [frame(x + 1, np.eye(3)) for x in positions],
    })
    meta = SimpleNamespace(features=table.features, fps=10, stats={})
    dataset = SimpleNamespace(meta=meta, fps=meta.fps, hf_dataset=table,
                              episode_data_index={'from': [0, 3], 'to': [3, 5]})
    policy = ACTConfig(device='cpu')
    prepare_task_dataset(dataset, policy)
    assert policy.n_action_steps == HORIZON
    assert dataset.delta_indices['action'] == list(range(HORIZON))
    stats = meta.stats['action']
    assert stats['count'].item() == 9
    np.testing.assert_allclose(stats['mean'][0], np.mean([1, 2, 3, 1, 2, 1, 1, 3, 1]))
    assert stats['max'][0] == 3
