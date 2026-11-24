import torch

from lerobot.datasets.compute_stats import aggregate_feature_stats, get_feature_stats
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import get_delta_indices
from physical_ai_server.data_processing.task_space import HORIZON, transform_actions


def configure_task_policy(policy):
    policy.n_action_steps = HORIZON
    if policy.type == 'diffusion':
        factor = 2 ** len(policy.down_dims)
        required = HORIZON + policy.n_obs_steps - 1
        policy.horizon = ((required + factor - 1) // factor) * factor
        policy.drop_n_last_frames = policy.horizon - required
    elif policy.type in ('act', 'smolvla', 'pi0', 'pi05', 'pi0fast'):
        policy.chunk_size = HORIZON
        if getattr(policy, 'temporal_ensemble_coeff', None) is not None:
            raise ValueError('Temporal ensembling requires actions in a common absolute frame')
    else:
        raise ValueError(f'{policy.type} does not support 30-step task space chunks')


def prepare_task_dataset(dataset, policy):
    configure_task_policy(policy)
    dataset.delta_timestamps = resolve_delta_timestamps(policy, dataset.meta)
    dataset.delta_indices = get_delta_indices(dataset.delta_timestamps, dataset.fps)
    table = dataset.hf_dataset.with_format('numpy', columns=['observation.state', 'action'])
    stats = []
    for start, end in zip(dataset.episode_data_index['from'], dataset.episode_data_index['to']):
        episode = table[int(start):int(end)]
        states, actions = episode['observation.state'], episode['action']
        for offset in policy.action_delta_indices:
            first, last = max(0, -offset), min(len(states), len(states) - offset)
            if first < last:
                delta = transform_actions(states[first:last], actions[first + offset:last + offset])
                stats.append(get_feature_stats(delta, axis=0, keepdims=False))
    dataset.meta.stats['action'] = aggregate_feature_stats(stats)


def relative_training_actions(batch):
    states = batch['observation.state']
    anchor = states[:, -1] if states.ndim == 3 else states
    delta = transform_actions(anchor.numpy()[:, None, :], batch['action'].numpy())
    return torch.from_numpy(delta)
