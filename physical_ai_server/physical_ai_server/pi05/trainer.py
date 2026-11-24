from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.utils.train_utils import get_step_checkpoint_dir, update_last_checkpoint
from torch import Generator
from torch.utils.data import DataLoader

from physical_ai_server.data_processing.task_space import profile_metadata
from physical_ai_server.pi05.client import Pi05Runtime, cpu_batch
from physical_ai_server.training.task_space import relative_training_actions
from physical_ai_server.training.trainers.lerobot.lerobot_trainer import LerobotTrainer


@PreTrainedConfig.register_subclass('pi05')
@dataclass
class Pi05DatasetConfig(PI0Config):
    """Legacy dataset window and training configuration for the isolated Pi05 model."""

    base_model: str = 'lerobot/pi05_base'
    optimizer_weight_decay: float = 0.01


class Pi05Trainer(LerobotTrainer):
    def train(self, cfg, stop_event=None):
        self._validate_config(cfg)
        dataset = self._create_dataset(cfg)
        features = dataset_to_policy_features(dataset.features)
        training = {
            'features': {key: {'type': ft.type.value, 'shape': ft.shape} for key, ft in features.items()},
            'stats': {key: dataset.meta.stats[key] for key in ('observation.state', 'action')},
            'horizon': cfg.policy.chunk_size, 'execute': cfg.policy.n_action_steps,
            'seed': cfg.seed if cfg.seed is not None else 1000, 'steps': cfg.steps, 'resume': cfg.resume,
        }
        source = (cfg.policy.pretrained_path if cfg.resume else
                  os.environ.get('PHYSICAL_AI_PI05_BASE', cfg.policy.base_model))
        runtime = Pi05Runtime(stop_event)
        try:
            self.current_step = runtime.call('load', source=str(source), device=cfg.policy.device,
                                             training=training)
            batches = cycle(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers,
                                       generator=Generator().manual_seed(training['seed'])))
            for step in range(self.current_step + 1, cfg.steps + 1):
                if stop_event is not None and stop_event.is_set():
                    break
                batch = next(batches)
                if self.task_space:
                    batch['action'] = relative_training_actions(batch)
                self.current_loss = runtime.call('train', batch=cpu_batch(batch))
                self.current_step = step
                if cfg.log_freq and step % cfg.log_freq == 0:
                    self.logger.info(f'Pi05 step {step} loss {self.current_loss:.6f}')
                if cfg.save_checkpoint and (step == cfg.steps or step % cfg.save_freq == 0):
                    path = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=path.parent) as temp:
                        checkpoint = Path(temp) / 'checkpoint'
                        runtime.call('save', directory=str(checkpoint), step=step)
                        model = checkpoint / 'pretrained_model'
                        cfg.save_pretrained(model)
                        if self.task_space:
                            (model / 'action_profile.json').write_text(json.dumps(profile_metadata()))
                        checkpoint.rename(path)
                    update_last_checkpoint(path)
        except InterruptedError:
            if stop_event is None or not stop_event.is_set():
                raise
        finally:
            runtime.close()
