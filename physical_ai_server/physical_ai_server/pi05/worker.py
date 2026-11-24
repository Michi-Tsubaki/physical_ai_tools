"""Executed with the isolated Pi05 interpreter."""

import importlib.metadata
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.train_utils import load_training_state, save_training_state
from safetensors import safe_open
from safetensors.torch import load_file, load_model


class Worker:
    def load(self, source, device, training=None):
        config = PreTrainedConfig.from_pretrained(source, cli_overrides=[f'--device={device}'])
        if config.type != 'pi05':
            raise ValueError(f'Expected pi05 weights, found {config.type}')
        fresh = training is not None and not training['resume']
        if training:
            torch.manual_seed(training['seed'])
            inputs = {key: PolicyFeature(type=FeatureType(ft['type']), shape=tuple(ft['shape']))
                      for key, ft in training['features'].items()}
            outputs = {'action': inputs.pop('action')}
            if fresh:
                config.input_features, config.output_features = inputs, outputs
                config.chunk_size, config.n_action_steps = training['horizon'], training['execute']
                config.normalization_mapping.update(STATE=NormalizationMode.MIN_MAX,
                                                    ACTION=NormalizationMode.MIN_MAX)
                config.use_relative_actions = False
                config.gradient_checkpointing = True
                if device.startswith('cuda') and torch.cuda.is_bf16_supported():
                    config.dtype = 'bfloat16'
            elif (config.input_features != inputs or config.output_features != outputs
                  or config.chunk_size != training['horizon'] or config.n_action_steps != training['execute']):
                raise ValueError('Pi05 checkpoint features or chunk length differ from the dataset')
        if not config.image_features or config.robot_state_feature.shape[0] > config.max_state_dim or (
                config.action_feature.shape[0] > config.max_action_dim):
            raise ValueError('Pi05 requires RGB input and state/action dimensions within its padded limits')

        weight_file = (Path(source) / 'model.safetensors' if Path(source).is_dir() else
                       hf_hub_download(source, 'model.safetensors'))
        self.policy = PI05Policy(config)
        with safe_open(str(weight_file), framework='pt') as weights:
            base_weights = 'action_in_proj.weight' in weights.keys()
        if base_weights:
            state = self.policy._fix_pytorch_state_dict_keys(load_file(str(weight_file)), config)
            self.policy.load_state_dict({
                (k if k.startswith('model.') else 'model.' + k): v for k, v in state.items()
            }, strict=True)
            del state
        else:
            load_model(self.policy, str(weight_file), strict=True)

        if fresh:
            self.pre, self.post = make_pre_post_processors(config, dataset_stats=training['stats'])
        else:
            self.pre, self.post = make_pre_post_processors(
                config, pretrained_path=source,
                preprocessor_overrides={'device_processor': {'device': device}},
                postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        if training:
            self.optimizer = config.get_optimizer_preset().build(self.policy.get_optim_params())
            self.scheduler = config.get_scheduler_preset().build(self.optimizer, training['steps'])
            self.policy.train()
            if training['resume']:
                step, self.optimizer, self.scheduler = load_training_state(
                    Path(source).parent, self.optimizer, self.scheduler)
                return step
            return 0
        self.policy.eval()
        return {'type': config.type, 'n_action_steps': config.n_action_steps}

    def preprocess(self, batch):
        return self.pre({key: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                         for key, v in batch.items()})

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self.policy(self.preprocess(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy.config.optimizer_grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().item()

    def save(self, directory, step):
        path = Path(directory) / 'pretrained_model'
        self.policy.save_pretrained(path)
        self.pre.save_pretrained(path)
        self.post.save_pretrained(path)
        save_training_state(Path(directory), step, self.optimizer, self.scheduler)

    @torch.inference_mode()
    def predict(self, batch):
        chunk = self.policy.predict_action_chunk(self.preprocess(batch))
        return self.post(chunk[:, :self.policy.config.n_action_steps])[0].float().cpu().numpy()


def main(fd):
    with Connection(fd) as channel:
        worker = Worker()
        while True:
            try:
                operation, data = channel.recv()
            except EOFError:
                return
            try:
                if importlib.metadata.version('lerobot') != '0.5.1':
                    raise RuntimeError('The Pi05 worker requires lerobot[pi]==0.5.1')
                method = {'load': worker.load, 'train': worker.train_step,
                          'save': worker.save, 'predict': worker.predict}[operation]
                result = method(**data)
            except Exception:
                channel.send((False, traceback.format_exc()))
                return
            channel.send((True, result))


if __name__ == '__main__':
    main(int(sys.argv[1]))
