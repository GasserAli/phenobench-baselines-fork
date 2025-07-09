""" Validate semantic segmentation model.
"""
import argparse
import os
import pdb
from typing import Dict
import wandb

import oyaml as yaml
from pytorch_lightning import Trainer

from callbacks import (ConfigCallback, PostprocessorrCallback,
                       VisualizerCallback, get_postprocessors, get_visualizers,ECECallback,EntropyVisualizationCallback,TestLossCallback,IoUCallback,UncertaintyCallbacks)
from datasets import get_data_module
from modules import get_backbone, get_criterion, module


def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--export_dir", required=True, help="Path to export dir which saves logs, metrics, etc.")
  parser.add_argument("--config", required=True, help="Path to configuration file (*.yaml)")
  parser.add_argument("--ckpt_path", required=True, help="Provide *.ckpt file to continue training.")

  args = vars(parser.parse_args())

  return args


def load_config(path_to_config_file: str) -> Dict:
  assert os.path.exists(path_to_config_file)

  with open(path_to_config_file) as istream:
    config = yaml.safe_load(istream)

  return config
 
def main(args: dict, learning_rate: float, batch_size: int, optimizer: str, resize: int):
  args = parse_args()
  cfg = load_config(args['config'])

  datasetmodule = get_data_module(cfg)
  criterion = get_criterion(cfg)

  # define backbone
  network = get_backbone(cfg)
  
  for i in ["train", "val", "test"]:
    cfg[f"{i}"]["batch_size"] = batch_size
    cfg[f"{i}"]["geometric_data_augmentations"]["image_resize"]["x_resize"] = resize
    cfg[f"{i}"]["geometric_data_augmentations"]["image_resize"]["y_resize"] = resize
    if i == "train":
      cfg[f"{i}"]["learning_rate"] = learning_rate

  seg_module = module.SegmentationNetwork(network, 
                                          criterion, 
                                          cfg['train']['learning_rate'],
                                          cfg['train']['weight_decay'],
                                          optimizer=optimizer,
                                          test_step_settings=cfg['test']['step_settings'])

  # Add callbacks
  visualizer_callback = VisualizerCallback(get_visualizers(cfg), cfg['train']['vis_train_every_x_epochs'])
  postprocessor_callback = PostprocessorrCallback(
      get_postprocessors(cfg), cfg['train']['postprocess_train_every_x_epochs'])
  config_callback = ConfigCallback(cfg)
  ece_callback = ECECallback()
  entropy_callback = EntropyVisualizationCallback()
  test_loss_callback = TestLossCallback()
  iou_callback = IoUCallback()
  uncertaintyCallbacks=UncertaintyCallbacks()
  
  

  # Setup trainer
  trainer = Trainer(default_root_dir=args['export_dir'],
                    gpus=cfg['test']['n_gpus'],
                    callbacks=[visualizer_callback, postprocessor_callback, config_callback,ece_callback, entropy_callback, test_loss_callback, iou_callback,uncertaintyCallbacks],)
  trainer.test(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])

def test(config = None):
  args = parse_args()

  with wandb.init(config=config):
    config = wandb.config
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    optimizer = config.optimizer
    resize = config.resize
    main(args, learning_rate, batch_size, optimizer, resize)


if __name__ == '__main__':
  sweep_config = {
    'method': 'grid',
    'metric': { 
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [8]
        },
        'resize': {
            'values': [128]
        },
        'optimizer': {
            'values': ['adam']
        },
        'learning_rate': {
            'values': [0.0001]
        },
    }
  }

  sweep_id = wandb.sweep(sweep_config, project="newPhenoTesting")
  # train()
  wandb.agent(sweep_id = sweep_id, project="newPhenoTesting", function=test)
