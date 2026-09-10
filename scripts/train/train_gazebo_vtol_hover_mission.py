#!/usr/bin/env python
"""Train the GPU-vectorized standard_vtol point-to-point hover mission."""

import logging
import sys

from train_F16sim import main


DEFAULT_ARGS = [
    '--env-name', 'Control',
    '--scenario-name', 'gazebo_vtol_hover_mission',
    '--model-name', 'GAZEBO',
    '--algorithm-name', 'ppo',
    '--experiment-name', 'vtol_point_hover_ppo',
    '--cuda',
    '--device', 'cuda:0',
    '--n-rollout-threads', '1024',
    '--num-env-steps', '2000000000',
    '--buffer-size', '256',
    '--ppo-epoch', '2',
    '--num-mini-batch', '8',
    '--data-chunk-length', '32',
    '--action-log-std-init', '-1.609437912',
    '--action-mean-init', '0.08', '0.08', '0.08', '0.08',
    '0.0', '0.0', '0.0', '0.0',
    '--lr', '0.00003',
    '--gamma', '0.999',
    '--gae-lambda', '0.98',
    '--entropy-coef', '0.01',
    '--target-kl', '0.03',
    '--max-log-ratio', '4.0',
    '--use-clipped-value-loss',
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main(DEFAULT_ARGS + sys.argv[1:])
