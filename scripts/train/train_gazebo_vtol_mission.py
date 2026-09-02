#!/usr/bin/env python
"""Train the GPU-vectorized standard_vtol point-to-point landing mission.

Command-line values appended by the caller override the defaults below.
"""

import sys
import logging

from train_F16sim import main


DEFAULT_ARGS = [
    '--env-name', 'Control',
    '--scenario-name', 'gazebo_vtol_mission',
    '--model-name', 'GAZEBO',
    '--algorithm-name', 'cpo',
    '--experiment-name', 'vtol_mission_nowind',
    '--cuda',
    '--device', 'cuda:0',
    '--n-rollout-threads', '1024',
    '--num-env-steps', '50000000',
    '--buffer-size', '256',
    '--ppo-epoch', '5',
    '--num-mini-batch', '8',
    '--data-chunk-length', '32',
    '--lr', '0.0003',
    '--gamma', '0.995',
    '--gae-lambda', '0.95',
    '--entropy-coef', '0.01',
    '--target-kl', '0.03',
    '--use-safety-aux',
    '--use-cost-constraints',
    '--safety-aux-horizon', '50',
    '--safety-aux-loss-coef', '0.1',
    '--safety-aux-pos-weight', '5.0',
    '--cost-limit', '0.05',
    '--cost-lagrange-init', '1.0',
    '--cost-lagrange-lr', '0.02',
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main(DEFAULT_ARGS + sys.argv[1:])
