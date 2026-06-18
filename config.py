import argparse
from tokenize import group


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = _get_prepare_config(parser)
    parser = _get_replaybuffer_config(parser)
    parser = _get_network_config(parser)
    parser = _get_recurrent_config(parser)
    parser = _get_optimizer_config(parser)
    parser = _get_ppo_config(parser)
    parser = _get_adv_config(parser)
    parser = _get_advnet_config(parser)
    parser = _get_selfplay_config(parser)
    parser = _get_save_config(parser)
    parser = _get_log_config(parser)
    parser = _get_eval_config(parser)
    parser = _get_render_config(parser)
    parser = _get_cir_config(parser)
    return parser


def _get_prepare_config(parser: argparse.ArgumentParser):
    """
    Prepare parameters:
        --algorithm-name <str>
            specifiy the algorithm, including `["ppo", "mappo"]`
        --experiment-name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default False, will use CPU to train; or else will use GPU;
        --n-training-threads <int>
            number of training threads working in parallel. by default 1
        --n-rollout-threads <int>
            number of parallel envs for training rollout. by default 4
        --n-render-rollout-threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num-env-steps <float>
            number of env steps to train (default: 1e7)
        --model-dir <str>
            by default None. set the path to pretrained model.
    """
    group = parser.add_argument_group("Prepare parameters")
    group.add_argument("--algorithm-name", type=str, default='ppo', choices=["ppo", "mappo"],
                       help="Specifiy the algorithm (default ppo)")
    group.add_argument("--experiment-name", type=str, default="check",
                       help="An identifier to distinguish different experiment.")
    group.add_argument("--seed", type=int, default=1,
                       help="Random seed for numpy/torch")
    group.add_argument("--device", type=str, default='cuda:0',
                       help="GPU id")
    group.add_argument("--cuda", action='store_true', default=False,
                       help="By default False, will use CPU to train; or else will use GPU;")
    group.add_argument("--n-training-threads", type=int, default=1,
                       help="Number of torch threads for training (default 1)")
    group.add_argument("--n-rollout-threads", type=int, default=4,
                       help="Number of parallel envs for training/evaluating rollout (default 4)")
    group.add_argument("--num-env-steps", type=float, default=1e7,
                       help='Number of environment steps to train (default: 1e7)')
    group.add_argument("--model-dir", type=str, default=None,
                       help="By default None. set the path to pretrained model.")
    group.add_argument("--init-actor-ckpt", type=str, default=None,
                       help="Optional actor checkpoint used to initialise PPO actor before training. "
                            "Supports raw actor state_dict, {'state_dict': ...}, and DAgger {'policy': ...}.")
    group.add_argument("--freeze-head-motor", action='store_true', default=False,
                       help="Force head (pusher) motor action[0] to -1 (zero thrust) at every step. "
                            "Use for hover task to match the PID expert which never activates motor 0.")
    return parser


def _get_replaybuffer_config(parser: argparse.ArgumentParser):
    """
    Replay Buffer parameters:
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --buffer-size <int>
            the maximum storage in the buffer.
        --use-proper-time-limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use-gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gae-lambda <float>
            gae lambda parameter (default: 0.95)
    """
    group = parser.add_argument_group("Replay Buffer parameters")
    group.add_argument("--gamma", type=float, default=0.99,
                       help='discount factor for rewards (default: 0.99)')
    group.add_argument("--buffer-size", type=int, default=200,
                       help="maximum storage in the buffer.")
    group.add_argument("--use-proper-time-limits", action='store_true', default=False,
                       help='compute returns taking into account time limits')
    group.add_argument("--use-gae", action='store_false', default=True,
                       help='Whether to use generalized advantage estimation')
    group.add_argument("--gae-lambda", type=float, default=0.95,
                       help='gae lambda parameter (default: 0.95)')
    return parser


def _get_network_config(parser: argparse.ArgumentParser):
    """
    Network parameters:
        --hidden-size <str>
            dimension of hidden layers for mlp pre-process networks
        --act-hidden-size <int>
            dimension of hidden layers for actlayer
        --activation-id
            choose 0 to use Tanh, 1 to use ReLU, 2 to use LeakyReLU, 3 to use ELU
        --use-feature-normalization
            by default False, otherwise apply LayerNorm to normalize feature extraction inputs.
        --gain
            by default 0.01, use the gain # of last action layer
    """
    group = parser.add_argument_group("Network parameters")
    group.add_argument("--hidden-size", type=str, default='128 128',
                       help="Dimension of hidden layers for mlp pre-process networks (default '128 128')")
    group.add_argument("--act-hidden-size", type=str, default='128 128',
                       help="Dimension of hidden layers for actlayer (default '128 128')")
    group.add_argument("--activation-id", type=int, default=1,  
                       help="Choose 0 to use Tanh, 1 to use ReLU, 2 to use LeakyReLU, 3 to use ELU (default 1)")
    group.add_argument("--use-feature-normalization", action='store_true', default=True,
                       help="Whether to apply LayerNorm to the feature extraction inputs")
    group.add_argument("--gain", type=float, default=0.01,
                       help="The gain # of last action layer")
    group.add_argument("--use-prior", action='store_true', default=False,
                       help="Whether to use prior hunman info to update network, use only on missile shoot task")
    return parser


def _get_recurrent_config(parser: argparse.ArgumentParser):
    """
    Recurrent parameters:
        --use-recurrent-policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent-hidden-size <int>
            Dimension of hidden layers for recurrent layers (default 128).
        --recurrent-hidden-layers <int>
            The number of recurrent layers (default 1).
        --data-chunk-length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    """
    group = parser.add_argument_group("Recurrent parameters")
    group.add_argument("--use-recurrent-policy", action='store_false', default=True,
                       help='Whether to use a recurrent policy')
    group.add_argument("--recurrent-hidden-size", type=int, default=128,
                       help="Dimension of hidden layers for recurrent layers (default 128)")
    group.add_argument("--recurrent-hidden-layers", type=int, default=1,
                       help="The number of recurrent layers (default 1)")
    group.add_argument("--data-chunk-length", type=int, default=10,
                       help="Time length of chunks used to train a recurrent_policy (default 10)")
    return parser


def _get_optimizer_config(parser: argparse.ArgumentParser):
    """
    Optimizer parameters:
        --lr <float>
            learning rate parameter (default: 5e-4, fixed).
    """
    group = parser.add_argument_group("Optimizer parameters")
    group.add_argument("--lr", type=float, default=5e-4,
                       help='learning rate (default: 5e-4)')
    return parser


def _get_ppo_config(parser: argparse.ArgumentParser):
    """
    PPO parameters:
        --ppo-epoch <int>
            number of ppo epochs (default: 10)
        --clip-param <float>
            ppo clip parameter (default: 0.2)
        --use-clipped-value-loss
            by default false. If set, clip value loss.
        --num-mini-batch <int>
            number of batches for ppo (default: 1)
        --value-loss-coef <float>
            ppo value loss coefficient (default: 1)
        --entropy-coef <float>
            ppo entropy term coefficient (default: 0.01)
        --use-max-grad-norm
            by default, use max norm of gradients. If set, do not use.
        --max-grad-norm <float>
            max norm of gradients (default: 0.5)
    """
    group = parser.add_argument_group("PPO parameters")
    group.add_argument("--ppo-epoch", type=int, default=10,
                       help='number of ppo epochs (default: 10)')
    group.add_argument("--clip-param", type=float, default=0.2,
                       help='ppo clip parameter (default: 0.2)')
    group.add_argument("--use-clipped-value-loss", action='store_true', default=False,
                       help="By default false. If set, clip value loss.")
    group.add_argument("--num-mini-batch", type=int, default=1,
                       help='number of batches for ppo (default: 1)')
    group.add_argument("--value-loss-coef", type=float, default=1,
                       help='ppo value loss coefficient (default: 1)')
    group.add_argument("--entropy-coef", type=float, default=0.01,
                       help='entropy term coefficient (default: 0.01)')
    group.add_argument("--use-max-grad-norm", action='store_false', default=True,
                       help="By default, use max norm of gradients. If set, do not use.")
    group.add_argument("--max-grad-norm", type=float, default=2,
                       help='max norm of gradients (default: 2)')
    group.add_argument("--target-kl", type=float, default=0.0,
                       help='Skip PPO minibatch updates when approximate KL exceeds this value. '
                            '0 disables the guard.')
    group.add_argument("--max-log-ratio", type=float, default=20.0,
                       help='Clamp PPO log probability ratio before exp to avoid inf ratios.')
    group.add_argument("--use-safety-aux", action='store_true', default=False,
                       help='Enable auxiliary actor-head prediction of bad_done within a future horizon.')
    group.add_argument("--safety-aux-horizon", type=int, default=25,
                       help='Future transition horizon for safety auxiliary bad_done labels.')
    group.add_argument("--safety-aux-loss-coef", type=float, default=0.1,
                       help='Loss coefficient for the safety auxiliary BCE loss.')
    group.add_argument("--safety-aux-pos-weight", type=float, default=5.0,
                       help='Positive-class weight for safety auxiliary BCE loss.')
    return parser

def _get_adv_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Adv parameters")
    group.add_argument("--magnitude", type=float, default=0.5,
                       help='noise amplification factor')
    group.add_argument("--adv-action-lr", type=float, default=0.1,
                       help='adversary action learning rate')
    group.add_argument("--epsilon", type=float, default=0.01,
                    help='adversary noise limitation')
    group.add_argument("--horizon", type=int, default=10,
                    help='return count horizon')
    group.add_argument("--noise", type=str, default='adv_noise',
                    help='adversary action noise type')
    group.add_argument("--sudden-mag", type=float, default=0.02,
                    help='sudden noise probability')
    group.add_argument("--random-weight", type=float, default=0.02,
                    help='random noise weight')
    return parser

def _get_advnet_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Advnet parameters")
    group.add_argument("--adv-gamma", type=float, default=0.99,
                       help='discount factor for rewards')
    group.add_argument("--adv-value-loss-coef", type=float, default=1.0,
                       help='adversary value loss coefficient')
    group.add_argument("--adv-max-grad-norm", type=float, default=1.,
                    help='max norm of gradients (default: 2)')
    group.add_argument("--adv-num-learning-epochs", type=int, default=5,
                    help='num learning epochs')
    group.add_argument("--adv-num-mini-batch", type=int, default=5,
                    help='number of batches for adv')
    group.add_argument("--adv-lr", type=float, default=1.e-3,
                    help='adversary learning rate')
    return parser

def _get_selfplay_config(parser: argparse.ArgumentParser):
    """
    Selfplay parameters:
        --use-selfplay
            by default false. If set, use selfplay algorithms.
        --selfplay-algorithm <str>
            specifiy the selfplay algorithm, including `["sp", "fsp"]`
        --n-choose-opponents <int>
            number of different opponents chosen for rollout. (default 1)
        --init-elo <float>
            initial ELO for policy performance. (default 1000.0)
    """
    group = parser.add_argument_group("Selfplay parameters")
    group.add_argument("--use-selfplay", action='store_true', default=False,
                       help="By default false. If set, use selfplay algorithms.")
    group.add_argument("--selfplay-algorithm", type=str, default='sp', choices=["sp", "fsp", "pfsp"],
                       help="Specifiy the selfplay algorithm (default 'sp')")
    group.add_argument('--n-choose-opponents', type=int, default=1,
                       help="number of different opponents chosen for rollout. (default 1)")
    group.add_argument('--init-elo', type=float, default=1000.0,
                       help="initial ELO for policy performance. (default 1000.0)")
    return parser


def _get_save_config(parser: argparse.ArgumentParser):
    """
    Save parameters:
        --save-interval <int>
            time duration between contiunous twice models saving.
    """
    group = parser.add_argument_group("Save parameters")
    group.add_argument("--save-interval", type=int, default=1,
                       help="time duration between contiunous twice models saving. (default 1)")
    return parser



 
def _get_log_config(parser: argparse.ArgumentParser):
    """
    Log parameters:
        --log-interval <int>
            time duration between contiunous twice log printing.
    """
    group = parser.add_argument_group("Log parameters")
    group.add_argument("--log-interval", type=int, default=1,
                       help="time duration between contiunous twice log printing. (default 1)")
    return parser


def _get_eval_config(parser: argparse.ArgumentParser):
    """
    Eval parameters:
        --use-eval
            by default, do not start evaluation. If set, start evaluation alongside with training.
        --n-eval-rollout-threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --eval-interval <int>
            time duration between contiunous twice evaluation progress.
        --eval-episodes <int>
            number of episodes of a single evaluation.
    """
    group = parser.add_argument_group("Eval parameters")
    group.add_argument("--use-eval", action='store_true', default=False,
                       help="by default, do not start evaluation. If set, start evaluation alongside with training.")
    group.add_argument("--n-eval-rollout-threads", type=int, default=1000,
                       help="Number of parallel envs for evaluating rollout (default 1)")
    group.add_argument("--eval-interval", type=int, default=25,
                       help="time duration between contiunous twice evaluation progress. (default 25)")
    group.add_argument("--eval-episodes", type=int, default=32,
                       help="number of episodes of a single evaluation. (default 32)")
    return parser

def _get_render_config(parser: argparse.ArgumentParser):
    """
    Render parameters:
        --render-opponent-index <int>
            the index of opponent policy in the opponent pool. by default 0
        --render-index <int>
            the index of opponent policy in the opponent pool. by default 0
    """
    group = parser.add_argument_group("Render parameters")
    group.add_argument("--render-opponent-index", type=str, default='latest', help="the index of opponent policy in the opponent pool. by default latest")
    group.add_argument("--render-index", type=str, default='latest', help="the index of ego policy. by default latest")
    return parser

def _get_cir_config(parser: argparse.ArgumentParser):
    """
    Cirriculum parameters:
    """
    group = parser.add_argument_group("Cirriculum parameters")
    group.add_argument("--adv-cir", action='store_true', default='False', help="Use C-RPARL adversary cirriculum")
    group.add_argument("--tracking-cir", action='store_true', default='True', help="Use Tracking task cirriculum")
    return parser

if __name__ == "__main__":
    parser = get_config()
    all_args = parser.parse_args()
