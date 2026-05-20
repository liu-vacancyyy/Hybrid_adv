"""
DAgger student policy. Architecture is identical to
``algorithms.ppo.ppo_actor.PPOActor`` so the resulting state-dict can be
loaded into a PPO runner verbatim.

forward(obs, rnn_states, masks) and act(obs) follow the PPOActor signature so
recurrent chunk training (T>1) works out of the box. DAgger uses the
deterministic mean (== action_dist.mode()) for the MSE loss.
"""
from types import SimpleNamespace
import torch
import torch.nn as nn

from algorithms.ppo.ppo_actor import PPOActor


def default_actor_args(**overrides):
    """SimpleNamespace compatible with PPOActor's ``args``.

    Defaults mirror scripts/train_rc.sh / scripts/train_heading.sh.
    """
    defaults = dict(
        hidden_size='128 128',
        act_hidden_size='128 128',
        activation_id=1,
        use_feature_normalization=True,
        use_recurrent_policy=False,
        recurrent_hidden_size=128,
        recurrent_hidden_layers=1,
        use_prior=False,
        gain=0.01,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PPOActorStudent(nn.Module):
    """Wrapper around PPOActor exposing a DAgger-friendly interface.

    Public API:
        forward(obs, rnn_states, masks, deterministic=True)
            -> (actions, new_rnn_states)
            Differentiable. Supports T=1 (rollout) and T>1 (chunk training).
        act(obs) -> actions
            No-grad rollout helper. Maintains a rollout hidden state across
            calls. Use reset_rollout_state(...) to zero it on env resets and
            set_done_mask(done) right after env.step.

    state_dict() / load_state_dict() forward directly to the underlying
    PPOActor so checkpoints are interchangeable with the PPO runner.
    """

    def __init__(self, obs_space, act_space, args=None, device='cpu'):
        super().__init__()
        if args is None:
            args = default_actor_args()
        self.args = args
        self.device = torch.device(device)
        self.actor = PPOActor(args, obs_space, act_space, device=self.device)

        # Rollout-time hidden state and mask (used by act())
        self._rollout_rnn_states = None
        self._rollout_masks = None

    # ------------------------------------------------------------------ #
    @property
    def use_recurrent_policy(self):
        return bool(self.args.use_recurrent_policy)

    @property
    def recurrent_hidden_size(self):
        return int(self.args.recurrent_hidden_size)

    @property
    def recurrent_hidden_layers(self):
        return int(self.args.recurrent_hidden_layers)

    def zeros_rnn_states(self, batch):
        """Zero hidden state of shape [batch, L, H] (matches PPO buffer)."""
        return torch.zeros(
            batch,
            self.recurrent_hidden_layers,
            self.recurrent_hidden_size,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Rollout state management
    # ------------------------------------------------------------------ #
    def reset_rollout_state(self, batch, reset_mask=None):
        """Zero rollout hidden state (entirely, or only rows in reset_mask).

        Also forces those rows' next-step masks to 1 (start of episode).
        """
        if self._rollout_rnn_states is None or self._rollout_rnn_states.shape[0] != batch:
            self._rollout_rnn_states = self.zeros_rnn_states(batch)
            self._rollout_masks = torch.ones(batch, 1, device=self.device)
            return
        if reset_mask is None:
            self._rollout_rnn_states.zero_()
            self._rollout_masks.fill_(1.0)
        else:
            self._rollout_rnn_states[reset_mask] = 0.0
            self._rollout_masks[reset_mask] = 1.0

    def set_done_mask(self, done_mask):
        """Mark just-terminated envs so the next forward zeros their hidden.

        ``done_mask`` is the boolean tensor returned by env.step (any of
        done|bad_done|exceed). Internally stores ``masks = 1 - done`` so the
        next call to ``act`` will gate the GRU's hidden via that mask.
        """
        if self._rollout_masks is None:
            return
        m = (1.0 - done_mask.float()).reshape(-1, 1)
        self._rollout_masks.copy_(m)

    # ------------------------------------------------------------------ #
    # Forward / act
    # ------------------------------------------------------------------ #
    def forward(self, obs, rnn_states=None, masks=None, deterministic=True):
        """
        Differentiable forward.

        Shapes:
            T=1 (rollout / non-recurrent training):
                obs        : [N, obs_dim]
                rnn_states : [N, L, H]  (zeros if non-recurrent)
                masks      : [N, 1]
            T>1 (recurrent chunk training):
                obs        : [T*N, obs_dim]   (time-major flatten)
                rnn_states : [N, L, H]        (state BEFORE the chunk start)
                masks      : [T*N, 1]
        Returns deterministic mean action and new rnn_states.
        """
        batch = obs.shape[0]
        if rnn_states is None:
            rnn_states = (self.zeros_rnn_states(batch)
                          if self.use_recurrent_policy
                          else torch.zeros(batch, 1, 1, device=self.device))
        if masks is None:
            masks = torch.ones(batch, 1, device=self.device)

        actions, _log_probs, new_rnn = self.actor(
            obs, rnn_states, masks, deterministic=deterministic
        )
        return actions, new_rnn

    @torch.no_grad()
    def act(self, obs):
        """No-grad rollout step. Returns the deterministic mean action."""
        batch = obs.shape[0]
        if self.use_recurrent_policy:
            if (self._rollout_rnn_states is None
                    or self._rollout_rnn_states.shape[0] != batch):
                self.reset_rollout_state(batch)
            rnn_states = self._rollout_rnn_states
            masks = self._rollout_masks
        else:
            rnn_states = torch.zeros(batch, 1, 1, device=self.device)
            masks = torch.ones(batch, 1, device=self.device)

        was_training = self.training
        self.eval()
        actions, _lp, new_rnn = self.actor(
            obs, rnn_states, masks, deterministic=True
        )
        if was_training:
            self.train()

        if self.use_recurrent_policy:
            self._rollout_rnn_states = new_rnn.detach()
        return actions

    # ------------------------------------------------------------------ #
    # PPO-compatible checkpointing
    # ------------------------------------------------------------------ #
    def state_dict(self, *a, **kw):
        return self.actor.state_dict(*a, **kw)

    def load_state_dict(self, sd, strict=True):
        return self.actor.load_state_dict(sd, strict=strict)
