"""
DAgger trainer with a PPO-style rollout buffer.

Per iteration k:
    1. beta_k = max(beta_min, beta0 * decay^k)
    2. rollout T steps with N parallel envs into one RolloutBlock:
            for t in 0..T-1:
                a_exp = expert(env.model)            # PID label
                a_stu = policy.act(obs)              # student rollout
                a_run = mix(a_exp, a_stu, beta_k)
                store (obs[t], a_exp[t], rnn_states[t], masks[t]) in block
                obs, ..., done = env.step(a_run)
                policy.set_done_mask(done)           # gate GRU at next step
                expert.reset(reset_mask)
        Append the block to a deque of recent blocks (DAgger aggregation).
        Optionally keep env/GRU/PID rollout state across iterations so short
        rollout blocks still traverse long trajectories over multiple collects.
    3. Train the student on the aggregated dataset:
            non-recurrent: feed-forward generator over all transitions
            recurrent    : chunk generator (chunks of length L) honoring
                           the masks recorded during rollout - identical
                           semantics to ``ReplayBuffer.recurrent_generator``.

Buffers live entirely on GPU as torch tensors.
"""
import os
import time
import contextlib
import io
from collections import deque
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class RolloutBlock:
    """One contiguous N-env rollout of length T, kept on `device`.

    Field shapes (with H = recurrent_hidden_size, L_g = recurrent_hidden_layers):
        obs        : [T, N, obs_dim]
        actions    : [T, N, act_dim]    (expert label)
        masks      : [T, N, 1]          (1 = continues from previous step)
        rnn_states : [T, N, L_g, H]     (state fed INTO step t)
    """

    def __init__(self, T, N, obs_dim, act_dim, L_g, H, device):
        self.T = T
        self.N = N
        self.obs        = torch.zeros(T, N, obs_dim, device=device)
        self.actions    = torch.zeros(T, N, act_dim, device=device)
        self.masks      = torch.ones (T, N, 1,       device=device)
        self.rnn_states = torch.zeros(T, N, L_g, H,  device=device)


class DAggerTrainer:
    def __init__(self, env, expert, policy,
                 rollout_steps=256,
                 max_blocks=6,
                 lr=3e-4, weight_decay=1e-5,
                 mini_batches=8, train_epochs=4,
                 data_chunk_length=8,
                 max_chunks_per_minibatch=4096,
                 beta0=1.0, beta_decay=0.5, beta_min=0.0,
                 max_grad_norm=1.0,
                 suppress_env_stdout=True,
                 continue_across_iters=False,
                 device='cuda:0',
                 ckpt_dir=None,
                 log_dir=None):
        self.env = env
        self.expert = expert
        self.policy = policy.to(device)
        self.device = torch.device(device)

        self.optim = torch.optim.Adam(self.policy.parameters(),
                                      lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        self.T = int(rollout_steps)
        self.max_blocks = int(max_blocks)
        self.mini_batches = int(mini_batches)
        self.train_epochs = int(train_epochs)
        self.data_chunk_length = int(data_chunk_length)
        self.max_chunks_per_minibatch = int(max_chunks_per_minibatch)
        self.max_grad_norm = float(max_grad_norm)
        self.suppress_env_stdout = bool(suppress_env_stdout)
        self.continue_across_iters = bool(continue_across_iters)

        self.beta0 = float(beta0)
        self.beta_decay = float(beta_decay)
        self.beta_min = float(beta_min)

        # Aggregated dataset: deque of recent RolloutBlocks
        self.blocks = deque(maxlen=self.max_blocks)

        self.ckpt_dir = ckpt_dir
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)

        # TensorBoard
        if log_dir is not None and _TB_AVAILABLE:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[TensorBoard] logging to {log_dir}")
        else:
            self.writer = None
            if log_dir is not None and not _TB_AVAILABLE:
                print("[TensorBoard] tensorboard not installed; skipping TB logging.")

        # Cache shapes
        self.N = env.n
        self.obs_dim = env.num_observation
        self.act_dim = env.num_actions
        self.L_g = self.policy.recurrent_hidden_layers
        self.H = self.policy.recurrent_hidden_size
        self._rollout_obs = None

    # ------------------------------------------------------------------ #
    def _new_block(self):
        return RolloutBlock(self.T, self.N, self.obs_dim, self.act_dim,
                            self.L_g, self.H, self.device)

    def _reset_rollout_stream(self):
        """Start a new rollout stream and reset recurrent controller state."""
        obs = self.env.reset()
        self.expert.reset()
        self.policy.reset_rollout_state(self.env.n)
        self._sync_expert_targets()
        self._rollout_obs = obs.detach()
        return self._rollout_obs

    def _sync_expert_targets(self):
        task = self.env.task
        if hasattr(task, 'target_altitude') and hasattr(task, 'target_heading'):
            self.expert.set_targets(
                target_altitude=task.target_altitude,
                target_heading=task.target_heading,
                target_npos=getattr(task, 'target_npos', None),
                target_epos=getattr(task, 'target_epos', None),
            )

    # ------------------------------------------------------------------ #
    def collect(self, beta):
        """Run T env steps and push one RolloutBlock to the deque.

        Asymmetric DAgger (privileged expert / noisy student):
          * a_exp = expert.compute_action(env.model)
              -> reads CLEAN, PRIVILEGED ground-truth state directly from
                 the dynamics model (no sensor noise applied).
          * a_stu = policy.act(obs)
              -> uses the NOISY observation that env.step() returned (the
                 task pipeline applies zero-mean Gaussian sensor noise via
                 HoverTask._build_obs(add_sensor_noise=True)).
          * block.obs[t]     = noisy student input  (training input)
            block.actions[t] = clean expert action  (training label)
          * a_run = mix(a_exp, a_stu, beta) drives the env; beta=0 yields
            a pure-student rollout (state distribution = student-induced).

        Returns mean per-step reward over the rollout.
        """
        env = self.env
        device = self.device

        if self.continue_across_iters:
            if self._rollout_obs is None:
                obs = self._reset_rollout_stream()
            else:
                obs = self._rollout_obs
                self._sync_expert_targets()
        else:
            obs = self._reset_rollout_stream()

        block = self._new_block()
        rew_sum      = torch.zeros(env.n, device=device)
        done_count     = 0
        bad_done_count = 0
        exceed_count   = 0

        for t in range(self.T):
            # Snapshot pre-step rnn state and mask (what GRU will see at step t)
            if self.policy.use_recurrent_policy:
                block.rnn_states[t] = self.policy._rollout_rnn_states.detach()
                block.masks[t] = self.policy._rollout_masks.detach()
            # else: rnn_states stays zeros, masks stays ones

            block.obs[t] = obs.detach()

            a_exp = self.expert.compute_action(env.model).detach()
            a_stu = self.policy.act(obs).detach()

            block.actions[t] = a_exp

            if beta >= 1.0:
                a_run = a_exp
            elif beta <= 0.0:
                a_run = a_stu
            else:
                use_exp = (torch.rand(env.n, 1, device=device) < beta).float()
                a_run = use_exp * a_exp + (1.0 - use_exp) * a_stu

            if self.suppress_env_stdout:
                with contextlib.redirect_stdout(io.StringIO()):
                    obs, rew, done, bad_done, exceed, _ = env.step(a_run)
            else:
                obs, rew, done, bad_done, exceed, _ = env.step(a_run)
            rew_sum += rew
            done_count     += int(done.sum())
            bad_done_count += int(bad_done.sum())
            exceed_count   += int(exceed.sum())

            reset_mask = (done | bad_done | exceed)
            # Tell the student to gate its GRU on the next step.
            self.policy.set_done_mask(reset_mask)
            if reset_mask.any():
                self.expert.reset(reset_mask)
                # Reset terminated envs before the next action is computed.
                # Otherwise the next action would be based on terminal obs but
                # applied after BaseEnv.step() silently resets those env rows.
                obs = env.reset()
            self._sync_expert_targets()

        self._rollout_obs = obs.detach()
        self.blocks.append(block)
        total_steps = self.T * self.N
        self._last_done_rate     = done_count     / total_steps
        self._last_bad_done_rate = bad_done_count / total_steps
        self._last_exceed_rate   = exceed_count   / total_steps
        return float((rew_sum / self.T).mean())

    # ------------------------------------------------------------------ #
    # Generators
    # ------------------------------------------------------------------ #
    def _feed_forward_batches(self):
        """Yield (obs, action) batches uniformly over all stored transitions."""
        if not self.blocks:
            return
        obs_all = torch.cat([b.obs.reshape(-1, self.obs_dim)        for b in self.blocks], dim=0)
        act_all = torch.cat([b.actions.reshape(-1, self.act_dim)    for b in self.blocks], dim=0)
        B = obs_all.shape[0]
        mb = max(B // self.mini_batches, 1)
        for _ in range(self.train_epochs):
            perm = torch.randperm(B, device=self.device)
            for s in range(0, B, mb):
                idx = perm[s:s + mb]
                yield obs_all[idx], act_all[idx], None, None  # rnn_init, masks

    def _recurrent_chunks(self):
        """Yield (obs[L*N], action[L*N], rnn_init[N,L_g,H], masks[L*N,1]).

        Vectorised, NON-OVERLAPPING segmentation: each (block, env)
        trajectory of length T is sliced into floor(T / L) contiguous
        chunks of length L.  rnn_init for chunk c is the GRU state that
        was fed into the env at time c*L (snapshot stored in
        block.rnn_states[c*L]).

        Per-minibatch chunk count is capped by
        ``max_chunks_per_minibatch`` to bound the BPTT graph size
        (memory ~ mb * L * obs_dim).
        """
        if not self.blocks:
            return
        L = self.data_chunk_length
        # Pre-stack into (n_chunks_total, L, *) tensors once per .train() call.
        obs_chunks_list, act_chunks_list = [], []
        mask_chunks_list, rnn_init_list  = [], []
        for blk in self.blocks:
            T = blk.T
            n_seg = T // L                  # non-overlapping segments per env
            if n_seg == 0:
                continue
            T_use = n_seg * L
            # (T_use, N, *) -> (n_seg, L, N, *) -> (n_seg*N, L, *)
            obs   = blk.obs    [:T_use].reshape(n_seg, L, blk.N, self.obs_dim)
            act   = blk.actions[:T_use].reshape(n_seg, L, blk.N, self.act_dim)
            mask  = blk.masks  [:T_use].reshape(n_seg, L, blk.N, 1)
            # rnn snapshot at the START of each segment (t = 0, L, 2L, ...)
            seg_starts = torch.arange(0, T_use, L, device=self.device)
            rnn_i = blk.rnn_states.index_select(0, seg_starts)   # (n_seg, N, L_g, H)

            # collapse (n_seg, N) into batch dim
            obs_chunks_list .append(obs .permute(0, 2, 1, 3).reshape(-1, L, self.obs_dim))
            act_chunks_list .append(act .permute(0, 2, 1, 3).reshape(-1, L, self.act_dim))
            mask_chunks_list.append(mask.permute(0, 2, 1, 3).reshape(-1, L, 1))
            rnn_init_list   .append(rnn_i.reshape(-1, self.L_g, self.H))

        if not obs_chunks_list:
            return

        obs_all  = torch.cat(obs_chunks_list,  dim=0)   # (C, L, obs_dim)
        act_all  = torch.cat(act_chunks_list,  dim=0)   # (C, L, act_dim)
        mask_all = torch.cat(mask_chunks_list, dim=0)   # (C, L, 1)
        rnn_all  = torch.cat(rnn_init_list,    dim=0)   # (C, L_g, H)

        C = obs_all.shape[0]
        mb = max(C // self.mini_batches, 1)
        mb = min(mb, self.max_chunks_per_minibatch)     # cap BPTT memory

        for _ in range(self.train_epochs):
            perm = torch.randperm(C, device=self.device)
            for s in range(0, C, mb):
                idx = perm[s:s + mb]
                if idx.numel() == 0:
                    continue
                obs_b  = obs_all .index_select(0, idx)        # (n, L, obs_dim)
                act_b  = act_all .index_select(0, idx)
                mask_b = mask_all.index_select(0, idx)
                rnn_b  = rnn_all .index_select(0, idx)        # (n, L_g, H)
                n = idx.numel()
                # (n, L, *) -> (L, n, *) -> (L*n, *) so adjacent rows along
                # batch dim are consecutive timesteps for the SAME chunk
                # (matches RNN layer expectation in ppo_actor flatten path).
                obs_b  = obs_b .transpose(0, 1).reshape(L * n, self.obs_dim)
                act_b  = act_b .transpose(0, 1).reshape(L * n, self.act_dim)
                mask_b = mask_b.transpose(0, 1).reshape(L * n, 1)
                yield obs_b, act_b, rnn_b, mask_b

    # ------------------------------------------------------------------ #
    def train(self):
        """One full pass of supervised training over the aggregated buffer."""
        if not self.blocks:
            return float('nan')

        gen = (self._recurrent_chunks if self.policy.use_recurrent_policy
               else self._feed_forward_batches)
        last_loss = 0.0
        last_grad_norm = 0.0
        n_updates = 0
        for obs_b, act_b, rnn_init, masks in gen():
            pred, _ = self.policy(obs_b, rnn_states=rnn_init, masks=masks,
                                  deterministic=True)
            loss = self.loss_fn(pred, act_b)
            self.optim.zero_grad()
            loss.backward()
            last_grad_norm = float(
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            )
            self.optim.step()
            last_loss = float(loss.detach())
            n_updates += 1
        self._last_grad_norm = last_grad_norm
        return last_loss if n_updates > 0 else float('nan')

    # ------------------------------------------------------------------ #
    def fit(self, n_iters, log_every=1):
        history = []
        # initialise tracking attrs in case first iter skips collect somehow
        self._last_done_rate     = 0.0
        self._last_bad_done_rate = 0.0
        self._last_exceed_rate   = 0.0
        self._last_grad_norm     = 0.0
        for k in range(n_iters):
            beta = max(self.beta_min, self.beta0 * (self.beta_decay ** k))
            t0 = time.time()
            mean_rew = self.collect(beta)
            t1 = time.time()
            loss = self.train()
            t2 = time.time()
            n_trans = sum(b.T * b.N for b in self.blocks)
            entry = dict(iter=k, beta=beta, loss=loss,
                         rollout_reward=mean_rew,
                         buf_transitions=n_trans,
                         done_rate=self._last_done_rate,
                         bad_done_rate=self._last_bad_done_rate,
                         exceed_rate=self._last_exceed_rate,
                         grad_norm=self._last_grad_norm,
                         t_collect=t1 - t0, t_train=t2 - t1)
            history.append(entry)

            # --- TensorBoard ---
            if self.writer is not None:
                step = k
                self.writer.add_scalar('train/loss',            loss,                        step)
                self.writer.add_scalar('train/grad_norm',       self._last_grad_norm,        step)
                self.writer.add_scalar('rollout/mean_reward',   mean_rew,                    step)
                self.writer.add_scalar('rollout/beta',          beta,                        step)
                self.writer.add_scalar('rollout/buf_transitions', n_trans,                   step)
                self.writer.add_scalar('rollout/done_rate',     self._last_done_rate,        step)
                self.writer.add_scalar('rollout/bad_done_rate', self._last_bad_done_rate,    step)
                self.writer.add_scalar('rollout/exceed_rate',   self._last_exceed_rate,      step)
                self.writer.add_scalar('time/collect_s',        t1 - t0,                    step)
                self.writer.add_scalar('time/train_s',          t2 - t1,                    step)
                self.writer.flush()

            if (k % log_every) == 0:
                print(f"[DAgger {k:3d}/{n_iters}] beta={beta:.3f}  "
                      f"buf={n_trans:>7d}  loss={loss:.4e}  "
                      f"grad_norm={self._last_grad_norm:.3f}  "
                      f"rollout_rew={mean_rew:+.3f}  "
                      f"bad_done={self._last_bad_done_rate:.3f}  "
                      f"(collect {t1 - t0:.1f}s, train {t2 - t1:.1f}s)")
            if self.ckpt_dir is not None:
                self.save(os.path.join(self.ckpt_dir, 'dagger_latest.pt'))
                self.save(os.path.join(self.ckpt_dir, f'dagger_iter{k:04d}.pt'))

            # Release cached blocks of this iter's autograd / temp tensors
            # so the next collect/train doesn't trip on fragmented memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.writer is not None:
            self.writer.close()
        return history

    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'optim':  self.optim.state_dict(),
            'args':   vars(self.policy.args),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        if 'optim' in ckpt:
            self.optim.load_state_dict(ckpt['optim'])
