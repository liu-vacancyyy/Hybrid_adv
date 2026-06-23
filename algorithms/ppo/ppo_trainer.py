import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = float(getattr(args, 'target_kl', 0.0))
        self.max_log_ratio = float(getattr(args, 'max_log_ratio', 20.0))
        self.use_safety_aux = bool(getattr(args, 'use_safety_aux', False))
        self.safety_aux_loss_coef = float(getattr(args, 'safety_aux_loss_coef', 0.1))
        self.safety_aux_pos_weight = float(getattr(args, 'safety_aux_pos_weight', 5.0))
        self.use_cost_constraints = (
            bool(getattr(args, 'use_cost_constraints', False))
            or getattr(args, 'algorithm_name', '') == 'cpo'
        )
        self.cost_limit = float(getattr(args, 'cost_limit', 0.02))
        self.cost_lagrange = max(0.0, float(getattr(args, 'cost_lagrange_init', 1.0)))
        self.cost_lagrange_lr = float(getattr(args, 'cost_lagrange_lr', 0.05))
        self.cost_value_loss_coef = float(getattr(args, 'cost_value_loss_coef', 0.25))
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

    def _zero_update(self, reason=1.0):
        zero = torch.zeros((), device=self.device)
        return (
            zero, zero, zero, zero, zero, zero, zero,
            0.0, 0.0, 0.0, reason, 0.0, 0.0, 0.0,
        )

    def _finite_tensors(self, *tensors):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

    def _finite_parameters(self, module):
        return all(torch.isfinite(p).all().item() for p in module.parameters())

    def ppo_update(self, policy: PPOPolicy, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample[:9]
        cursor = 9

        cost_advantages_batch = None
        cost_returns_batch = None
        cost_value_preds_batch = None
        rnn_states_cost_critic_batch = None
        if self.use_cost_constraints:
            cost_advantages_batch, cost_returns_batch, cost_value_preds_batch, \
                rnn_states_cost_critic_batch = sample[cursor:cursor + 4]
            cursor += 4

        safety_targets_batch = None
        safety_valid_batch = None
        if self.use_safety_aux:
            safety_targets_batch, safety_valid_batch = sample[cursor:cursor + 2]

        obs_check = check(obs_batch).to(**self.tpdv)
        actions_check = check(actions_batch).to(**self.tpdv)
        masks_check = check(masks_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        if not self._finite_tensors(
                obs_check, actions_check, masks_check, old_action_log_probs_batch,
                advantages_batch, returns_batch, value_preds_batch):
            return self._zero_update()

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            obs_check,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_check,
            masks_check,
        )
        if not self._finite_tensors(values, action_log_probs, dist_entropy):
            return self._zero_update()

        log_ratio = action_log_probs - old_action_log_probs_batch
        safe_log_ratio = torch.clamp(log_ratio, -self.max_log_ratio, self.max_log_ratio)
        approx_kl = (
            (torch.exp(safe_log_ratio.detach()) - 1.0) - safe_log_ratio.detach()
        ).mean()
        if self.target_kl > 0.0 and approx_kl.detach().item() > 1.5 * self.target_kl:
            ratio_for_log = torch.exp(safe_log_ratio.detach()).mean()
            return (
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
                ratio_for_log,
                0.0,
                0.0,
                approx_kl.detach().item(),
                1.0,
                0.0,
                0.0,
                0.0,
            )

        # Obtain the loss function
        ratio = torch.exp(safe_log_ratio)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()
        cost_policy_loss = torch.zeros((), device=self.device)
        cost_value_loss = torch.zeros((), device=self.device)
        if self.use_cost_constraints and cost_advantages_batch is not None:
            cost_advantages_batch = check(cost_advantages_batch).to(**self.tpdv)
            cost_returns_batch = check(cost_returns_batch).to(**self.tpdv)
            cost_value_preds_batch = check(cost_value_preds_batch).to(**self.tpdv)
            cost_values = policy.evaluate_cost_values(
                obs_check,
                rnn_states_cost_critic_batch,
                masks_check,
            )
            if not self._finite_tensors(
                    cost_advantages_batch, cost_returns_batch, cost_value_preds_batch,
                    cost_values):
                return self._zero_update()

            cost_surr1 = ratio * cost_advantages_batch
            cost_surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * cost_advantages_batch
            )
            # Conservative clipped surrogate: minimize the larger estimated cost.
            cost_policy_loss = torch.max(cost_surr1, cost_surr2).mean()

            if self.use_clipped_value_loss:
                cost_value_pred_clipped = cost_value_preds_batch + (
                    cost_values - cost_value_preds_batch
                ).clamp(-self.clip_param, self.clip_param)
                cost_losses = (cost_values - cost_returns_batch).pow(2)
                cost_losses_clipped = (cost_value_pred_clipped - cost_returns_batch).pow(2)
                cost_value_loss = 0.5 * torch.max(cost_losses, cost_losses_clipped)
            else:
                cost_value_loss = 0.5 * (cost_returns_batch - cost_values).pow(2)
            cost_value_loss = cost_value_loss.mean()

        safety_aux_loss = torch.zeros((), device=self.device)
        safety_aux_acc = 0.0
        safety_aux_pos_rate = 0.0
        safety_aux_valid_rate = 0.0
        if self.use_safety_aux and safety_targets_batch is not None:
            safety_targets_batch = check(safety_targets_batch).to(**self.tpdv)
            safety_valid_batch = check(safety_valid_batch).to(**self.tpdv)
            safety_logits = policy.predict_safety(
                obs_check,
                rnn_states_actor_batch,
                masks_check,
            )
            if not self._finite_tensors(safety_logits, safety_targets_batch, safety_valid_batch):
                return self._zero_update()
            pos_weight = torch.tensor(self.safety_aux_pos_weight, device=self.device)
            raw_bce = nn.functional.binary_cross_entropy_with_logits(
                safety_logits,
                safety_targets_batch,
                pos_weight=pos_weight,
                reduction='none',
            )
            valid_sum = torch.clamp(safety_valid_batch.sum(), min=1.0)
            safety_aux_loss = (raw_bce * safety_valid_batch).sum() / valid_sum
            with torch.no_grad():
                pred = (torch.sigmoid(safety_logits) >= 0.5).float()
                safety_aux_acc = (
                    ((pred == safety_targets_batch).float() * safety_valid_batch).sum()
                    / valid_sum
                ).item()
                safety_aux_pos_rate = (
                    (safety_targets_batch * safety_valid_batch).sum() / valid_sum
                ).item()
                safety_aux_valid_rate = safety_valid_batch.mean().item()

        loss = (
            policy_loss
            + value_loss * self.value_loss_coef
            + policy_entropy_loss * self.entropy_coef
            + safety_aux_loss * self.safety_aux_loss_coef
            + cost_policy_loss * self.cost_lagrange
            + cost_value_loss * self.cost_value_loss_coef
        )
        if not torch.isfinite(loss).item():
            return self._zero_update()

        # Optimize the loss function
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor_grads = [p.grad for p in policy.actor.parameters() if p.grad is not None]
        critic_params = list(policy.critic.parameters())
        if getattr(policy, 'cost_critic', None) is not None:
            critic_params += list(policy.cost_critic.parameters())
        critic_grads = [p.grad for p in critic_params if p.grad is not None]
        if not self._finite_tensors(*actor_grads, *critic_grads):
            policy.optimizer.zero_grad(set_to_none=True)
            return self._zero_update()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(critic_params)
        if not torch.isfinite(torch.tensor([actor_grad_norm, critic_grad_norm], device=self.device)).all().item():
            policy.optimizer.zero_grad(set_to_none=True)
            return self._zero_update()
        policy.optimizer.step()
        finite_cost_critic = (
            getattr(policy, 'cost_critic', None) is None
            or self._finite_parameters(policy.cost_critic)
        )
        if not (
            self._finite_parameters(policy.actor)
            and self._finite_parameters(policy.critic)
            and finite_cost_critic
        ):
            raise FloatingPointError("PPO optimizer produced non-finite policy parameters")

        return (
            policy_loss,
            value_loss,
            policy_entropy_loss,
            safety_aux_loss.detach(),
            cost_policy_loss.detach(),
            cost_value_loss.detach(),
            ratio.detach().mean(),
            actor_grad_norm,
            critic_grad_norm,
            approx_kl.detach().item(),
            0.0,
            safety_aux_acc,
            safety_aux_pos_rate,
            safety_aux_valid_rate,
        )

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['safety_aux_loss'] = 0
        train_info['cost_policy_loss'] = 0
        train_info['cost_value_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['approx_kl'] = 0
        train_info['skipped_updates'] = 0
        train_info['safety_aux_acc'] = 0
        train_info['safety_aux_pos_rate'] = 0
        train_info['safety_aux_valid_rate'] = 0
        rollout_episode_cost = self._rollout_episode_cost(buffer)
        train_info['constraint/episode_cost'] = rollout_episode_cost
        train_info['constraint/cost_limit'] = self.cost_limit
        train_info['constraint/lagrange_before_update'] = self.cost_lagrange

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = ReplayBuffer.feed_forward_generator(buffer, self.num_mini_batch)

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, safety_aux_loss, \
                    cost_policy_loss, cost_value_loss, ratio, \
                    actor_grad_norm, critic_grad_norm, approx_kl, skipped, \
                    safety_aux_acc, safety_aux_pos_rate, safety_aux_valid_rate = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['safety_aux_loss'] += safety_aux_loss.item()
                train_info['cost_policy_loss'] += cost_policy_loss.item()
                train_info['cost_value_loss'] += cost_value_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.item() if torch.is_tensor(ratio) else ratio
                train_info['approx_kl'] += approx_kl
                train_info['skipped_updates'] += skipped
                train_info['safety_aux_acc'] += safety_aux_acc
                train_info['safety_aux_pos_rate'] += safety_aux_pos_rate
                train_info['safety_aux_valid_rate'] += safety_aux_valid_rate

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if k.startswith('constraint/'):
                continue
            train_info[k] /= num_updates

        if self.use_cost_constraints:
            self.cost_lagrange = max(
                0.0,
                self.cost_lagrange
                + self.cost_lagrange_lr * (rollout_episode_cost - self.cost_limit),
            )
        train_info['constraint/lagrange_after_update'] = self.cost_lagrange

        return train_info

    def _rollout_episode_cost(self, buffer):
        buffers = [buffer] if isinstance(buffer, ReplayBuffer) else buffer
        cost_sum = 0.0
        completed = 0.0
        for buf in buffers:
            if not getattr(buf, 'use_cost_constraints', False):
                continue
            cost_sum += float(buf.costs.sum())
            completed += float((buf.masks[1:] < 0.5).sum())
            completed += float((buf.bad_masks[1:] < 0.5).sum())
        if completed <= 0.0:
            return 0.0
        return cost_sum / completed
