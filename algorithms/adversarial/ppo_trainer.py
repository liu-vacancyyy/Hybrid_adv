import torch
import torch.nn as nn
from typing import List, Union

from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm
from .ppo_policy import AdversarialPPOPolicy


class AdversarialPPOTrainer:
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        self.adv_lipschitz_coef = float(getattr(args, "adv_lipschitz_coef", 0.0))

    def ppo_update(self, policy: AdversarialPPOPolicy, sample):
        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            obs_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()
        lipschitz_product = policy.actor.lipschitz_product()
        lipschitz_penalty = self.adv_lipschitz_coef * lipschitz_product
        loss = (
            policy_loss
            + value_loss * self.value_loss_coef
            + policy_entropy_loss * self.entropy_coef
            + lipschitz_penalty
        )

        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                policy.actor.parameters(), self.max_grad_norm
            ).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(
                policy.critic.parameters(), self.max_grad_norm
            ).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return (
            policy_loss,
            value_loss,
            policy_entropy_loss,
            lipschitz_product,
            lipschitz_penalty,
            ratio,
            actor_grad_norm,
            critic_grad_norm,
        )

    def train(self, policy: AdversarialPPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "policy_entropy_loss": 0,
            "adv_lipschitz_product": 0,
            "adv_lipschitz_penalty": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0,
            "ratio_abs_dev": 0,
            "ratio_clip_frac": 0,
            "advantage_mean": 0,
            "advantage_std": 0,
        }

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(
                    buffer, self.num_mini_batch, self.data_chunk_length
                )
            else:
                data_generator = ReplayBuffer.feed_forward_generator(buffer, self.num_mini_batch)

            for sample in data_generator:
                advantages_for_log = check(sample[4]).to(**self.tpdv)
                policy_loss, value_loss, policy_entropy_loss, lipschitz_product, \
                    lipschitz_penalty, ratio, actor_grad_norm, critic_grad_norm = self.ppo_update(
                        policy, sample
                    )

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["policy_entropy_loss"] += policy_entropy_loss.item()
                train_info["adv_lipschitz_product"] += lipschitz_product.item()
                train_info["adv_lipschitz_penalty"] += lipschitz_penalty.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += ratio.mean().item()
                train_info["ratio_abs_dev"] += torch.abs(ratio - 1.0).mean().item()
                train_info["ratio_clip_frac"] += (
                    torch.abs(ratio - 1.0) > self.clip_param
                ).float().mean().item()
                train_info["advantage_mean"] += advantages_for_log.mean().item()
                train_info["advantage_std"] += advantages_for_log.std(unbiased=False).item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for key in train_info:
            train_info[key] /= num_updates
        return train_info
