import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space
        self.use_cost_constraints = (
            bool(getattr(args, 'use_cost_constraints', False))
            or getattr(args, 'algorithm_name', '') == 'cpo'
        )

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)
        self.cost_critic = (
            PPOCritic(args, self.obs_space, self.device)
            if self.use_cost_constraints else None
        )

        params = [
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
        ]
        if self.cost_critic is not None:
            params.append({'params': self.cost_critic.parameters()})
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def get_cost_values(self, obs, rnn_states_cost_critic, masks):
        if self.cost_critic is None:
            raise RuntimeError('get_cost_values called while cost constraints are disabled')
        values, rnn_states_cost_critic = self.cost_critic(
            obs, rnn_states_cost_critic, masks
        )
        return values, rnn_states_cost_critic

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def evaluate_cost_values(self, obs, rnn_states_cost_critic, masks):
        if self.cost_critic is None:
            raise RuntimeError('evaluate_cost_values called while cost constraints are disabled')
        values, _ = self.cost_critic(obs, rnn_states_cost_critic, masks)
        return values

    def predict_safety(self, obs, rnn_states_actor, masks):
        return self.actor.predict_safety(obs, rnn_states_actor, masks)

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        if self.cost_critic is not None:
            self.cost_critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        if self.cost_critic is not None:
            self.cost_critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
