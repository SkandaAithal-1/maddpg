import numpy as np
import torch
from agent import Agent
from typing import List, Optional
from networks import CNNHead 
import torch.nn.functional as F
from utils import RunningMeanStd

from gradient_estimators import GradientEstimator

class MADDPG:
    def __init__(
        self,
        env,
        critic_lr : float,
        actor_lr : float,
        cnn_lr : float,
        gradient_clip : float,
        hidden_dim_width : int,
        gamma : float,
        soft_update_size : float,
        policy_regulariser : float,
        gradient_estimator : GradientEstimator,
        standardise_rewards : bool,
        pretrained_agents : Optional[ List[Agent] ] = None,
    ):
        self.n_agents = env.n_agents
        self.gamma = gamma
        obs_dims = [obs.shape[0] for obs in env.observation_space]
        act_dims = [act.shape[1] for act in env.action_space]
        self.agents = [
            Agent(
                agent_idx=ii,
                obs_dims=obs_dims,
                act_dims=act_dims,
                # TODO: Consider changing this to **config
                hidden_dim_width=hidden_dim_width,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                gradient_clip=gradient_clip,
                soft_update_size=soft_update_size,
                policy_regulariser=policy_regulariser,
                gradient_estimator=gradient_estimator,
            )
            for ii in range(self.n_agents)
        ] if pretrained_agents is None else pretrained_agents

        self.embedder = CNNHead()
        self.embed_optim = torch.optim.Adam(self.embedder.parameters(), lr=cnn_lr, eps=0.001)

        self.return_std = RunningMeanStd(shape=(self.n_agents,)) if standardise_rewards else None
        self.gradient_estimator = gradient_estimator # Keep reference to GE object

    def acts(self, obs: List):
        obs, goals, states = obs
        new_obs = []
        for ii in range(self.n_agents):
            tempObs = self.embedder(torch.Tensor(obs[ii]).unsqueeze(0))
            tempObs = torch.cat((torch.Tensor(tempObs), torch.Tensor(goals[ii]).unsqueeze(0), torch.Tensor(states[ii]).unsqueeze(0)), dim=1)
            new_obs.append(tempObs)

        actions = [self.agents[ii].act_behaviour(torch.Tensor(new_obs[ii]).squeeze()) for ii in range(self.n_agents)]
        return actions

    def update(self, sample):
        # sample['obs'] : agent batch obs
        for ii in range(self.n_agents):
            sample['obs'][ii] = self.embedder(torch.Tensor(sample['obs'][ii]))
            # sample['obs'][ii] = torch.cat((sample['obs'][ii], sample['goals'][ii], sample['states'][ii]), dim=1)
            sample['nobs'][ii] = self.embedder(torch.Tensor(sample['nobs'][ii]))
            # sample['nobs'][ii] = torch.cat((sample['nobs'][ii], sample['goals'][ii], sample['states'][ii]), dim=1)

        batched_obs = torch.concat(sample['obs'], axis=1).detach()
        batched_nobs = torch.concat(sample['nobs'], axis=1).detach()

        # ********
        # TODO: This is all a bit cumbersome--could be cleaner?

        target_actions = [
            self.agents[ii].act_target(torch.concat((sample['nobs'][ii], sample['ngoals'][ii], sample['nstates'][ii]), axis=1))
            for ii in range(self.n_agents)
        ]

        target_actions_one_hot = [
            F.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        sampled_actions_one_hot = [
            F.one_hot(sample['acts'][ii].to(torch.int64), num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        # ********
        # Standardise rewards if requested
        rewards = sample['rwds']
        if self.return_std is not None:
            self.return_std.update(rewards)
            rewards = ((rewards.T - self.return_std.mean) / torch.sqrt(self.return_std.var)).T
        # ********

        info = {}
        batched_goals = torch.concat(sample['goals'], axis=1)
        batched_ngoals = torch.concat(sample['ngoals'], axis=1)
        batched_states = torch.concat(sample['states'], axis=1)
        batched_nstates = torch.concat(sample['nstates'], axis=1)

        self.embed_optim.zero_grad()

        for ii, agent in enumerate(self.agents):


            batched_obs = torch.cat((sample['obs'][ii], sample['goals'][ii], sample['states'][ii], batched_goals, batched_states), axis=1)
            batched_nobs = torch.cat((sample['nobs'][ii], sample['ngoals'][ii], sample['nstates'][ii], batched_ngoals, batched_nstates), axis=1)

            critic_loss, critic_grad_norm = agent.update_critic(
                all_obs=batched_obs.detach(),
                all_nobs=batched_nobs.detach(),
                target_actions_per_agent=target_actions_one_hot,
                sampled_actions_per_agent=sampled_actions_one_hot,
                rewards=rewards[ii].unsqueeze(dim=1),
                dones=sample['dones'][ii].unsqueeze(dim=1),
                gamma=self.gamma,
            )

            actor_loss, actor_grad_norm = agent.update_actor(
                all_obs=batched_obs,
                agent_obs=torch.concat((sample['obs'][ii], sample['goals'][ii], sample['states'][ii]), axis=1),
                sampled_actions=sampled_actions_one_hot,
            )

            info[f"Actor_loss_agent_{ii}"] = actor_loss
            info[f"Critic_loss_agent_{ii}"] = critic_loss
            info[f"Actor_grad_norm_agent_{ii}"] = actor_grad_norm
            info[f"Critic_grad_norm_agent_{ii}"] = critic_grad_norm

        self.embed_optim.step()

        for agent in self.agents:
            agent.soft_update()

        # self.gradient_estimator.update_state() # Update GE state, if necessary

        return info 
