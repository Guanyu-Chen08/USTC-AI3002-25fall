from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse
import math

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()
num_episodes = args.num_episodes
checkpoint = args.checkpoint


class BoardTransformer(nn.Module):
    def __init__(self, board_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.board_size = board_size
        self.seq_len = board_size * board_size
        
        self.token_embedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        B, N, _ = x.shape
        x_flat = x.view(B, -1) 
        
        tokens = self.token_embedding(x_flat) 
        tokens = tokens + self.pos_embedding
        
        features = self.transformer_encoder(tokens) 
        return features


class Actor(nn.Module):
    """
    The actor uses a Transformer architecture to generate policies.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        self.d_model = 128
        
        
        self.backbone = BoardTransformer(
            board_size=board_size, 
            d_model=self.d_model,
            nhead=4,
            num_layers=3
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
      
        if len(x.shape) == 2:
            x_tensor = torch.tensor(x).to(device).long().unsqueeze(0)
        else:
            x_tensor = torch.tensor(x).to(device).long()
            if x_tensor.dim() == 4:
                x_tensor = x_tensor.squeeze(1) 

        features = self.backbone(x_tensor)
        
        logits = self.policy_head(features).squeeze(-1)
        
        flat_inputs = x_tensor.view(logits.size(0), -1)
        mask = (flat_inputs == 0) 
        
        masked_logits = torch.where(mask, logits, torch.tensor(-1e9).to(device))
        
        output = torch.softmax(masked_logits, dim=-1) 
        
        return output


class Critic(nn.Module):
    """
    The critic uses a Transformer architecture to generate Q-values.
    It predicts a Q-value for EVERY position on the board simultaneously, 
    and then we select the ones corresponding to the actions taken.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        
        self.d_model = 128

        self.backbone = BoardTransformer(
            board_size=board_size, 
            d_model=self.d_model,
            nhead=4,
            num_layers=3
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        indices = torch.tensor([_position_to_index(self.board_size, r, c) for r, c in action]).to(device)
        
        if len(x.shape) == 2:
            x_tensor = torch.tensor(x).to(device).long().unsqueeze(0)
        else:
            x_tensor = torch.tensor(x).to(device).long()
            if x_tensor.dim() == 4:
                x_tensor = x_tensor.squeeze(1)

        features = self.backbone(x_tensor)
        
        all_qs = self.value_head(features).squeeze(-1)
        output = all_qs[torch.arange(len(indices)), indices]

        return output


class GobangModel(nn.Module):
    """
    The GobangModel class integrates the Actor and Critic classes.
    """

    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size

        self.actor = Actor(board_size=board_size)
        self.critic = Critic(board_size=board_size)
        
        self.to(device)

    def forward(self, x, action):
        """
        Return the policy vector π(s) and Q-values Q(s, a).
        """
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, entropy_coef=0.05):
        """
        :param entropy_coef: Entropy regularization coefficient
        """
        eps = 1e-10
        
        targets = rewards + gamma * next_qs.detach()
        critic_loss = nn.MSELoss()(targets, qs)

        indices = torch.tensor([_position_to_index(self.board_size, x, y) for x, y in actions]).to(device)
        aimed_policy = policy[torch.arange(len(indices)), indices]
        
        base_actor_loss = -torch.mean(torch.log(aimed_policy + eps) * qs.clone().detach())
        
        dist_entropy = -torch.sum(policy * torch.log(policy + eps), dim=1).mean()
        actor_loss = base_actor_loss - (entropy_coef * dist_entropy)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) 
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic.optimizer.step()
        
        return actor_loss, critic_loss


if __name__ == "__main__":
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "num_episodes": num_episodes,
                    "checkpoint": checkpoint,
                    "board_size": 12,
                    "bound": 5,
                    "model_type": "Transformer"
                }
            )
            print("Wandb initialized successfully.")
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")
            print("Continuing without wandb...")
    
    agent = GobangModel(board_size=12, bound=5).to(device)
    
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint)
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass