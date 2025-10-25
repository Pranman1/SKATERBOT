from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torch
import torch.nn as nn

class Policy(GaussianMixin, Model):
    """Simple policy network."""
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    """Simple value network."""
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

PPO_CONFIG = PPO_DEFAULT_CONFIG.copy()
PPO_CONFIG["rollouts"] = 16
PPO_CONFIG["learning_epochs"] = 5
PPO_CONFIG["mini_batches"] = 4
PPO_CONFIG["discount_factor"] = 0.99
PPO_CONFIG["lambda"] = 0.95
PPO_CONFIG["learning_rate"] = 3e-4
PPO_CONFIG["learning_rate_scheduler"] = None
PPO_CONFIG["grad_norm_clip"] = 1.0
PPO_CONFIG["entropy_loss_scale"] = 0.0
PPO_CONFIG["value_loss_scale"] = 2.0
PPO_CONFIG["state_preprocessor"] = None
PPO_CONFIG["value_preprocessor"] = None

