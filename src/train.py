"""
Train G1 robot to balance on skateboard.

Usage:
    python train.py --headless                  # Train with 4 envs headless
    python train.py --num_envs 4                # Train with visualization
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 to balance on skateboard")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from envs.robot_env import G1BalanceEnvCfg
from models.ppo import Policy, Value, PPO_CONFIG

def main():
    # Create environment
    env_cfg = G1BalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap environment for skrl
    env = wrap_env(env)
    
    device = env.device
    
    # Create models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)
    
    # Create memory
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
    
    # Create agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=PPO_CONFIG,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # Configure trainer
    cfg_trainer = {"timesteps": 100, "headless": args_cli.headless}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    # Train
    trainer.train()
    
    # Save model
    trainer.agents.save("../g1_balance_policy.pt")
    print("\nTraining complete! Model saved to ../g1_balance_policy.pt")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

