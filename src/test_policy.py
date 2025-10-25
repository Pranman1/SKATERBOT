"""
Test trained G1 balance policy.

Usage:
    python test_policy.py --checkpoint ../g1_balance_policy.pt
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1 balance policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default="./g1_balance_policy.pt", help="Path to checkpoint")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from skrl.envs.wrappers.torch import wrap_env

from envs.robot_env import G1BalanceEnvCfg
from models.ppo import Policy, Value

def main():
    # Create environment
    env_cfg = G1BalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = wrap_env(env)
    
    device = env.device
    
    # Create and load policy
    policy = Policy(env.observation_space, env.action_space, device)
    policy.load(args_cli.checkpoint)
    policy.eval()
    
    # Run policy
    print("Running trained policy...")
    obs, _ = env.reset()
    
    for _ in range(1000):
        with torch.no_grad():
            actions, _, _ = policy.compute({"states": obs}, role="policy")
        obs, rewards, dones, truncated, info = env.step(actions)
    
    print("Test complete!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

