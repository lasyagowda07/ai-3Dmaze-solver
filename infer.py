import numpy as np
import torch
from env import Maze3DEnv
from model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_games(n=20):
    env = Maze3DEnv(sx=11, sy=11, sz=5, max_steps=400, seed=None)

    model = DQN(input_dim=18, hidden=256, output_dim=6).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/maze3d_dqn.pt", map_location=DEVICE))
    model.eval()

    rewards = []
    for i in range(n):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                st = torch.tensor(s, device=DEVICE).unsqueeze(0)
                a = int(torch.argmax(model(st), dim=1).item())
            res = env.step(a)
            s, r, done = res.obs, res.reward, res.done
            total += r

        rewards.append(total)
        print(f"game {i+1}/{n} reward={total:.2f}")

    print(f"avg reward: {np.mean(rewards):.2f} | max: {np.max(rewards):.2f}")


if __name__ == "__main__":
    run_games(20)