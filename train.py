import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

from env import Maze3DEnv
from model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size=256):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns).astype(np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def epsilon_by_episode(ep, eps_start=1.0, eps_end=0.05, decay=1200):
    # smooth exponential decay
    return eps_end + (eps_start - eps_end) * np.exp(-ep / decay)


def main():
    os.makedirs("checkpoints", exist_ok=True)

    env = Maze3DEnv(sx=11, sy=11, sz=5, max_steps=400, seed=None)
    input_dim = 18
    n_actions = 6

    q = DQN(input_dim=input_dim, hidden=256, output_dim=n_actions).to(DEVICE)
    tgt = DQN(input_dim=input_dim, hidden=256, output_dim=n_actions).to(DEVICE)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    rb = ReplayBuffer(capacity=80_000)

    gamma = 0.99
    batch_size = 256
    warmup = 4_000
    target_update = 700  # steps

    max_episodes = 2500
    best_mean = -1e9
    scores = deque(maxlen=50)

    global_step = 0

    for ep in range(1, max_episodes + 1):
        s = env.reset()
        done = False
        ep_reward = 0.0

        eps = float(epsilon_by_episode(ep))

        while not done:
            global_step += 1

            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    st = torch.tensor(s, device=DEVICE).unsqueeze(0)
                    a = int(torch.argmax(q(st), dim=1).item())

            res = env.step(a)
            ns, r, done = res.obs, res.reward, res.done
            ep_reward += r

            rb.push(s, a, r, ns, done)
            s = ns

            if len(rb) >= warmup:
                bs, ba, br, bns, bd = rb.sample(batch_size)

                bs_t = torch.tensor(bs, device=DEVICE)
                ba_t = torch.tensor(ba, device=DEVICE).unsqueeze(1)
                br_t = torch.tensor(br, device=DEVICE).unsqueeze(1)
                bns_t = torch.tensor(bns, device=DEVICE)
                bd_t = torch.tensor(bd, device=DEVICE).unsqueeze(1)

                qsa = q(bs_t).gather(1, ba_t)

                with torch.no_grad():
                    max_next = tgt(bns_t).max(dim=1, keepdim=True).values
                    target = br_t + gamma * (1.0 - bd_t) * max_next

                loss = F.smooth_l1_loss(qsa, target)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if global_step % target_update == 0:
                    tgt.load_state_dict(q.state_dict())

        scores.append(ep_reward)
        mean50 = float(np.mean(scores))

        if mean50 > best_mean and ep >= 100:
            best_mean = mean50
            torch.save(q.state_dict(), "checkpoints/maze3d_dqn.pt")
            print(f"saved new best model | mean50={best_mean:.2f}")

        if ep % 25 == 0:
            print(f"ep {ep:4d} | ep_reward={ep_reward:7.2f} | mean50={mean50:7.2f} | eps={eps:.2f}")

    print("Training done.")


if __name__ == "__main__":
    main()