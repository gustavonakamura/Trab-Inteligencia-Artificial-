import argparse
import csv
from game_env import FlappyEnv, Config
from expert_policy import expert_action
import numpy as np
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", type=str, default="data.csv")
    ap.add_argument("--gap", type=int, default=150, help="pipe gap (dificuldade)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="prob. de ação aleatória (ruído)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render_every", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    env = FlappyEnv(Config(pipe_gap=args.gap, seed=args.seed))

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["y_norm","vy_norm","dist_norm","delta_gap_norm","action"])
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                a = expert_action(obs)
                if np.random.rand() < args.epsilon:
                    a = np.random.randint(0, 2)
                wr.writerow([*obs.tolist(), a])
                obs, r, done, info = env.step(a)
                if args.render_every and (ep % args.render_every == 0):
                    try: env.render()
                    except SystemExit: return
            print(f"[coleta] ep {ep+1}/{args.episodes} score={info.get('score',0)}")
    print(f"dataset salvo em {args.out}")

if __name__ == "__main__":
    main()
