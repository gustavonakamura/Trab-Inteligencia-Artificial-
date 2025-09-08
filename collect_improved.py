import argparse
import csv
from game_env import FlappyEnv, Config
from expert_policy import expert_action
import numpy as np
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=str, default="data_improved.csv")
    ap.add_argument("--gap", type=int, default=250)
    ap.add_argument("--epsilon", type=float, default=0.05)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42)
    env = FlappyEnv(Config(pipe_gap=args.gap, seed=42))

    data = []
    total_score = 0
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_score = 0
        
        while not done:
            a = expert_action(obs)
            if np.random.rand() < args.epsilon:
                a = 1 - a  # inverte com probabilidade epsilon
            
            data.append([obs[0], obs[1], obs[2], obs[3], a])
            obs, _, done, info = env.step(a)
            if 'score' in info:
                ep_score = info['score']
        
        total_score += ep_score
        print(f"[coleta] ep {ep+1}/{args.episodes} score={ep_score}")
    
    print(f"Score médio: {total_score/args.episodes:.2f}")
    
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["y_norm","vy_norm","dist_norm","delta_gap_norm","action"])
        wr.writerows(data)
    
    print(f"dataset salvo em {args.out}")

if __name__ == "__main__":
    main()
