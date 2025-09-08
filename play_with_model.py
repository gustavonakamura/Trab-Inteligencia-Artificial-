import argparse
import numpy as np
from game_env import FlappyEnv, Config, pygame

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def infer_action(obs, pack):
    w = pack.item().get("w")
    b = pack.item().get("b")
    mean = pack.item().get("mean")
    std = pack.item().get("std")
    x = (obs.reshape(1,-1) - mean) / (std + 1e-6)
    p = sigmoid(x @ w + b)[0,0]
    return 1 if p >= 0.5 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="weights.npy")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    pack = np.load(args.weights, allow_pickle=True)
    env = FlappyEnv(Config(pipe_gap=250))

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = infer_action(obs, pack)
            obs, r, done, info = env.step(a)
            total += r
            try:
                env.render()
            except SystemExit:
                return
        print(f"Ep {ep+1}: score={info.get('score')} return={total:.2f}")

if __name__ == "__main__":
    main()
