import argparse
import numpy as np
from game_env import FlappyEnv, Config

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
    ap.add_argument("--weights", type=str, default="weights_improved.npy")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--gap", type=int, default=250)
    args = ap.parse_args()

    pack = np.load(args.weights, allow_pickle=True)
    env = FlappyEnv(Config(pipe_gap=args.gap))

    total_score = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_score = 0
        
        while not done:
            a = infer_action(obs, pack)
            obs, reward, done, info = env.step(a)
            ep_return += reward
            if 'score' in info:
                ep_score = info['score']
        
        total_score += ep_score
        print(f"Ep {ep+1}: score={ep_score} return={ep_return:.2f}")
    
    print(f"Score médio: {total_score/args.episodes:.2f}")

if __name__ == "__main__":
    main()
