import argparse
from game_env import FlappyEnv, Config
from expert_policy import expert_action

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--gap", type=int, default=150)
    args = ap.parse_args()

    config = Config(pipe_gap=250)  # Gap muito maior para facilitar
    env = FlappyEnv(config)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        score = 0
        done = False
        
        while not done:
            action = expert_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if 'score' in info:
                score = info['score']
        
        print(f"Ep {ep}: score={score} return={total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
