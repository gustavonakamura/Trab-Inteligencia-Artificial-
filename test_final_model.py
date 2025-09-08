import argparse
import numpy as np
from game_env import FlappyEnv, Config

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def model_action(obs, pack):
    w = pack.item().get("w")
    b = pack.item().get("b")
    mean = pack.item().get("mean")
    std = pack.item().get("std")
    x = (obs.reshape(1,-1) - mean) / (std + 1e-6)
    p = sigmoid(x @ w + b)[0,0]
    return 1 if p >= 0.5 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--weights", type=str, default="weights_final.npy")
    args = ap.parse_args()

    print("🎮 TESTE FINAL DO MODELO DE IA TREINADO")
    print("=" * 40)
    
    pack = np.load(args.weights, allow_pickle=True)
    env = FlappyEnv(Config(pipe_gap=400, seed=888))  # Mesma config dos dados
    
    scores = []
    returns = []
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        score = 0
        steps = 0
        
        while True:
            action = model_action(obs, pack)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if 'score' in info:
                score = info['score']
                
            if done or steps > 500:
                break
        
        scores.append(score)
        returns.append(total_reward)
        
        emoji = "🏆" if score >= 2 else "🎯" if score >= 1 else "❌"
        print(f"Ep {ep+1:2d}: Score={score:2d}, Return={total_reward:5.1f}, Steps={steps:3d} {emoji}")
    
    print()
    print("📊 ESTATÍSTICAS FINAIS:")
    print(f"   Score médio: {np.mean(scores):.2f}")
    print(f"   Melhor score: {max(scores)}")
    print(f"   Taxa de sucesso: {sum(1 for s in scores if s > 0)/len(scores)*100:.1f}%")
    print(f"   Return médio: {np.mean(returns):.1f}")
    
    if max(scores) >= 2:
        print("\n🎊 FANTÁSTICO! IA consegue passar múltiplos canos!")
    elif max(scores) >= 1:
        print("\n👏 SUCESSO! IA consegue navegar pelos obstáculos!")
    else:
        print("\n🔧 IA ainda está aprendendo...")

if __name__ == "__main__":
    main()
