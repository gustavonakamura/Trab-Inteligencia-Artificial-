#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game_env import FlappyEnv, Config
from expert_policy import expert_action

def test_expert_policy():
    """Testa a política expert isoladamente para diagnosticar problemas"""
    print("🔍 Testando política expert...")
    
    # Configuração com gap ainda maior para debug
    config = Config(pipe_gap=500, seed=42)
    env = FlappyEnv(config)
    
    obs, _ = env.reset()
    print(f"📊 Observação inicial: {obs}")
    
    total_steps = 0
    episodes = 0
    max_episodes = 5
    
    for episode in range(max_episodes):
        print(f"\n🚀 Episódio {episode + 1}")
        obs, _ = env.reset()
        steps = 0
        score = 0
        
        for step in range(1000):  # Máximo 1000 steps por episódio
            y_norm, vy_norm, dist_norm, delta_gap_norm = obs
            
            # Mostra dados detalhados a cada 50 steps
            if step % 50 == 0:
                print(f"Step {step:3d}: y={y_norm:.3f}, vy={vy_norm:.3f}, dist={dist_norm:.3f}, delta={delta_gap_norm:.3f}")
            
            # Verifica limites críticos
            if y_norm < 0.0 or y_norm > 1.0:
                print(f"⚠️  POSIÇÃO CRÍTICA! y_norm = {y_norm:.3f}")
            
            action = expert_action(obs)
            
            # Mostra ações de pulo
            if action == 1 and step % 10 == 0:
                print(f"   🚀 PULO! (y={y_norm:.3f}, vy={vy_norm:.3f})")
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if 'score' in info:
                if info['score'] > score:
                    score = info['score']
                    print(f"   🎉 PASSOU CANO! Score: {score}")
            
            if done:
                print(f"   💥 Episódio terminou em {steps} steps, Score: {score}")
                break
        
        episodes += 1
        total_steps += steps
    
    avg_steps = total_steps / episodes
    print(f"\n📈 Resumo após {episodes} episódios:")
    print(f"   Média de steps: {avg_steps:.1f}")
    print(f"   Total de steps: {total_steps}")

def debug_environment():
    """Testa o ambiente para ver se há problemas de normalização"""
    print("\n🔧 Testando ambiente...")
    
    config = Config(pipe_gap=500, seed=42)
    env = FlappyEnv(config)
    
    print(f"Window height: {env.window_height}")
    print(f"Window width: {env.window_width}")
    print(f"Pipe gap: {config.pipe_gap}")
    
    obs, _ = env.reset()
    print(f"Observação inicial: {obs}")
    
    # Testa algumas ações manuais
    for i in range(10):
        action = 1 if i % 3 == 0 else 0  # Pula a cada 3 steps
        obs, reward, done, info = env.step(action)
        y_norm, vy_norm, dist_norm, delta_gap_norm = obs
        
        print(f"Step {i}: action={action}, y_norm={y_norm:.3f}, done={done}")
        
        if done:
            print(f"   Ambiente terminou no step {i}")
            break

if __name__ == "__main__":
    test_expert_policy()
    debug_environment()
