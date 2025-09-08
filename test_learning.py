#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from game_env import FlappyEnv, Config
from expert_policy import expert_action

def sigmoid(z):
    """Função sigmoid para logística"""
    z = np.clip(z, -500, 500)  # Evita overflow
    return 1 / (1 + np.exp(-z))

def model_action(obs, weights_dict):
    """Ação do modelo usando pesos treinados"""
    # Extrai componentes do dicionário de pesos
    w = weights_dict['w'].flatten()
    b = weights_dict['b']
    mean = weights_dict['mean'].flatten()
    std = weights_dict['std'].flatten()
    
    # Normaliza observação
    obs_norm = (obs - mean) / (std + 1e-8)
    
    # Calcula predição
    z = np.dot(obs_norm, w) + b
    prob = sigmoid(z)
    return 1 if prob > 0.5 else 0

def test_model(weights_file, num_episodes=10):
    """Testa um modelo específico"""
    if not os.path.exists(weights_file):
        return None
    
    weights_dict = np.load(weights_file, allow_pickle=True).item()
    config = Config(pipe_gap=400, seed=42)
    env = FlappyEnv(config)
    
    total_score = 0
    total_steps = 0
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_score = 0
        steps = 0
        
        for step in range(1000):
            action = model_action(obs, weights_dict)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if 'score' in info:
                episode_score = info['score']
            
            if done:
                break
        
        total_score += episode_score
        total_steps += steps
        if episode_score > 0:
            successes += 1
    
    return {
        'avg_score': total_score / num_episodes,
        'avg_steps': total_steps / num_episodes,
        'success_rate': (successes / num_episodes) * 100,
        'total_episodes': num_episodes
    }

def test_expert_policy(num_episodes=10):
    """Testa a política expert para comparação"""
    config = Config(pipe_gap=400, seed=42)
    env = FlappyEnv(config)
    
    total_score = 0
    total_steps = 0
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_score = 0
        steps = 0
        
        for step in range(1000):
            action = expert_action(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if 'score' in info:
                episode_score = info['score']
            
            if done:
                break
        
        total_score += episode_score
        total_steps += steps
        if episode_score > 0:
            successes += 1
    
    return {
        'avg_score': total_score / num_episodes,
        'avg_steps': total_steps / num_episodes,
        'success_rate': (successes / num_episodes) * 100,
        'total_episodes': num_episodes
    }

def main():
    print("🤖 TESTE DE APRENDIZADO DA IA - Flappy Bird")
    print("=" * 50)
    
    # Testa política expert primeiro
    print("🧠 Testando Política Expert (baseline)...")
    expert_results = test_expert_policy(20)
    if expert_results:
        print(f"   📊 Score médio: {expert_results['avg_score']:.2f}")
        print(f"   ⏱️  Steps médios: {expert_results['avg_steps']:.1f}")
        print(f"   ✅ Taxa sucesso: {expert_results['success_rate']:.1f}%")
    
    print("\n🎯 Testando Modelos Treinados...")
    
    # Lista alguns modelos específicos para testar
    models_to_test = [
        'weights.npy',
        'runs/best_weights.npy',
        'runs/run_1_weights.npy',
        'runs/run_50_weights.npy',
        'runs/run_100_weights.npy',
        'runs/run_144_weights.npy'  # Último modelo
    ]
    
    results = []
    
    for model_file in models_to_test:
        print(f"\n🔍 Testando: {model_file}")
        result = test_model(model_file, 15)
        
        if result:
            print(f"   📊 Score médio: {result['avg_score']:.2f}")
            print(f"   ⏱️  Steps médios: {result['avg_steps']:.1f}")
            print(f"   ✅ Taxa sucesso: {result['success_rate']:.1f}%")
            
            results.append({
                'model': model_file,
                **result
            })
        else:
            print(f"   ❌ Arquivo não encontrado!")
    
    # Análise comparativa
    print("\n" + "=" * 50)
    print("📈 ANÁLISE DE APRENDIZADO:")
    print("=" * 50)
    
    if expert_results and results:
        expert_score = expert_results['avg_score']
        expert_success = expert_results['success_rate']
        
        print(f"🧠 Expert Policy - Score: {expert_score:.2f}, Sucesso: {expert_success:.1f}%")
        print("📚 Modelos Treinados:")
        
        best_model = None
        best_score = -1
        
        for result in results:
            model_name = result['model'].replace('runs/', '').replace('.npy', '')
            score = result['avg_score']
            success = result['success_rate']
            
            # Comparação com expert
            score_diff = score - expert_score
            success_diff = success - expert_success
            
            status = "📈" if score >= expert_score * 0.8 else "📉"
            
            print(f"   {status} {model_name:<15} - Score: {score:.2f} ({score_diff:+.2f}) | Sucesso: {success:.1f}% ({success_diff:+.1f}%)")
            
            if score > best_score:
                best_score = score
                best_model = result
        
        if best_model:
            print(f"\n🏆 MELHOR MODELO: {best_model['model']}")
            print(f"   Score: {best_model['avg_score']:.2f}")
            print(f"   Taxa de Sucesso: {best_model['success_rate']:.1f}%")
            
            # Verifica se está aprendendo
            if best_model['avg_score'] >= expert_score * 0.7:
                print("\n✅ SIM! A IA ESTÁ APRENDENDO!")
                print("   O modelo consegue imitar a política expert com eficácia.")
            else:
                print("\n⚠️  A IA ainda está aprendendo...")
                print("   O modelo precisa de mais treinamento ou ajustes.")
        
        # Análise de progressão
        run_models = [r for r in results if 'run_' in r['model']]
        if len(run_models) >= 2:
            run_models.sort(key=lambda x: int(x['model'].split('_')[1]))
            first_score = run_models[0]['avg_score']
            last_score = run_models[-1]['avg_score']
            
            print(f"\n📊 PROGRESSÃO DE TREINAMENTO:")
            print(f"   Primeiro modelo: {first_score:.2f}")
            print(f"   Último modelo: {last_score:.2f}")
            print(f"   Melhoria: {last_score - first_score:+.2f}")
            
            if last_score > first_score:
                print("   🎯 PROGRESSO POSITIVO! A IA melhorou com o treinamento.")
            else:
                print("   🤔 Progresso variável. Pode precisar de mais dados ou ajustes.")

if __name__ == "__main__":
    main()
