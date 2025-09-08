#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
from game_env import FlappyEnv, Config
from expert_policy import expert_action

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def draw_game(bird_y, pipes, score, steps):
    """Desenha o jogo em ASCII art"""
    height = 20
    width = 40
    
    # Converte posições normalizadas para coordenadas da tela
    bird_row = int((1 - bird_y) * height)
    bird_row = max(0, min(height-1, bird_row))
    
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Desenha as bordas
    for i in range(height):
        screen[i][0] = '|'
        screen[i][width-1] = '|'
    for j in range(width):
        screen[0][j] = '-'
        screen[height-1][j] = '-'
    
    # Desenha os canos
    for pipe_x, gap_center in pipes:
        pipe_col = int(pipe_x * width)
        if 0 <= pipe_col < width:
            gap_row = int((1 - gap_center) * height)
            gap_size = 6  # Tamanho do gap
            
            # Cano superior
            for row in range(max(1, gap_row - gap_size//2)):
                if 0 <= row < height:
                    screen[row][pipe_col] = '█'
            
            # Cano inferior  
            for row in range(min(height-1, gap_row + gap_size//2), height-1):
                if 0 <= row < height:
                    screen[row][pipe_col] = '█'
    
    # Desenha o pássaro
    if 0 <= bird_row < height:
        screen[bird_row][5] = '🐦'
    
    # Imprime a tela
    clear_screen()
    print("🎮 FLAPPY BIRD IA - VISUALIZAÇÃO PRÁTICA")
    print("=" * 50)
    print(f"Score: {score} | Steps: {steps}")
    print()
    
    for row in screen:
        print(''.join(row))
    
    print()
    print("🐦 = Pássaro | █ = Canos | Score quando passa pelos canos!")

def main():
    config = Config(pipe_gap=300, seed=42)
    env = FlappyEnv(config)
    
    print("🚀 Iniciando visualização do Flappy Bird IA...")
    print("Pressione Ctrl+C para parar")
    time.sleep(2)
    
    episode = 1
    
    try:
        while True:
            print(f"\n🎯 EPISÓDIO {episode}")
            obs, _ = env.reset()
            score = 0
            steps = 0
            
            while True:
                # Estado atual
                y_norm, vy_norm, dist_norm, delta_gap_norm = obs
                
                # Simula posições dos canos para visualização
                pipes = []
                if dist_norm < 1.0:
                    pipe_x = dist_norm
                    gap_center = 0.5 - delta_gap_norm  # Inverte para visualização
                    pipes.append((pipe_x, gap_center))
                
                # Desenha o jogo
                draw_game(y_norm, pipes, score, steps)
                
                # IA decide ação
                action = expert_action(obs)
                action_text = "PULO! 🚀" if action == 1 else "Planando..."
                print(f"IA decidiu: {action_text}")
                
                # Executa ação
                obs, reward, done, info = env.step(action)
                steps += 1
                
                if 'score' in info:
                    new_score = info['score']
                    if new_score > score:
                        print("🎉 PASSOU PELO CANO! +1 PONTO!")
                        time.sleep(1)
                    score = new_score
                
                time.sleep(0.2)  # Velocidade da animação
                
                if done:
                    draw_game(obs[0], pipes, score, steps)
                    if score > 0:
                        print(f"🏆 SUCESSO! Score final: {score}")
                    else:
                        print("💥 Colidiu! Tentando novamente...")
                    time.sleep(2)
                    break
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Visualização encerrada!")

if __name__ == "__main__":
    main()
