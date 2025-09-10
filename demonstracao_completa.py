#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEMONSTRAÇÃO COMPLETA DO TRABALHO DE INTELIGÊNCIA ARTIFICIAL
============================================================

Este arquivo demonstra TODAS as funcionalidades implementadas no projeto:
1. Política Especialista (Expert Policy)
2. Coleta de Dataset
3. Treinamento de Regressão Logística 
4. Teste do Modelo Treinado
5. Visualização com Pygame
6. Comparação entre Expert e IA

Projeto: Flappy Bird com IA usando Regressão Logística e Aprendizado por Imitação
Autor: Trabalho de Inteligência Artificial
"""

import pygame
import sys
import numpy as np
import time
import os
from game_env import FlappyEnv, Config
from expert_policy import expert_action

# Inicialização do pygame
pygame.init()

# Configurações da tela
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
BIRD_SIZE = 15
PIPE_WIDTH = 60

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
BLUE = (30, 144, 255)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
ORANGE = (255, 165, 0)
PURPLE = (147, 112, 219)
CYAN = (0, 206, 209)

def sigmoid(z):
    """Função sigmoide para o modelo de regressão logística"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def model_action(obs, weights_pack):
    """Predição da ação usando o modelo treinado"""
    w = weights_pack.item().get("w")
    b = weights_pack.item().get("b")
    mean = weights_pack.item().get("mean")
    std = weights_pack.item().get("std")
    
    # Normalização das features
    x = (obs.reshape(1, -1) - mean) / (std + 1e-6)
    
    # Predição
    p = sigmoid(x @ w + b)[0, 0]
    return 1 if p >= 0.5 else 0

class DemonstracaoCompleta:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🤖 DEMONSTRAÇÃO COMPLETA - TRABALHO DE IA")
        self.clock = pygame.time.Clock()
        
        # Fontes
        self.title_font = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Configuração do ambiente
        self.config = Config(pipe_gap=300, seed=42)
        self.env = FlappyEnv(self.config)
        
        # Estados da demonstração
        self.current_demo = 0
        self.demos = [
            "Expert Policy",
            "Modelo Treinado", 
            "Comparação Expert vs IA",
            "Análise de Features"
        ]
        
        # Carregar modelo treinado (se existir)
        self.model_loaded = False
        self.weights_pack = None
        
        # Tentar carregar diferentes arquivos de pesos
        weight_files = ["weights_final.npy", "weights.npy", "runs/best_weights.npy"]
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                try:
                    self.weights_pack = np.load(weight_file, allow_pickle=True)
                    self.model_loaded = True
                    print(f"✅ Modelo carregado de: {weight_file}")
                    break
                except:
                    continue
        
        if not self.model_loaded:
            print("⚠️  Nenhum modelo encontrado. Treine primeiro com train_logreg.py")
    
    def draw_bird(self, y_norm, color=YELLOW):
        """Desenha o pássaro na posição normalizada"""
        bird_y = int(y_norm * SCREEN_HEIGHT)
        bird_x = 150
        
        # Corpo do pássaro
        pygame.draw.circle(self.screen, color, (bird_x, bird_y), BIRD_SIZE)
        pygame.draw.circle(self.screen, BLACK, (bird_x, bird_y), BIRD_SIZE, 2)
        
        # Olho
        eye_x = bird_x + 5
        eye_y = bird_y - 3
        pygame.draw.circle(self.screen, BLACK, (eye_x, eye_y), 2)
        
        # Bico
        beak_points = [(bird_x + BIRD_SIZE - 2, bird_y), 
                       (bird_x + BIRD_SIZE + 5, bird_y - 2), 
                       (bird_x + BIRD_SIZE + 5, bird_y + 2)]
        pygame.draw.polygon(self.screen, ORANGE, beak_points)
    
    def draw_pipes(self, dist_norm, delta_gap_norm):
        """Desenha os canos"""
        if dist_norm <= 1.0:
            # Posição do cano
            pipe_x = int(150 + dist_norm * (SCREEN_WIDTH - 300))
            
            # Centro do gap
            gap_center_norm = 0.5 - delta_gap_norm
            gap_center = int(gap_center_norm * SCREEN_HEIGHT)
            gap_size = self.config.pipe_gap
            
            # Cano superior
            top_height = gap_center - gap_size // 2
            if top_height > 0:
                pygame.draw.rect(self.screen, GREEN, 
                               (pipe_x, 0, PIPE_WIDTH, top_height))
                pygame.draw.rect(self.screen, BLACK, 
                               (pipe_x, 0, PIPE_WIDTH, top_height), 3)
            
            # Cano inferior
            bottom_y = gap_center + gap_size // 2
            bottom_height = SCREEN_HEIGHT - bottom_y
            if bottom_height > 0:
                pygame.draw.rect(self.screen, GREEN, 
                               (pipe_x, bottom_y, PIPE_WIDTH, bottom_height))
                pygame.draw.rect(self.screen, BLACK, 
                               (pipe_x, bottom_y, PIPE_WIDTH, bottom_height), 3)
            
            # Linha central do gap (referência)
            pygame.draw.line(self.screen, RED, 
                           (pipe_x, gap_center), 
                           (pipe_x + PIPE_WIDTH, gap_center), 2)
    
    def draw_info_panel(self, score, steps, action, obs, agent_name, color):
        """Desenha painel de informações"""
        y_norm, vy_norm, dist_norm, delta_gap_norm = obs
        
        # Painel de fundo
        panel_rect = pygame.Rect(10, 10, 300, 200)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Título do agente
        title_text = self.title_font.render(f"🤖 {agent_name}", True, color)
        self.screen.blit(title_text, (20, 20))
        
        # Informações básicas
        info_y = 55
        infos = [
            f"Score: {score}",
            f"Steps: {steps}",
            f"Ação: {'🚀 PULO' if action == 1 else '🌊 Plana'}",
            "",
            "📊 Features Normalizadas:",
            f"  Y_pos: {y_norm:.3f}",
            f"  Vel_Y: {vy_norm:.3f}",
            f"  Dist: {dist_norm:.3f}",
            f"  Gap_Δ: {delta_gap_norm:.3f}"
        ]
        
        for i, info in enumerate(infos):
            if info == "":
                info_y += 5
                continue
            text_color = RED if action == 1 and "Ação" in info else BLACK
            text = self.font.render(info, True, text_color)
            self.screen.blit(text, (20, info_y + i * 18))
    
    def draw_features_analysis(self, obs, action):
        """Desenha análise detalhada das features"""
        y_norm, vy_norm, dist_norm, delta_gap_norm = obs
        
        # Painel direito para análise
        panel_rect = pygame.Rect(SCREEN_WIDTH - 320, 10, 310, 400)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Título
        title = self.title_font.render("📈 Análise de Features", True, PURPLE)
        self.screen.blit(title, (SCREEN_WIDTH - 310, 20))
        
        # Análise de cada feature
        y_start = 60
        features = [
            ("Posição Y", y_norm, "Altura do pássaro (0=topo, 1=base)"),
            ("Velocidade Y", vy_norm, "Vel. vertical (-1=subindo, +1=caindo)"),
            ("Distância", dist_norm, "Dist. até próximo cano (0=muito perto)"),
            ("Delta Gap", delta_gap_norm, "Pos. relativa ao centro do gap")
        ]
        
        for i, (name, value, desc) in enumerate(features):
            y_pos = y_start + i * 80
            
            # Nome da feature
            name_text = self.font.render(name, True, BLUE)
            self.screen.blit(name_text, (SCREEN_WIDTH - 310, y_pos))
            
            # Valor
            value_text = self.small_font.render(f"Valor: {value:.3f}", True, BLACK)
            self.screen.blit(value_text, (SCREEN_WIDTH - 310, y_pos + 20))
            
            # Descrição
            desc_lines = self.wrap_text(desc, 35)
            for j, line in enumerate(desc_lines):
                desc_text = self.small_font.render(line, True, BLACK)
                self.screen.blit(desc_text, (SCREEN_WIDTH - 310, y_pos + 40 + j * 15))
            
            # Barra de progresso
            bar_rect = pygame.Rect(SCREEN_WIDTH - 310, y_pos + 65, 200, 10)
            pygame.draw.rect(self.screen, WHITE, bar_rect)
            pygame.draw.rect(self.screen, BLACK, bar_rect, 1)
            
            # Normalizar valor para barra (-1 a 1 -> 0 a 1)
            norm_value = (value + 1) / 2 if name == "Velocidade Y" or name == "Delta Gap" else value
            norm_value = max(0, min(1, norm_value))
            
            bar_fill = pygame.Rect(SCREEN_WIDTH - 310, y_pos + 65, int(200 * norm_value), 10)
            color = GREEN if 0.3 <= norm_value <= 0.7 else ORANGE if 0.1 <= norm_value <= 0.9 else RED
            pygame.draw.rect(self.screen, color, bar_fill)
    
    def wrap_text(self, text, max_chars):
        """Quebra texto em linhas"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def draw_controls(self):
        """Desenha controles na parte inferior"""
        control_y = SCREEN_HEIGHT - 100
        
        # Fundo dos controles
        control_rect = pygame.Rect(0, control_y, SCREEN_WIDTH, 100)
        pygame.draw.rect(self.screen, (240, 240, 240), control_rect)
        pygame.draw.rect(self.screen, BLACK, control_rect, 2)
        
        # Título
        title = self.title_font.render("🎮 CONTROLES", True, BLACK)
        self.screen.blit(title, (20, control_y + 10))
        
        # Demonstração atual
        demo_text = self.font.render(f"Demo Atual: {self.demos[self.current_demo]}", True, BLUE)
        self.screen.blit(demo_text, (20, control_y + 40))
        
        # Instruções
        controls = [
            "ESPAÇO: Próxima Demo",
            "ESC: Sair",
            "R: Reiniciar Episódio"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_text, (300 + i * 150, control_y + 40))
    
    def run_expert_demo(self):
        """Demonstração da política especialista"""
        obs, _ = self.env.reset()
        score = 0
        steps = 0
        
        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.current_demo = (self.current_demo + 1) % len(self.demos)
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        score = 0
                        steps = 0
            
            # Ação da política especialista
            action = expert_action(obs)
            obs, reward, done, info = self.env.step(action)
            steps += 1
            
            if 'score' in info:
                score = info['score']
            
            # Desenhar
            self.screen.fill(CYAN)
            self.draw_pipes(obs[2], obs[3])
            self.draw_bird(obs[0], YELLOW)
            self.draw_info_panel(score, steps, action, obs, "EXPERT POLICY", GREEN)
            self.draw_controls()
            
            # Painel explicativo
            exp_rect = pygame.Rect(SCREEN_WIDTH - 320, 250, 310, 200)
            pygame.draw.rect(self.screen, WHITE, exp_rect)
            pygame.draw.rect(self.screen, BLACK, exp_rect, 2)
            
            exp_title = self.title_font.render("🧠 Lógica Expert", True, GREEN)
            self.screen.blit(exp_title, (SCREEN_WIDTH - 310, 260))
            
            logic_text = [
                "Heurística simples:",
                "",
                "• Pula se muito abaixo do gap",
                "  (delta_gap > 0.12)",
                "",
                "• Pula se próximo do cano",
                "  (dist < 0.25) E abaixo do",
                "  centro E caindo",
                "",
                "• Caso contrário: não pula"
            ]
            
            for i, line in enumerate(logic_text):
                if line == "":
                    continue
                text = self.small_font.render(line, True, BLACK)
                self.screen.blit(text, (SCREEN_WIDTH - 310, 290 + i * 15))
            
            pygame.display.flip()
            self.clock.tick(60)
            
            if done:
                time.sleep(1)
                obs, _ = self.env.reset()
                score = 0
                steps = 0
    
    def run_model_demo(self):
        """Demonstração do modelo treinado"""
        if not self.model_loaded:
            # Mostrar mensagem de erro
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.current_demo = (self.current_demo + 1) % len(self.demos)
                            return True
                        elif event.key == pygame.K_ESCAPE:
                            return False
                
                self.screen.fill(WHITE)
                error_text = self.title_font.render("❌ MODELO NÃO ENCONTRADO", True, RED)
                self.screen.blit(error_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
                
                help_text = self.font.render("Execute train_logreg.py primeiro", True, BLACK)
                self.screen.blit(help_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 - 10))
                
                self.draw_controls()
                pygame.display.flip()
                self.clock.tick(60)
        
        obs, _ = self.env.reset()
        score = 0
        steps = 0
        
        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.current_demo = (self.current_demo + 1) % len(self.demos)
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        score = 0
                        steps = 0
            
            # Ação do modelo
            action = model_action(obs, self.weights_pack)
            obs, reward, done, info = self.env.step(action)
            steps += 1
            
            if 'score' in info:
                score = info['score']
            
            # Desenhar
            self.screen.fill(CYAN)
            self.draw_pipes(obs[2], obs[3])
            self.draw_bird(obs[0], BLUE)
            self.draw_info_panel(score, steps, action, obs, "MODELO IA", BLUE)
            self.draw_features_analysis(obs, action)
            self.draw_controls()
            
            pygame.display.flip()
            self.clock.tick(60)
            
            if done:
                time.sleep(1)
                obs, _ = self.env.reset()
                score = 0
                steps = 0
    
    def run_comparison_demo(self):
        """Demonstração comparativa"""
        if not self.model_loaded:
            return self.run_model_demo()  # Vai mostrar erro
        
        # Dois ambientes
        env1 = FlappyEnv(Config(pipe_gap=300, seed=42))
        env2 = FlappyEnv(Config(pipe_gap=300, seed=42))
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        score1 = score2 = 0
        steps1 = steps2 = 0
        
        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.current_demo = (self.current_demo + 1) % len(self.demos)
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_r:
                        obs1, _ = env1.reset()
                        obs2, _ = env2.reset()
                        score1 = score2 = 0
                        steps1 = steps2 = 0
            
            # Ações
            action1 = expert_action(obs1)
            action2 = model_action(obs2, self.weights_pack)
            
            # Steps
            obs1, reward1, done1, info1 = env1.step(action1)
            obs2, reward2, done2, info2 = env2.step(action2)
            
            steps1 += 1
            steps2 += 1
            
            if 'score' in info1:
                score1 = info1['score']
            if 'score' in info2:
                score2 = info2['score']
            
            # Desenhar lado a lado
            self.screen.fill(WHITE)
            
            # Linha divisória
            pygame.draw.line(self.screen, BLACK, 
                           (SCREEN_WIDTH//2, 0), 
                           (SCREEN_WIDTH//2, SCREEN_HEIGHT), 3)
            
            # Expert (esquerda)
            self.draw_bird_side(obs1[0], 0, YELLOW)
            self.draw_pipes_side(obs1[2], obs1[3], 0)
            self.draw_info_side(score1, steps1, action1, obs1, "EXPERT", GREEN, 0)
            
            # IA (direita)
            self.draw_bird_side(obs2[0], 1, BLUE)
            self.draw_pipes_side(obs2[2], obs2[3], 1)
            self.draw_info_side(score2, steps2, action2, obs2, "IA MODEL", BLUE, 1)
            
            # Título central
            title = self.title_font.render("⚔️  EXPERT vs IA  ⚔️", True, RED)
            self.screen.blit(title, (SCREEN_WIDTH//2 - 120, 10))
            
            self.draw_controls()
            
            pygame.display.flip()
            self.clock.tick(60)
            
            if done1 and done2:
                time.sleep(2)
                obs1, _ = env1.reset()
                obs2, _ = env2.reset()
                score1 = score2 = 0
                steps1 = steps2 = 0
            elif done1:
                obs1, _ = env1.reset()
                score1 = 0
                steps1 = 0
            elif done2:
                obs2, _ = env2.reset()
                score2 = 0
                steps2 = 0
    
    def draw_bird_side(self, y_norm, side, color):
        """Desenha pássaro em um lado da tela"""
        bird_y = int(y_norm * (SCREEN_HEIGHT - 120))
        bird_x = 100 + side * SCREEN_WIDTH // 2
        
        pygame.draw.circle(self.screen, color, (bird_x, bird_y), BIRD_SIZE)
        pygame.draw.circle(self.screen, BLACK, (bird_x, bird_y), BIRD_SIZE, 2)
    
    def draw_pipes_side(self, dist_norm, delta_gap_norm, side):
        """Desenha canos em um lado da tela"""
        if dist_norm <= 1.0:
            pipe_x = int(100 + side * SCREEN_WIDTH // 2 + dist_norm * (SCREEN_WIDTH // 2 - 200))
            gap_center_norm = 0.5 - delta_gap_norm
            gap_center = int(gap_center_norm * (SCREEN_HEIGHT - 120))
            gap_size = self.config.pipe_gap
            
            # Cano superior
            top_height = gap_center - gap_size // 2
            if top_height > 0:
                pygame.draw.rect(self.screen, GREEN, 
                               (pipe_x, 50, PIPE_WIDTH, top_height))
            
            # Cano inferior
            bottom_y = gap_center + gap_size // 2
            bottom_height = (SCREEN_HEIGHT - 120) - bottom_y
            if bottom_height > 0:
                pygame.draw.rect(self.screen, GREEN, 
                               (pipe_x, bottom_y + 50, PIPE_WIDTH, bottom_height))
    
    def draw_info_side(self, score, steps, action, obs, name, color, side):
        """Desenha informações de um lado"""
        x_offset = side * SCREEN_WIDTH // 2 + 10
        
        # Painel
        panel_rect = pygame.Rect(x_offset, SCREEN_HEIGHT - 200, SCREEN_WIDTH//2 - 20, 80)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, color, panel_rect, 2)
        
        # Info
        name_text = self.font.render(name, True, color)
        score_text = self.font.render(f"Score: {score}", True, BLACK)
        action_text = self.font.render(f"{'🚀 PULO' if action == 1 else '🌊 PLANA'}", True, RED if action == 1 else BLUE)
        
        self.screen.blit(name_text, (x_offset + 10, SCREEN_HEIGHT - 190))
        self.screen.blit(score_text, (x_offset + 10, SCREEN_HEIGHT - 170))
        self.screen.blit(action_text, (x_offset + 10, SCREEN_HEIGHT - 150))
    
    def run_analysis_demo(self):
        """Demonstração com análise de features"""
        obs, _ = self.env.reset()
        score = 0
        steps = 0
        
        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.current_demo = 0  # Volta para o início
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        score = 0
                        steps = 0
            
            # Ação do modelo (se disponível) ou expert
            if self.model_loaded:
                action = model_action(obs, self.weights_pack)
                agent_name = "IA ANALYSIS"
                color = PURPLE
                bird_color = PURPLE
            else:
                action = expert_action(obs)
                agent_name = "EXPERT ANALYSIS"
                color = GREEN
                bird_color = GREEN
            
            obs, reward, done, info = self.env.step(action)
            steps += 1
            
            if 'score' in info:
                score = info['score']
            
            # Desenhar
            self.screen.fill((230, 240, 255))  # Fundo azul claro
            self.draw_pipes(obs[2], obs[3])
            self.draw_bird(obs[0], bird_color)
            self.draw_info_panel(score, steps, action, obs, agent_name, color)
            self.draw_features_analysis(obs, action)
            self.draw_controls()
            
            pygame.display.flip()
            self.clock.tick(60)
            
            if done:
                time.sleep(1)
                obs, _ = self.env.reset()
                score = 0
                steps = 0
    
    def run(self):
        """Loop principal da demonstração"""
        print("🎮 INICIANDO DEMONSTRAÇÃO COMPLETA")
        print("=" * 50)
        print("🤖 Expert Policy: Política heurística")
        print("🧠 Modelo IA: Regressão logística treinada")
        print("⚔️  Comparação: Expert vs IA lado a lado")
        print("📊 Análise: Features em tempo real")
        print("=" * 50)
        
        running = True
        while running:
            if self.current_demo == 0:
                running = self.run_expert_demo()
            elif self.current_demo == 1:
                running = self.run_model_demo()
            elif self.current_demo == 2:
                running = self.run_comparison_demo()
            elif self.current_demo == 3:
                running = self.run_analysis_demo()
        
        pygame.quit()
        print("\n✅ Demonstração finalizada!")
        print("🎓 Trabalho de IA concluído com sucesso!")

def main():
    """Função principal"""
    print("🎓 TRABALHO DE INTELIGÊNCIA ARTIFICIAL")
    print("📝 Flappy Bird com Regressão Logística")
    print("🎯 Aprendizado por Imitação")
    print()
    
    # Verificar se há modelos treinados
    weight_files = ["weights_final.npy", "weights.npy", "runs/best_weights.npy"]
    model_found = any(os.path.exists(f) for f in weight_files)
    
    if not model_found:
        print("⚠️  ATENÇÃO: Nenhum modelo treinado encontrado!")
        print("💡 Execute primeiro:")
        print("   python train_logreg.py")
        print("   ou")
        print("   python run_experiments.py")
        print()
        print("📋 A demonstração mostrará apenas a Expert Policy")
        print()
    else:
        print("✅ Modelo(s) encontrado(s)! Demonstração completa disponível.")
        print()
    
    print("🎮 Iniciando demonstração...")
    time.sleep(2)
    
    demo = DemonstracaoCompleta()
    demo.run()

if __name__ == "__main__":
    main()
