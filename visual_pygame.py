#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import sys
import time
from game_env import FlappyEnv, Config
from expert_policy import expert_action

# Inicialização do pygame
pygame.init()

# Configurações da tela
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BIRD_SIZE = 20
PIPE_WIDTH = 50

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

class VisualFlappyAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🤖 Flappy Bird IA - Visualização em Tempo Real")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Configuração do ambiente com mais espaço
        self.config = Config(pipe_gap=400, seed=42)  # Gap maior para facilitar navegação
        self.env = FlappyEnv(self.config)
        
    def draw_bird(self, y_norm):
        """Desenha o pássaro na posição normalizada"""
        bird_y = int(y_norm * SCREEN_HEIGHT)
        bird_x = 100
        
        # Desenha o pássaro como um círculo amarelo
        pygame.draw.circle(self.screen, YELLOW, (bird_x, bird_y), BIRD_SIZE)
        pygame.draw.circle(self.screen, BLACK, (bird_x, bird_y), BIRD_SIZE, 2)
        
        # Olho do pássaro
        eye_x = bird_x + 8
        eye_y = bird_y - 5
        pygame.draw.circle(self.screen, BLACK, (eye_x, eye_y), 3)
        
    def draw_pipes(self, dist_norm, delta_gap_norm):
        """Desenha os canos"""
        if dist_norm < 1.0:
            pipe_x = int(100 + dist_norm * (SCREEN_WIDTH - 200))
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
    
    def draw_info(self, score, steps, action, obs):
        """Desenha informações na tela"""
        y_norm, vy_norm, dist_norm, delta_gap_norm = obs
        
        # Score e steps
        score_text = self.font.render(f"Score: {score}", True, BLACK)
        steps_text = self.font.render(f"Steps: {steps}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 50))
        
        # Ação da IA
        action_text = "🚀 PULO!" if action == 1 else "🌊 Planando"
        action_color = RED if action == 1 else BLUE
        ai_text = self.font.render(f"IA: {action_text}", True, action_color)
        self.screen.blit(ai_text, (10, 90))
        
        # Dados técnicos
        info_y = 140
        info_texts = [
            f"Posição Y: {y_norm:.3f}",
            f"Velocidade: {vy_norm:.3f}",
            f"Distância cano: {dist_norm:.3f}",
            f"Delta gap: {delta_gap_norm:.3f}"
        ]
        
        for i, text in enumerate(info_texts):
            rendered = self.small_font.render(text, True, BLACK)
            self.screen.blit(rendered, (10, info_y + i * 25))
    
    def run_episode(self):
        """Executa um episódio com visualização"""
        obs, _ = self.env.reset()
        score = 0
        steps = 0
        running = True
        
        while running:
            # Processa eventos do pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Próximo episódio
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
            
            # IA decide ação
            action = expert_action(obs)
            
            # Executa ação no ambiente
            obs, reward, done, info = self.env.step(action)
            steps += 1
            
            if 'score' in info:
                score = info['score']
            
            # Limpa a tela
            self.screen.fill(WHITE)
            
            # Desenha elementos do jogo
            y_norm, vy_norm, dist_norm, delta_gap_norm = obs
            self.draw_pipes(dist_norm, delta_gap_norm)
            self.draw_bird(y_norm)
            self.draw_info(score, steps, action, obs)
            
            # Mostra mensagem se passou por cano
            if reward > 0.5:
                success_text = self.font.render("🎉 PASSOU PELO CANO! 🎉", True, GREEN)
                text_rect = success_text.get_rect(center=(SCREEN_WIDTH//2, 100))
                self.screen.blit(success_text, text_rect)
            
            # Instruções
            inst_text = self.small_font.render("ESPAÇO: Próximo episódio | ESC: Sair", True, BLACK)
            self.screen.blit(inst_text, (SCREEN_WIDTH - 300, SCREEN_HEIGHT - 30))
            
            # Atualiza a tela
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            
            if done:
                # Mostra resultado final
                result_text = f"🏆 Score Final: {score}" if score > 0 else "💥 Colidiu!"
                result_color = GREEN if score > 0 else RED
                final_text = self.font.render(result_text, True, result_color)
                text_rect = final_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                
                # Fundo para o texto
                pygame.draw.rect(self.screen, WHITE, text_rect.inflate(20, 10))
                pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10), 2)
                self.screen.blit(final_text, text_rect)
                
                restart_text = self.small_font.render("Pressione ESPAÇO para novo episódio", True, BLACK)
                restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40))
                self.screen.blit(restart_text, restart_rect)
                
                pygame.display.flip()
                
                # Espera input do usuário
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                return False
                break
        
        return True
    
    def run(self):
        """Loop principal do jogo"""
        print("🎮 Flappy Bird IA - Pygame Visualização")
        print("ESPAÇO: Próximo episódio | ESC: Sair")
        
        episode = 1
        try:
            while True:
                print(f"🚀 Iniciando episódio {episode}")
                
                # Mostra tela de início do episódio
                self.screen.fill(WHITE)
                title_text = self.font.render(f"🤖 Episódio {episode}", True, BLACK)
                title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                self.screen.blit(title_text, title_rect)
                
                start_text = self.small_font.render("Pressione ESPAÇO para começar", True, BLACK)
                start_rect = start_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40))
                self.screen.blit(start_text, start_rect)
                
                pygame.display.flip()
                
                # Espera comando para iniciar
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                return
                
                # Executa episódio
                if not self.run_episode():
                    break
                
                episode += 1
                
        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()
            print("👋 Visualização encerrada!")

def main():
    try:
        game = VisualFlappyAI()
        game.run()
    except Exception as e:
        print(f"Erro: {e}")
        print("Certifique-se de que o pygame está instalado e funcionando.")

if __name__ == "__main__":
    main()
import sys
import time
from game_env import FlappyEnv, Config
from expert_policy import expert_action

# Configurações visuais
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (135, 206, 235)  # Sky blue
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

class VisualFlappyGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🎮 Flappy Bird IA - Visualização Prática")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Configuração do jogo
        self.config = Config(pipe_gap=300, seed=42)
        self.env = FlappyEnv(self.config)
        
        # Estado do jogo
        self.episode = 1
        self.reset_game()
        
    def reset_game(self):
        self.obs, _ = self.env.reset()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_action = 0
        self.action_timer = 0
        
    def get_pipe_positions(self):
        """Calcula posições dos canos para renderização"""
        y_norm, vy_norm, dist_norm, delta_gap_norm = self.obs
        
        pipes = []
        if dist_norm < 1.0:
            # Posição horizontal do cano
            pipe_x = WINDOW_WIDTH * 0.2 + (dist_norm * WINDOW_WIDTH * 0.6)
            
            # Centro do gap
            gap_center_y = WINDOW_HEIGHT * 0.5 - (delta_gap_norm * WINDOW_HEIGHT * 0.3)
            gap_size = self.config.pipe_gap
            
            # Cano superior
            upper_pipe = pygame.Rect(pipe_x - 30, 0, 60, gap_center_y - gap_size//2)
            # Cano inferior  
            lower_pipe = pygame.Rect(pipe_x - 30, gap_center_y + gap_size//2, 60, WINDOW_HEIGHT)
            
            pipes.append((upper_pipe, lower_pipe))
            
        return pipes
    
    def draw_bird(self):
        """Desenha o pássaro"""
        y_norm, vy_norm, _, _ = self.obs
        
        # Posição do pássaro
        bird_x = WINDOW_WIDTH * 0.2
        bird_y = WINDOW_HEIGHT * (1 - y_norm)
        
        # Cor baseada na velocidade
        if vy_norm < -0.3:  # Subindo rápido
            color = YELLOW
        elif vy_norm > 0.3:  # Caindo rápido
            color = RED
        else:
            color = GREEN
            
        # Desenha o pássaro
        pygame.draw.circle(self.screen, color, (int(bird_x), int(bird_y)), 15)
        pygame.draw.circle(self.screen, BLACK, (int(bird_x), int(bird_y)), 15, 2)
        
        # Olho
        eye_x = bird_x + 5
        eye_y = bird_y - 3
        pygame.draw.circle(self.screen, WHITE, (int(eye_x), int(eye_y)), 4)
        pygame.draw.circle(self.screen, BLACK, (int(eye_x), int(eye_y)), 2)
        
    def draw_pipes(self):
        """Desenha os canos"""
        pipes = self.get_pipe_positions()
        
        for upper_pipe, lower_pipe in pipes:
            # Canos
            pygame.draw.rect(self.screen, GREEN, upper_pipe)
            pygame.draw.rect(self.screen, GREEN, lower_pipe)
            
            # Bordas dos canos
            pygame.draw.rect(self.screen, BLACK, upper_pipe, 3)
            pygame.draw.rect(self.screen, BLACK, lower_pipe, 3)
            
    def draw_ui(self):
        """Desenha interface do usuário"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        
        # Episódio
        episode_text = self.font.render(f"Episódio: {self.episode}", True, BLACK)
        self.screen.blit(episode_text, (10, 50))
        
        # Steps
        steps_text = self.small_font.render(f"Steps: {self.steps}", True, BLACK)
        self.screen.blit(steps_text, (10, 90))
        
        # Última ação da IA
        if self.action_timer > 0:
            action_text = "🚀 IA PULOU!" if self.last_action == 1 else "🌊 IA Planando"
            color = YELLOW if self.last_action == 1 else BLUE
            action_surface = self.font.render(action_text, True, color)
            self.screen.blit(action_surface, (WINDOW_WIDTH - 200, 10))
            self.action_timer -= 1
            
        # Dados da IA
        y_norm, vy_norm, dist_norm, delta_gap_norm = self.obs
        
        info_y = WINDOW_HEIGHT - 120
        info_texts = [
            f"Altura: {y_norm:.2f}",
            f"Velocidade: {vy_norm:.2f}",
            f"Distância: {dist_norm:.2f}",
            f"Gap Delta: {delta_gap_norm:.2f}"
        ]
        
        for i, text in enumerate(info_texts):
            surface = self.small_font.render(text, True, BLACK)
            self.screen.blit(surface, (10, info_y + i * 20))
            
        # Instruções
        if not self.game_over:
            instruction = self.small_font.render("Pressione ESPAÇO para próximo episódio | ESC para sair", True, GRAY)
            self.screen.blit(instruction, (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT - 30))
            
    def draw_game_over(self):
        """Desenha tela de game over"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self.score > 0:
            text = f"🎉 SUCESSO! Score: {self.score} 🎉"
            color = GREEN
        else:
            text = "💥 Colidiu! Tentando novamente..."
            color = RED
            
        game_over_surface = self.font.render(text, True, color)
        text_rect = game_over_surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(game_over_surface, text_rect)
        
        restart_text = self.small_font.render("Pressione ESPAÇO para continuar", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 40))
        self.screen.blit(restart_text, restart_rect)
        
    def run(self):
        """Loop principal do jogo"""
        running = True
        auto_restart_timer = 0
        
        print("🎮 Iniciando visualização Pygame do Flappy Bird IA!")
        print("🚀 A IA vai jogar automaticamente!")
        print("📝 Pressione ESPAÇO para pular episódios ou ESC para sair")
        
        while running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.game_over:
                            self.episode += 1
                            self.reset_game()
                            auto_restart_timer = 0
                        else:
                            # Pula para próximo episódio
                            self.game_over = True
                            auto_restart_timer = 60  # 1 segundo
            
            # Lógica do jogo
            if not self.game_over:
                # IA toma decisão
                action = expert_action(self.obs)
                self.last_action = action
                self.action_timer = 30  # Mostra ação por 0.5 segundos
                
                # Executa ação
                self.obs, reward, done, info = self.env.step(action)
                self.steps += 1
                
                if 'score' in info:
                    self.score = info['score']
                    
                if done:
                    self.game_over = True
                    auto_restart_timer = 120  # 2 segundos de pausa
                    
            else:
                # Auto restart após delay
                if auto_restart_timer > 0:
                    auto_restart_timer -= 1
                else:
                    self.episode += 1
                    self.reset_game()
            
            # Renderização
            self.screen.fill(BLUE)  # Céu azul
            
            if not self.game_over:
                self.draw_pipes()
                self.draw_bird()
            else:
                # Ainda mostra o jogo em background
                self.draw_pipes()
                self.draw_bird()
                self.draw_game_over()
                
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

def main():
    try:
        game = VisualFlappyGame()
        game.run()
    except Exception as e:
        print(f"Erro: {e}")
        print("Talvez o pygame não consiga abrir uma janela neste ambiente.")
        print("Tente executar em um ambiente com interface gráfica.")

if __name__ == "__main__":
    main()
