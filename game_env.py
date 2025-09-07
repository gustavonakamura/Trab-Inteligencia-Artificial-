import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

@dataclass
class Config:
    width: int = 400
    height: int = 600
    player_x: int = 80
    player_size: int = 20
    gravity: float = 0.35
    flap_impulse: float = -7.0
    pipe_speed: float = 3.0
    pipe_gap: int = 150
    pipe_width: int = 50
    pipe_interval_px: int = 220
    max_steps: int = 10000
    seed: Optional[int] = None
    vy_min: float = -12
    vy_max: float = 12

class FlappyEnv:
    """
    Observação (4 features):
        x0 = y_norm              (0..1)
        x1 = vy_norm             (-1..1 clamped)
        x2 = dist_norm           (0..1)  distância horizontal até a borda direita do próximo cano
        x3 = delta_gap_norm      (-1..1) centro do gap - y, normalizado por altura
    Ação: 0 = nada | 1 = pular
    Recompensa (usada só para referência durante coleta): +0.1 vivo, +1 ao passar cano, -1 colisão.
    """
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.y = 0.0
        self.vy = 0.0
        self.pipes: List[Tuple[float, float]] = []
        self.steps = 0
        self.score = 0
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.y = self.cfg.height * 0.5
        self.vy = 0.0
        self.steps = 0
        self.score = 0
        self.pipes.clear()
        self._spawn_pipe(self.cfg.width + 80)
        self._spawn_pipe(self.cfg.width + 80 + self.cfg.pipe_interval_px)
        return self._obs(), {"score": self.score}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action == 1:
            self.vy = self.cfg.flap_impulse
        self.vy += self.cfg.gravity
        self.vy = max(self.cfg.vy_min, min(self.vy, self.cfg.vy_max))
        self.y += self.vy

        passed_reward = 0.0
        new_pipes = []
        for (x, gy) in self.pipes:
            x2 = x - self.cfg.pipe_speed
            if x + self.cfg.pipe_width >= self.cfg.player_x and x2 + self.cfg.pipe_width < self.cfg.player_x:
                self.score += 1
                passed_reward += 1.0
            if x2 + self.cfg.pipe_width > 0:
                new_pipes.append((x2, gy))
        self.pipes = new_pipes

        if len(self.pipes) == 0 or (self.pipes[-1][0] < self.cfg.width - self.cfg.pipe_interval_px):
            self._spawn_pipe(self.cfg.width + 40)

        done = False
        reward = 0.1 + passed_reward

        if self.y < 0 or self.y > self.cfg.height:
            reward -= 1.0
            done = True
        else:
            for (x, gy) in self.pipes:
                if self.cfg.player_x + self.cfg.player_size > x and self.cfg.player_x < x + self.cfg.pipe_width:
                    gap_top = gy - self.cfg.pipe_gap / 2
                    gap_bottom = gy + self.cfg.pipe_gap / 2
                    player_top = self.y - self.cfg.player_size / 2
                    player_bottom = self.y + self.cfg.player_size / 2
                    if player_top < gap_top or player_bottom > gap_bottom:
                        reward -= 1.0
                        done = True
                        break

        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            done = True

        return self._obs(), reward, done, {"score": self.score}

    def _spawn_pipe(self, x: float):
        margin = 90
        gy = self.rng.randint(margin, self.cfg.height - margin)
        self.pipes.append((x, float(gy)))

    def _nearest_pipe(self) -> Tuple[float, float]:
        candidates = [(x, gy) for (x, gy) in self.pipes if x + self.cfg.pipe_width >= self.cfg.player_x - 1]
        if not candidates:
            return float(self.cfg.width), self.cfg.height * 0.5
        x, gy = min(candidates, key=lambda t: t[0])
        dist_right = max(0.0, (x + self.cfg.pipe_width) - self.cfg.player_x)
        return dist_right, gy

    def _obs(self) -> np.ndarray:
        dist_right, gy = self._nearest_pipe()
        y_norm = self.y / self.cfg.height
        vy_norm = max(-1.0, min(1.0, self.vy / max(1e-6, self.cfg.vy_max)))
        dist_norm = max(0.0, min(1.0, dist_right / self.cfg.width))
        delta_gap_norm = max(-1.0, min(1.0, (gy - self.y) / self.cfg.height))
        return np.array([y_norm, vy_norm, dist_norm, delta_gap_norm], dtype=np.float32)

    # ------------ Render (só para avaliação/jogo humano) ------------
    def render(self):
        if pygame is None:
            raise RuntimeError("Pygame não instalado.")
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
            pygame.display.set_caption("Flappy Supervisionado")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 26)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((35, 35, 40))
        for (x, gy) in self.pipes:
            gap_top = gy - self.cfg.pipe_gap / 2
            gap_bottom = gy + self.cfg.pipe_gap / 2
            pygame.draw.rect(self.screen, (80, 200, 120), (x, 0, self.cfg.pipe_width, gap_top))
            pygame.draw.rect(self.screen, (80, 200, 120), (x, gap_bottom, self.cfg.pipe_width, self.cfg.height - gap_bottom))
        pygame.draw.rect(self.screen, (230, 210, 60),
                         (self.cfg.player_x - self.cfg.player_size/2, self.y - self.cfg.player_size/2,
                          self.cfg.player_size, self.cfg.player_size))
        txt = self.font.render(f"Score: {self.score}", True, (230, 230, 230))
        self.screen.blit(txt, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.quit()
            self.screen = None
