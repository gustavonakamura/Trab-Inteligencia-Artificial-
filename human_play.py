from game_env import FlappyEnv, Config, pygame

def main():
    env = FlappyEnv(Config())
    obs, info = env.reset()
    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1
        obs, r, done, info = env.step(action)
        env.render()
        if done:
            obs, info = env.reset()

if __name__ == "__main__":
    if pygame is None:
        raise RuntimeError("Instale pygame: pip install pygame")
    main()
