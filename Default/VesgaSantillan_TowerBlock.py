# Proyecto por Edwin Vesga y Bryan Santillan
# Juego: Tower Stack | Algoritmo de ML: QLearning
# Importamos librerias
import pygame
import numpy as np
import time
import csv
import os

# Configuración del juego
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BLOCK_HEIGHT = 20
BLOCK_SPEED = 2
FPS = 60
MAX_BLOCKS = 21
GAME_OVER_DURATION = 5
LAST_BLOCK_SIZE = 100
LEARNING_EPISODES = 500  # Incrementado el número de episodios de aprendizaje
MAX_SCORE = LAST_BLOCK_SIZE*MAX_BLOCKS
episode_data_to_save = []  # Informacion de cada episodio para guardar.
# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Inicialización de Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class Block:
    def __init__(self, x, y, width, speed, color):
        self.rect = pygame.Rect(x, y, width, BLOCK_HEIGHT)
        self.speed = speed
        self.color = color
        self.static = False

    def update(self):
        if not self.static:
            self.rect.x += self.speed
            if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
                self.speed = -self.speed

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class StackGame:
    def __init__(self):
        self.blocks = [Block(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - BLOCK_HEIGHT, 100, BLOCK_SPEED, RED)]
        self.game_over = False
        self.score = 0
        self.last_size = 100

    def add_block(self):
        if len(self.blocks) < MAX_BLOCKS:
            last_block = self.blocks[-1]
            if last_block.static:
                new_width = last_block.rect.width
                new_color = self.get_next_color()
                new_block = Block(last_block.rect.x, last_block.rect.y - BLOCK_HEIGHT, new_width, -last_block.speed, new_color)
                self.blocks.append(new_block)
                self.score += new_width
            else:
                last_block.static = True
                if len(self.blocks) > 1:
                    prev_block = self.blocks[-2]
                    if last_block.rect.left < prev_block.rect.left:
                        last_block.rect.width = last_block.rect.width - (prev_block.rect.left - last_block.rect.left)
                        last_block.rect.x = prev_block.rect.left
                    if last_block.rect.right > prev_block.rect.right:
                        last_block.rect.width = last_block.rect.width - (last_block.rect.right - prev_block.rect.right)
                    if last_block.rect.right <= prev_block.rect.left or last_block.rect.left >= prev_block.rect.right:
                        self.game_over = True
        else:
            self.game_over = True

    def step(self, action):
        if action == 1:
            self.add_block()
            reward = self.blocks[-1].rect.width
            done = self.game_over
        else:
            reward = 0
            done = False
        state = self.blocks[-1].rect.width
        return state, reward, done

    def update(self):
        for block in self.blocks:
            block.update()

    def draw(self, screen):
        for block in self.blocks:
            block.draw(screen)
        episode_data_to_save.append([len(self.blocks), self.score])
        font = pygame.font.Font(None, 30)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (10, SCREEN_HEIGHT - 30))

        current_block = self.blocks[-1]
        if current_block.rect.width >= 1:
            self.last_size = current_block.rect.width
        size_text = font.render(f"Size: {self.last_size}", True, BLACK)
        screen.blit(size_text, (SCREEN_WIDTH - 120, 10))

        if self.game_over:
            game_over_text = font.render("GAME OVER", True, BLACK)
            score_final_text = font.render(f"Final Score: {self.score}", True, BLACK)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            score_final_rect = score_final_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            screen.blit(game_over_text, game_over_rect)
            screen.blit(score_final_text, score_final_rect)
            last_block = self.blocks[-1]
            last_block.draw(screen)

    def get_next_color(self):
        color_cycle = [RED, BLUE, YELLOW]
        current_color = self.blocks[-1].color
        current_index = color_cycle.index(current_color)
        next_index = (current_index + 1) % len(color_cycle)
        return color_cycle[next_index]

class QLearningAgent:
    def __init__(self, game, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
        self.game = game
        self.q_table = np.random.uniform(low=-2, high=0, size=(MAX_SCORE + 1, 2))  # Inicialización de la tabla Q con valores aleatorios
        self.alpha = alpha  # Taza de aprendizaje.
        self.gamma = gamma  # Factor de descuento.
        self.epsilon = epsilon  # Taza de exploración.
        self.epsilon_decay = epsilon_decay  # Declive de la taza de exploración.
        self.episode_duration = []
        self.episode_scores = []  # Puntaje para cada episodio.
        self.episode_data = []  # Informacion de cada episodio para guardar.

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1])  # Seleccion aleatoria para la exploración.
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        future_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_max)
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def save_episode_data(self, filename='episode_data.csv'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Time", "Blocks placed", "Final Score"])
            for episode, data in enumerate(self.episode_data):
                writer.writerow([episode + 1] + data)

def main():
    game = StackGame()
    agent = QLearningAgent(game)

    print("Menu:")
    print("1. Juego un jugador.")
    print("2. Ver aprendizaje QLearning")
    choice = input("Selecione una opcion (1 o 2): ")

    if choice == "1":
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not game.game_over:
                        game.add_block()
            game.update()
            screen.fill(WHITE)
            game.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)

    elif choice == "2":
        num_episodes = LEARNING_EPISODES
        for episode in range(num_episodes):
            game = StackGame()  # Restaurar juego para un nuevo episodio.
            start_time = time.time()  # Tiempo inicial para cada episodio.
            state = game.blocks[-1].rect.width
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                action = agent.get_action(state)
                next_state, reward, done = game.step(action)

                agent.update_q_table(state, action, reward, next_state)
                state = next_state

                if done:
                    end_time = time.time()
                    duration = end_time - start_time
                    agent.episode_data.append([duration]+episode_data_to_save[-1])  # Guardar informacion del episodio
                    time.sleep(1) 

                    break

                game.update()
                screen.fill(WHITE)
                game.draw(screen)

                pygame.display.flip()
                clock.tick(FPS)
        agent.save_episode_data()
    pygame.quit()

if __name__ == "__main__":
    main()