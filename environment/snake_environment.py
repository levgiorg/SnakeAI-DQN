import pygame
import random
import numpy as np
import sys
import gym
from gym import spaces

from config import Config
from .direction import Direction
from .point import Point


class SnakeEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=640, height=480, seed=None):
        super(SnakeEnvironment, self).__init__()
        pygame.init()
        config = Config()

        env_config = config.get_section('environment')
        self.width = env_config.get('width')
        self.height = env_config.get('height')
        self.block_size = env_config.get('block_size',)
        self.speed = env_config.get('speed')
        self.colors = env_config.get('colors') 

        font_config = env_config.get('font')
        self.font = pygame.font.SysFont(font_config['name'], font_config['size'])

        # Set seed for environment randomness
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Define action space and observation space
        self.action_space = spaces.Discrete(3)
        self.state_size = 11  # Length of the state vector
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Initialize game state
        self.display = None
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Randomize initial direction
        self.direction = random.choice(list(Direction))

        # Randomize initial head position within the grid
        grid_width = self.width // self.block_size
        grid_height = self.height // self.block_size
        head_x = random.randint(0, grid_width - 1) * self.block_size
        head_y = random.randint(0, grid_height - 1) * self.block_size
        self.head = Point(head_x, head_y)

        # Ensure the snake's body is correctly positioned
        self.snake = [self.head,
                    Point(self.head.x - self.block_size, self.head.y),
                    Point(self.head.x - (2 * self.block_size), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.iteration = 0
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        self.iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        if self._is_collision():
            reward = -10  # Penalty for collision
            done = True
        else:
            if self.head == self.food:
                self.score += 1
                reward = 10  # Reward for eating food
                self._place_food()
            else:
                self.snake.pop()  # Move forward without eating

        return reward, done, self.score

    def _move(self, action):
        # Actions: 0 = Straight, 1 = Right Turn, 2 = Left Turn
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == 0:
            new_direction = clockwise[idx]  # No change
        elif action == 1:
            next_idx = (idx + 1) % 4
            new_direction = clockwise[next_idx]  # Right turn
        elif action == 2:
            next_idx = (idx - 1) % 4
            new_direction = clockwise[next_idx]  # Left turn
        else:
            raise ValueError(f"Invalid action: {action}")

        self.direction = new_direction

        # Update head position
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def _is_collision(self, point=None):
        if point is None:
            point = self.head

        # Check boundaries
        if point.x > self.width - self.block_size or point.x < 0 or point.y > self.height - self.block_size or point.y < 0:
            return True

        # Check if it hits itself
        if point in self.snake[1:]:
            return True

        return False

    def get_frame(self):
        # Get the current frame as a numpy array
        frame = pygame.surfarray.array3d(self.display)
        # Transpose the axes to match imageio's expectation (width, height, channels)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def get_state(self):
        # Head position
        head_x = self.head.x
        head_y = self.head.y

        # Current direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Danger detection
        point_straight = self._get_next_point(self.direction)
        point_right = self._get_next_point(self._rotate_direction(self.direction, 1))
        point_left = self._get_next_point(self._rotate_direction(self.direction, -1))

        danger_straight = self._is_collision(point_straight)
        danger_right = self._is_collision(point_right)
        danger_left = self._is_collision(point_left)

        # Food location relative to head
        food_left = self.food.x < head_x
        food_right = self.food.x > head_x
        food_up = self.food.y < head_y
        food_down = self.food.y > head_y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down)
        ]
        return np.array(state, dtype=int)

    def render(self, mode='human'):
        if self.display is None:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')

        self.display.fill(self.colors['black'])

        for part in self.snake:
            pygame.draw.rect(self.display, self.colors['green1'], pygame.Rect(part.x, part.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, self.colors['green2'], pygame.Rect(part.x + 4, part.y + 4, 12, 12))

        pygame.draw.rect(self.display, self.colors['red'], pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        text = self.font.render("Score: " + str(self.score), True, self.colors['white'])
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            self.display = None
        pygame.quit()

    def _get_next_point(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
        return Point(x, y)

    def _rotate_direction(self, current_direction, turn):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(current_direction)
        idx = (idx + turn) % len(directions)
        return directions[idx]
