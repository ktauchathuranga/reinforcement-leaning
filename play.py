import torch
import numpy as np
from game import SnakeGameAI, Point, Direction
from model import Linear_QNet

class Player:
    def __init__(self):
        # Initialize the neural network with the same architecture as in agent.py
        self.model = Linear_QNet(11, 256, 3)
        # Load the trained model’s state dictionary from the saved file
        self.model.load_state_dict(torch.load('./model/model.pth'))
        # Set the model to evaluation mode (no training)
        self.model.eval()

    def get_state(self, game):
        # This method is identical to Agent.get_state in agent.py
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        # Convert state to a PyTorch tensor
        state0 = torch.tensor(state, dtype=torch.float)
        # Get the model’s prediction
        prediction = self.model(state0)
        # Choose the action with the highest Q-value
        move = torch.argmax(prediction).item()
        # Convert to one-hot format [straight, right, left]
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def play():
    # Create instances of Player and SnakeGameAI
    player = Player()
    game = SnakeGameAI(speed=20)  # Slower speed for watching (adjust as needed)

    while True:
        # Get the current state
        state = player.get_state(game)
        # Decide the action using the trained model
        action = player.get_action(state)
        # Perform the action in the game
        reward, done, score = game.play_step(action)

        if done:
            # Game over: reset and print the score
            game.reset()
            print(f"Game over! Score: {score}")
            # Continues playing until the window is closed

if __name__ == '__main__':
    play()
