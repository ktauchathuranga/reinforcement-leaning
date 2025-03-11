# Snake Game AI with Reinforcement Learning

This project demonstrates how to train an AI agent to play the classic Snake game using reinforcement learning techniques. The agent learns to maximize its score by navigating the snake to eat food while avoiding collisions with walls and itself. The implementation uses PyTorch for building and training the neural network model.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Playing with the Trained Model](#playing-with-the-trained-model)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The Snake Game AI project uses a reinforcement learning approach to train an agent to play the Snake game. The agent learns through trial and error, receiving rewards for eating food and penalties for collisions. The neural network model is trained using Q-learning, and the project includes scripts for both training the model and playing the game with the trained model.

## Features

- **Reinforcement Learning**: Implements Q-learning with experience replay and a deep neural network.
- **PyTorch Integration**: Utilizes PyTorch for building and training the neural network model.
- **Customizable Game Speed**: Allows adjusting the game speed for training and playing.
- **Model Saving and Loading**: Automatically saves the best-performing model during training and loads it for playing.
- **Real-time Visualization**: Displays the game interface during training and playing.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/snake-game-ai.git
   cd snake-game-ai
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.6+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   If there’s no `requirements.txt`, install the following packages manually:
   ```bash
   pip install torch numpy matplotlib pygame
   ```

3. **Verify Installation**:
   Run the training script to ensure everything is set up correctly:
   ```bash
   python agent.py
   ```

## Usage

### Training the Model

To train the model, run the `agent.py` script:
```bash
python agent.py
```
- The training process will start, and you can monitor the agent’s progress through the console output and the game interface.
- The model is automatically saved to `./model/model.pth` whenever a new high score is achieved.

### Playing with the Trained Model

To play the game using the trained model, run the `play.py` script:
```bash
python play.py
```
- The game will run at a slower speed (adjustable in the script) to make it watchable.
- The AI will control the snake, and the game will reset automatically when the snake collides with a wall or itself.

## Project Structure

- **`agent.py`**: Contains the main training loop and the `Agent` class responsible for learning.
- **`game.py`**: Defines the `SnakeGameAI` class, which handles the game logic and interface.
- **`helper.py`**: Includes utility functions for plotting the training progress.
- **`model.py`**: Defines the neural network model (`Linear_QNet`) and the training logic (`QTrainer`).
- **`play.py`**: Script to play the game using the trained model.

## How It Works

1. **State Representation**:
    - The state is an 11-dimensional vector capturing the snake’s surroundings, including dangers in different directions and the position of the food relative to the snake’s head.

2. **Action Space**:
    - The agent can choose from three actions: go straight, turn right, or turn left.

3. **Reward System**:
    - `+10` for eating food.
    - `-10` for colliding with walls or itself.
    - `0` for other moves.

4. **Training**:
    - The agent uses Q-learning with a neural network to approximate the Q-values.
    - Experience replay is used to store and sample past experiences for training.
    - The model is trained using the Adam optimizer and Mean Squared Error loss.

5. **Playing**:
    - The trained model is loaded and used to make decisions in the game without further training.

## Contributing

Contributions are welcome! If you’d like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push to your branch.
5. Create a pull request.

Please ensure your code follows the project’s coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
