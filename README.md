# Snake AI Game

1. **Introduction**

   This project implements a Snake game with an AI agent trained using Q-learning. The agent learns to play the Snake game by making decisions on whether to go straight, turn left, or turn right based on its current state.

2. **Features**

   - Snake game with Pygame: Classic Snake game environment.
   - Q-learning AI: An agent trained using Q-learning to make decisions during the game.
   - Neural Network: Deep Q-network (DQN) model implemented using PyTorch.
   - Training: The agent is trained through multiple game iterations to improve its decision-making abilities.
   - Visualizations: Graphical representation of game scores and mean scores during training.

3. **Requirements**

   - Python 3.x
   - Pygame
   - PyTorch
   - Matplotlib
   - IPython

4. **How to Run the Game**

   - **Install Python:**
     If you don't have Python installed, you can download it from [Python Downloads](https://www.python.org/downloads/).

   - **Clone the Repository:**
     ```bash
     git clone https://github.com/ReageNkoana/SnakeGameAI.git
     cd SnakeGameAI
     ```

   - **Create and Activate a Virtual Environment (Optional but Recommended):**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```

   - **Install Additional Libraries (Matplotlib and IPython):**
     ```bash
     pip install matplotlib ipython
     ```

   - **Install Additional Libraries (PyTorch):**
     ```bash
     pip install torch
     ```

   - **Run the Training Script:**
     ```bash
     python agent.py
     ```

5. **Usage**

   - Run `agent.py` to train the Snake AI. The training progress will be displayed, and you can observe the agent's performance.
   - Adjust hyperparameters in the code to experiment with different configurations.

6. **Project Structure**

   - `agent.py`: Main script for training the Snake AI.
   - `snakegameAI.py`: Snake game environment.
   - `model.py`: Definition of the neural network model.
   - `helper.py`: Helper functions for plotting scores.

7. Credits

- The game design is inspired by classic Snake games.

8. **Author**

   Reage Nkoana

9. **Acknowledgments**

   - Pygame Community: [https://www.pygame.org/contribute.html](https://www.pygame.org/contribute.html)
   - PyTorch Team: [https://pytorch.org/](https://pytorch.org/)
   - Matplotlib Team: [https://matplotlib.org/stable/users/index.html](https://matplotlib.org/stable/users/index.html)
