import time
import random
import pytesseract
import pyautogui
import numpy as np
from PIL import Image

# If you installed Tesseract in a custom location, set the path here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# RL "Model" placeholders
# -----------------------------
class SimpleModel:
    """
    A mock model that stores experiences.
    Replace with your actual RL approach (TensorFlow, PyTorch, or stable-baselines).
    """
    def __init__(self):
        self.experiences = []

    def remember(self, state, action, reward, next_state):
        """
        Store experience in a buffer (list) for training.
        """
        self.experiences.append((state, action, reward, next_state))

    def train(self):
        """
        Implement your training logic here (e.g., Q-learning update, neural net backprop, etc.).
        """
        # Just print how many experiences we have
        print(f"Training on {len(self.experiences)} experiences...")
        # For real RL, you'd sample from memory, compute loss, update weights, etc.

    def save(self, filename="model_checkpoint.pkl"):
        """
        Save the model to disk (pickle, or torch.save, or whichever).
        """
        print(f"Model saved to {filename}")

    def load(self, filename="model_checkpoint.pkl"):
        """
        Load the model from disk.
        """
        print(f"Model loaded from {filename}")

# Instantiate our mock model
model = SimpleModel()

# -----------------------------
# Screen / OCR helpers
# -----------------------------
def check_kmh_present(region=(100, 100, 200, 50)):
    """
    Take a screenshot of a specified region, run OCR, 
    and check if 'km/h' is present in the recognized text.
    
    Args:
        region: A tuple (left, top, width, height) specifying the region.

    Returns:
        bool: True if 'km/h' was detected, False otherwise.
    """

    # region=(x, y, width, height)
    screenshot = pyautogui.screenshot(region=region)
    # Convert image for OCR
    text = pytesseract.image_to_string(screenshot)

    # We can do a simple check if 'km/h' is in the text
    # (and handle small or empty text)
    if len(text.strip()) < 2:
        # The text is too small/empty to be useful
        return False

    # Convert to lower and check
    return "km/h" in text.lower()


def get_current_speed(region=(100, 100, 200, 50)):
    """
    OPTIONAL: If you want to parse the numeric value of speed (e.g., "199 km/h" -> 199).
    We'll do a naive parse from recognized text.
    
    Return:
        float: current speed if parsed, else 0.
    """
    screenshot = pyautogui.screenshot(region=region)
    text = pytesseract.image_to_string(screenshot)

    # Let's just search for digits before "km/h"
    text = text.lower()
    if "km/h" in text:
        # E.g., text might be "199 km/h\n"
        before_kmh = text.split("km/h")[0]
        # Keep only digits and punctuation in that portion
        # We'll do something naive like:
        filtered = "".join(ch for ch in before_kmh if ch.isdigit() or ch == '.')
        try:
            return float(filtered)
        except ValueError:
            return 0.0
    return 0.0


# -----------------------------
# Action / Environment logic
# -----------------------------
def press_key(action):
    """
    Presses one of the arrow keys based on action ID.
    0 -> Down, 1 -> Left, 2 -> Right, etc.
    """
    if action == 0:
        pyautogui.keyDown('down')
        time.sleep(0.05)
        pyautogui.keyUp('down')
    elif action == 1:
        pyautogui.keyDown('left')
        time.sleep(0.05)
        pyautogui.keyUp('left')
    elif action == 2:
        pyautogui.keyDown('right')
        time.sleep(0.05)
        pyautogui.keyUp('right')
    # add more actions if needed (e.g., nitro, up arrow, etc.)

def compute_reward(speed):
    """
    Example of a reward function. 
    Just some placeholder logic: 
    - If speed is high, reward is higher.
    - If speed is 0, reward is negative.
    """
    if speed <= 0:
        return -1.0
    else:
        return speed / 100.0  # scale it down so it's not too big


def get_state():
    """
    Return whatever 'state' representation you want for your RL.
    For example, speed alone, or a screenshot, etc.
    We'll just do speed as a float for simplicity.
    """
    current_speed = get_current_speed(region=(1000, 850, 200, 50))  # adapt coords as needed
    return current_speed


# -----------------------------
# Main training loop
# -----------------------------
def train_on_asphalt8(
    screenshot_region=(1000, 850, 200, 50), 
    check_interval=0.5, 
    max_episodes=10
):
    """
    Main function that:
      1. Checks if 'km/h' is visible. If not, we pause or break.
      2. Takes random actions (left/right/down).
      3. Computes reward, saves to model, trains periodically.
      4. Repeats for a fixed number of 'episodes' (or until user stops).
    """

    # Try to load existing model knowledge (if you want):
    # model.load(filename="model_checkpoint.pkl")

    episode = 0
    while episode < max_episodes:
        print(f"Starting Episode {episode+1}/{max_episodes}...")

        # Check if game is showing 'km/h'
        if not check_kmh_present(region=screenshot_region):
            print("km/h not detected. Waiting...")
            time.sleep(check_interval)
            continue  # or break if you want to stop training entirely

        # Once we see km/h, we assume the race is on:
        print("km/h detected. Training started.")

        # Simple example loop for each episode
        steps = 0
        max_steps_in_episode = 50  # arbitrary
        while steps < max_steps_in_episode:
            # Get current state
            state = get_state()

            # Decide on an action
            # (Here: random. In real RL you'd use policy from model.)
            action = random.choice([0, 1, 2])

            # Press the key
            press_key(action)

            # Small delay to let the game respond
            time.sleep(0.2)

            # Get next state
            next_state = get_state()
            # Compute reward
            reward = compute_reward(next_state)

            # Store experience
            model.remember(state, action, reward, next_state)

            # Check if km/h is still visible
            if not check_kmh_present(region=screenshot_region):
                print("km/h disappeared mid-race. Possibly paused or ended.")
                break  # exit from this episode loop

            steps += 1

        # After finishing an episode, let's train and optionally save:
        model.train()

        # Save the model every episode (or less often if you prefer):
        model.save(filename="model_checkpoint.pkl")

        episode += 1

        # Sleep a bit before next episode
        time.sleep(2)

    print("Training finished.")

# -----------------------------
# Run the script
# -----------------------------
if __name__ == "__main__":
    # Adjust these coordinates to match where "km/h" appears in your screen.
    # Also adjust for your actual resolution or monitor setup.
    # region = (x, y, width, height)
    kmh_region = (1000, 850, 200, 50)  # EXAMPLE coordinates

    train_on_asphalt8(
        screenshot_region=kmh_region, 
        check_interval=2.0, 
        max_episodes=5
    )
