import gym
from gym import spaces

# Convert MultiDiscrete to Discrete so Keras-RL Agents will work
class MD2D_ActionWrapper(gym.ActionWrapper):
  def __init__(self, env, buttons):
    super(MD2D_ActionWrapper, self).__init__(env)
    self.action_space = spaces.Discrete(len(buttons))
    self.buttons = buttons

  def action(self, action):
    return self.buttons.get(action)

  def reverse_action(self, action):
    for k in self.buttons.keys():
      if(self.buttons[k] == action):
        return self.buttons[k]
    return 0

"""
Buttons must be passed to MD2D_ActionWrapper as a dictionary!

EXAMPLE:

  buttons = {
     0: [0, 0, 0, 0, 0, 0],  # Do Nothing
     1: [1, 0, 0, 0, 0, 0],  # Up
     2: [0, 1, 0, 0, 0, 0],  # Left
     3: [0, 0, 1, 0, 0, 0],  # Down
     4: [0, 0, 0, 1, 0, 0],  # Right
     5: [0, 0, 0, 0, 1, 0],  # A
     6: [0, 0, 0, 0, 0, 1],  # B
     7: [0, 0, 0, 0, 1, 1],  # A + B
     8: [1, 0, 0, 1, 0, 0],  # Up    + Right
     9: [1, 1, 0, 0, 0, 0],  # Up    + Left
    10: [0, 0, 1, 1, 0, 0],  # Down  + Right
    11: [0, 1, 1, 0, 0, 0],  # Down  + Left
    12: [1, 0, 0, 0, 1, 0],  # Up    + A
    13: [0, 1, 0, 0, 1, 0],  # Left  + A
    14: [0, 0, 1, 0, 1, 0],  # Down  + A
    15: [0, 0, 0, 1, 1, 0],  # Right + A
    16: [1, 0, 0, 0, 0, 1],  # Up    + B
    17: [0, 1, 0, 0, 0, 1],  # Left  + B
    18: [0, 0, 1, 0, 0, 1],  # Down  + B
    19: [0, 0, 0, 1, 0, 1],  # Right + B
    20: [1, 0, 0, 0, 1, 1],  # Up    + A+B
    21: [0, 1, 0, 0, 1, 1],  # Left  + A+B
    22: [0, 0, 1, 0, 1, 1],  # Down  + A+B
    23: [0, 0, 0, 1, 1, 1],  # Right + A+B
  }
"""
