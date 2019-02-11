import numpy as np
from holdem.utils import action_table, player_table, community_table

class Decoder():
    n_actions = 3 + 23

    betsizes = np.array([1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 1, 4/3, 5/3, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100])

    action_to_move = {0:action_table.CHECK,1:action_table.CALL,2:action_table.RAISE,3:action_table.FOLD}
    for i in range(4,n_actions + 1):
        action_to_move[i] = action_table.RAISE

    def decode(self, action, n_seats, minimum_raise, maximum_raise, pot_size, amount_to_call, debug=False):
        move = self.action_to_move[action]
        if amount_to_call == 0:
            if move in [action_table.CALL, action_table.FOLD]:
                move = action_table.CHECK # if np.random.random() < 0.5 else RAISE # Let betting be decided only with action > FOLD
        else:
            if move == action_table.CHECK:
                move = action_table.FOLD
        amount = minimum_raise
        if action > action_table.FOLD:
            amount_index = action - action_table.FOLD
            if amount_index == 0:
                amount = minimum_raise
            else:
                # Relative raise size
                amount = self.betsizes[amount_index - 1] * pot_size
                # Move size to legal range
        if move == action_table.RAISE and minimum_raise >= maximum_raise:
            if amount_to_call > 0:
                move = action_table.CALL
            else:
                move = action_table.CHECK
        amount = max(min(amount, maximum_raise), min(minimum_raise, maximum_raise))
        if debug:
            print('\n')
            print('maxraise', maximum_raise,'minraise', minimum_raise)
            print(move, amount)
        return [move, amount]