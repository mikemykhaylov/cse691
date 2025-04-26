import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from src.env import _get_winning_lines_3x3x3, register_env, print_3d_grid_indices

# --- Keep the Environment Code from the previous response ---
# (Including SuperTicTacToe3DEnv class, _get_winning_lines_3x3x3, register_env)
# ... (omitted for brevity - assume it's defined above this code) ...
# Make sure _get_winning_lines_3x3x3 is accessible or redefined here
_winning_lines = _get_winning_lines_3x3x3()

# --- Heuristic Agent Strategy ---

# Constants from the Env (or pass env/player constants)
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
DRAW = 3
CENTER_INDEX_3D = 13 # The index of the center cell (1,1,1) in a 3x3x3 grid (1 + 1*3 + 1*9 = 13)

def _check_lines(board_1d: np.ndarray, player: int, lines: List[Tuple[int, ...]]) -> List[Tuple[int, int]]:
    """
    Finds lines where the player has exactly two marks and the third cell is empty.
    Args:
        board_1d: The 1D representation of the board (size 27).
        player: The player (1 or 2) to check for.
        lines: The list of winning lines.
    Returns:
        List of tuples: (empty_cell_index, count) where count is 2 (meaning a winning opportunity).
                       Returns empty list if no such lines found.
    """
    potential_wins = []
    for line in lines:
        player_marks = 0
        empty_cell = -1
        valid_line = True
        for cell_idx in line:
            if board_1d[cell_idx] == player:
                player_marks += 1
            elif board_1d[cell_idx] == EMPTY:
                if empty_cell != -1: # More than one empty cell
                   valid_line = False
                   break
                empty_cell = cell_idx
            else: # Opponent's mark is in the line
                valid_line = False
                break

        if valid_line and player_marks == 2 and empty_cell != -1:
            potential_wins.append((empty_cell, 2)) # Index of cell to play for win

    return potential_wins

def check_immediate_win(board_1d: np.ndarray, player: int, lines: List[Tuple[int, ...]]) -> List[int]:
    """Returns a list of moves (indices) that immediately win for the player."""
    wins = _check_lines(board_1d, player, lines)
    return [move[0] for move in wins] # Return only the indices

def check_block(board_1d: np.ndarray, player: int, lines: List[Tuple[int, ...]]) -> List[int]:
    """Returns a list of moves (indices) that block the opponent's immediate win."""
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2
    blocks = _check_lines(board_1d, opponent, lines)
    return [move[0] for move in blocks] # Return only the indices

def check_fork(board_1d: np.ndarray, player: int, lines: List[Tuple[int, ...]]) -> List[int]:
    """
    Returns a list of moves (indices) that create a fork (>= 2 immediate win threats).
    """
    forking_moves = []
    empty_cells = np.where(board_1d == EMPTY)[0]

    for move_idx in empty_cells:
        # Simulate making the move
        temp_board = board_1d.copy()
        temp_board[move_idx] = player
        # Check how many winning opportunities are created *by this move*
        potential_wins_after_move = _check_lines(temp_board, player, lines)
        if len(potential_wins_after_move) >= 2:
            forking_moves.append(move_idx)

    return forking_moves

def get_center_move(board_1d: np.ndarray) -> List[int]:
    """Returns [center_index] if the center is available, else []."""
    if board_1d[CENTER_INDEX_3D] == EMPTY:
        return [CENTER_INDEX_3D]
    return []

def choose_action_within_board(
    small_board_1d: np.ndarray,
    player: int,
    available_moves_relative: List[int], # Indices 0-26 within this small board
    lines: List[Tuple[int, ...]],
    rng: np.random.Generator # Pass RNG for reproducibility
) -> Optional[int]:
    """
    Applies Case 1 logic to choose a move within a specific small board.
    Returns the chosen relative index (0-26) or None if no valid moves.
    """
    if not available_moves_relative:
        return None # Should not happen if called correctly

    available_set = set(available_moves_relative)

    # 1. Check for winning move
    win_moves = check_immediate_win(small_board_1d, player, lines)
    valid_win_moves = [m for m in win_moves if m in available_set]
    if valid_win_moves:
        return rng.choice(valid_win_moves) # Pick one if multiple

    # 2. Check for blocking move
    block_moves = check_block(small_board_1d, player, lines)
    valid_block_moves = [m for m in block_moves if m in available_set]
    if valid_block_moves:
        return rng.choice(valid_block_moves)

    # 3. Check for fork move
    fork_moves = check_fork(small_board_1d, player, lines)
    valid_fork_moves = [m for m in fork_moves if m in available_set]
    if valid_fork_moves:
        return rng.choice(valid_fork_moves)

    # 4. Check for center move
    center_move = get_center_move(small_board_1d)
    valid_center_move = [m for m in center_move if m in available_set]
    if valid_center_move:
        return valid_center_move[0] # Only one center

    # 5. Otherwise, choose randomly from available moves
    return rng.choice(available_moves_relative)


def choose_action_heuristic(
    observation: Dict[str, Any],
    info: Dict[str, Any],
    lines: List[Tuple[int, ...]],
    rng: np.random.Generator # Pass RNG for reproducibility
) -> int:
    """
    Chooses an action based on the heuristic strategy.
    """
    small_boards = observation["small_boards"]
    large_board = observation["large_board"]
    current_player = observation["current_player"]
    next_large_cell = observation["next_large_cell"]
    action_mask = info["action_mask"]

    valid_global_actions = np.where(action_mask == 1)[0]
    if len(valid_global_actions) == 0:
        raise ValueError("Heuristic agent called with no valid actions!") # Should be terminal

    target_large_idx = -1

    # --- Determine Target Large Cell ---
    if next_large_cell == 27: # Case 2: Player can play in any big cell
        playable_large_indices = sorted(list(set(a // 27 for a in valid_global_actions)))

        # 1. Check if any move wins the *large* board
        winning_large_moves = [] # Store tuples (large_idx, small_idx)
        for l_idx in playable_large_indices:
             # Check if playing in this large cell *could* win the game
             # This requires finding a small move within l_idx that wins l_idx
             small_board = small_boards[l_idx]
             potential_small_wins = check_immediate_win(small_board, current_player, lines)
             # Filter against actual available small moves in this large cell
             available_small_in_large = [a % 27 for a in valid_global_actions if a // 27 == l_idx]
             actual_small_wins = [sw for sw in potential_small_wins if sw in available_small_in_large]

             if actual_small_wins:
                  # Found move(s) in small board l_idx that win l_idx
                  # Now check if winning l_idx wins the *overall game*
                  temp_large_board = large_board.copy()
                  temp_large_board[l_idx] = current_player # Simulate winning the large cell
                  if check_immediate_win(temp_large_board, current_player, lines):
                      # Yes, winning l_idx wins the game. Store all small moves that do this.
                      for small_win_idx in actual_small_wins:
                          winning_large_moves.append((l_idx, small_win_idx))


        if winning_large_moves:
            # Found one or more moves that win the entire game.
            # If multiple, apply Case 1 logic to the *large board* to choose *which* large cell to target.
            # We only need to consider the large cells involved in the winning moves.
            involved_large_indices = sorted(list(set(m[0] for m in winning_large_moves)))

            if len(involved_large_indices) == 1:
                 target_large_idx = involved_large_indices[0]
                 # We already know the small winning moves in this cell
                 # Need to choose one small move using Case 1 within that small board
                 small_board = small_boards[target_large_idx]
                 available_small_in_target = [m[1] for m in winning_large_moves if m[0] == target_large_idx]
                 chosen_small_idx = choose_action_within_board(small_board, current_player, available_small_in_target, lines, rng)
                 return target_large_idx * 27 + chosen_small_idx
            else:
                 # Apply Case 1 logic to the large board to pick the best large cell target
                 # The "available moves" here are the large cell indices that lead to a win.
                 chosen_large_idx_target = choose_action_within_board(large_board.copy(), current_player, involved_large_indices, lines, rng)

                 # Now that we've picked the large cell, pick the best small move *within that cell*
                 # that actually wins the large cell (and thus the game).
                 small_board = small_boards[chosen_large_idx_target]
                 winning_small_moves_in_chosen_large = [m[1] for m in winning_large_moves if m[0] == chosen_large_idx_target]
                 chosen_small_idx = choose_action_within_board(small_board, current_player, winning_small_moves_in_chosen_large, lines, rng)
                 return chosen_large_idx_target * 27 + chosen_small_idx

        else:
            # 2. No immediate game win available. Pick a random available large cell.
            target_large_idx = rng.choice(playable_large_indices)
            # Fall through to Case 1 logic below

    else: # Case 1: Forced to play in a specific big cell
        target_large_idx = next_large_cell
        # Sanity check: ensure there are valid moves in this required cell
        if not any(a // 27 == target_large_idx for a in valid_global_actions):
             # This indicates an issue, maybe forced into a full/won board?
             # The env logic should prevent this by setting next_large_cell=27.
             # If it happens, fall back to choosing from any valid move.
             print(f"Warning: Heuristic agent forced into cell {target_large_idx} but no valid moves found there. Choosing randomly globally.")
             return rng.choice(valid_global_actions)


    # --- Apply Case 1 logic within the chosen/forced large cell ---
    small_board_to_play = small_boards[target_large_idx]
    available_small_moves_relative = [a % 27 for a in valid_global_actions if a // 27 == target_large_idx]

    if not available_small_moves_relative:
        # Should not happen if logic above is correct and action mask isn't all zero
        print(f"Error: No available small moves found in target large cell {target_large_idx}. Choosing randomly globally.")
        return rng.choice(valid_global_actions)


    chosen_small_idx = choose_action_within_board(
        small_board_to_play,
        current_player,
        available_small_moves_relative,
        lines,
        rng
    )

    if chosen_small_idx is None:
         # Should not happen if available_small_moves_relative was not empty
         print(f"Error: choose_action_within_board returned None for cell {target_large_idx}. Choosing randomly globally.")
         return rng.choice(valid_global_actions)


    # Convert relative small index back to global action index
    final_action = target_large_idx * 27 + chosen_small_idx

    # Final sanity check: ensure the chosen action is in the original mask
    if action_mask[final_action] == 0:
        print(f"CRITICAL Error: Heuristic chose invalid action {final_action} (Large: {target_large_idx}, Small: {chosen_small_idx}). Available: {valid_global_actions}. Choosing randomly.")
        # This indicates a bug in the heuristic logic or helper functions
        return rng.choice(valid_global_actions)

    return final_action


# --- Example Usage with Heuristic Agent ---
if __name__ == '__main__':
    # Ensure environment is registered if not already
    register_env()
    print_3d_grid_indices() # Print the index map for reference

    print("\n--- Running Sample Episode (Heuristic Agent vs Random Agent) ---")
    # Use the registered ID
    env = gym.make('SuperTicTacToe3D-v0', render_mode='human')

    # Use a fixed seed for reproducibility during testing
    seed = 42
    obs, info = env.reset(seed=seed)
    # Use the environment's RNG for agent choices as well
    rng = env.np_random


    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        action_mask = info['action_mask']
        current_player = obs['current_player']

        # Choose action based on player
        if current_player == PLAYER1:
            # Player 1 uses the heuristic strategy
            action = choose_action_heuristic(obs, info, _winning_lines, rng)
            agent_type = "Heuristic (P1)"
        else:
            # Player 2 uses the random strategy (from original example)
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) == 0:
                print("No valid actions available for Random Agent! Should be terminal.")
                break
            action = rng.choice(valid_actions) # Use env's RNG
            agent_type = "Random (P2)"


        print(f"\n--- Step {step_count + 1} ---")
        large_act_idx = action // 27
        small_act_idx = action % 27
        print(f"Agent: {agent_type} (Player {current_player})")
        print(f"Plays action {action} (Large Cell: {large_act_idx} ({large_act_idx%3}, {(large_act_idx//3)%3}, {large_act_idx//9}), Small Cell: {small_act_idx} ({small_act_idx%3}, {(small_act_idx//3)%3}, {small_act_idx//9}))")

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        print(f"Reward (to player who moved): {reward}") # Reward is +/-1 for the player who just moved if game ends
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        # Render is called inside step when render_mode='human'

        if terminated:
            print("\n--- Game Over ---")
            winner = info['game_winner']
            if winner == 1: print("Winner: Player 1 (X) - Heuristic")
            elif winner == 2: print("Winner: Player 2 (O) - Random")
            elif winner == 0: print("Result: Draw")
            else: print("Result: Unknown (Error?)")
            print(f"Total steps: {step_count}")

    env.close()
