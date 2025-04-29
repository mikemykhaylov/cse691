import copy  # Needed for deep copying environment states
import time  # For basic timing/reporting
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
from numpy import ndarray
import multiprocessing as mp
from functools import partial

# Assuming your heuristic agent file is src.agent.py
from src.agent import choose_action_heuristic, choose_action_within_board
from src.env import SuperTicTacToe3DEnv, _get_winning_lines_3x3x3, print_3d_grid_indices, register_env

_lines = _get_winning_lines_3x3x3()  # Get winning lines once

# Constants (can be fetched from env instance if needed)
PLAYER1 = 1
PLAYER2 = 2
DRAW = 0  # Assuming DRAW=0 in heuristic info['game_winner'] convention


class LookaheadAgent:
    """
    An agent for 3D Super Tic-Tac-Toe that uses N-step lookahead
    with heuristic rollouts to evaluate positions. Uses minimax logic
    on the number of moves required for the agent to win.
    """

    def __init__(self, player_id: int, lookahead_depth: int, env_for_copying: SuperTicTacToe3DEnv,
                 # Pass a sample env instance
                 rng: Optional[np.random.Generator] = None):
        """
        Initializes the Lookahead Agent.

        Args:
            player_id: The player ID this agent represents (1 or 2).
            lookahead_depth: The depth 'n' for the minimax search.
            env_for_copying: A sample instance of the environment used ONLY for deepcopying.
                             The agent will not modify this instance directly.
            rng: A NumPy random number generator for tie-breaking/rollouts.
                 If None, a default one will be created.
        """
        if lookahead_depth < 1:
            raise ValueError("Lookahead depth must be at least 1.")
        self.player_id = player_id
        self.opponent_id = PLAYER1 if player_id == PLAYER2 else PLAYER2
        self.n = lookahead_depth
        self._env_template = env_for_copying  # Keep a template for copying
        self.rng = rng if rng is not None else np.random.default_rng()
        self._lines = _lines  # Use precomputed lines

        # Check if required functions/classes are available
        if _lines is None:
            print("Warning: Winning lines data seems missing.")
        if 'choose_action_heuristic' not in globals() or not callable(choose_action_heuristic):
            raise RuntimeError("Heuristic function 'choose_action_heuristic' not available.")

    def _heuristic_rollout(self, env_state: SuperTicTacToe3DEnv) -> int:
        """
        Plays out a game from the given state using the heuristic agent
        for BOTH players until the game ends.

        Args:
            env_state: A copy of the environment representing the state to start from.

        Returns:
            Number of moves made during the rollout if self.player_id wins.
        """
        # Ensure we're working with a copy that can be modified
        current_env = copy.deepcopy(env_state)
        # We need observation and info to call the heuristic
        obs = current_env._get_obs()
        info = current_env._get_info()
        terminated = current_env._is_terminal
        truncated = False  # Assuming no truncation in the game logic
        up_to_this_point_moves = np.count_nonzero(obs['small_boards'])  # Moves before rollout
        rollout_moves = 0

        while not terminated and not truncated:
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask == 1)[0]

            if len(valid_actions) == 0:
                # Should ideally be caught by env termination, but handle defensively
                # This state is likely a draw if not already terminated
                break

            # Use the heuristic for the *current* player in the simulated env
            action = choose_action_heuristic(obs, info, self._lines, self.rng)

            # Ensure the heuristic chose a valid move (it should!)
            if action_mask[action] == 0:
                print(f"Warning: Heuristic chose invalid action {action} during rollout. Picking random.")
                action = self.rng.choice(valid_actions)

            # Step the copied environment
            obs, reward, terminated, truncated, info = current_env.step(action)
            rollout_moves += 1

            # Optional: Add a safety break for extremely long rollouts (indicative of issues)
            if rollout_moves > (27 * 27):  # Max possible moves
                print("Warning: Rollout exceeded maximum possible game length. Aborting.")
                raise RuntimeError("Rollout exceeded maximum possible game length.")

        # Game ended, evaluate the outcome
        total_moves_taken = up_to_this_point_moves + rollout_moves  # Total moves in the game state

        winner = info.get('game_winner', None)
        MAX_SCORE = 27 ** 2 + 1  # A score higher than any possible move count

        if winner == self.player_id:
            # Higher score for faster win (fewer moves)
            return MAX_SCORE - total_moves_taken
        elif winner == self.opponent_id:
            # Lower score for faster loss (fewer moves)
            return -(MAX_SCORE - total_moves_taken)
        else:  # Draw
            # Neutral or slightly negative score for draw
            return 0  # Or perhaps a small negative value

    def _minimax_search(self, env_state: SuperTicTacToe3DEnv, depth: int, taken_moves: ndarray) -> float:
        """
        Recursive minimax function.

        Args:
            env_state: A copy of the environment state at the current node.
            depth: The current depth in the search tree (starts at 0 or 1).

        Returns:
            The estimated value (minimum moves to win for self.player_id,
            or INF_MOVES for loss/draw) from this state.
        """
        # Check if game ended *before* reaching max depth or rollout
        # Inside _minimax_search:
        if env_state._is_terminal:
            winner = env_state._game_winner
            MAX_SCORE = 27 ** 2 + 1  # Use the same scale
            if winner == self.player_id:
                # Immediate win is best possible outcome
                return MAX_SCORE
            elif winner == self.opponent_id:
                # Immediate loss is worst possible outcome
                return -MAX_SCORE
            else:  # Draw
                return 0

        # Base case: Reached maximum lookahead depth
        if depth == self.n:
            return self._heuristic_rollout(env_state)

        # Recursive step
        current_player = env_state._current_player
        # Need observation and info to get valid moves from the *current* state
        obs = env_state._get_obs()
        info = env_state._get_info()
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        # print(f"LookaheadAgent: Exploring move subsequence {taken_moves} at depth {depth}, current player {current_player}.")

        if len(valid_actions) == 0:
            # Game ends here unexpectedly (should have been caught by _is_terminal)
            # Evaluate based on current state (likely a draw)
            raise RuntimeError("LookaheadAgent._minimax_search called with no valid moves and game is not terminal!")

        # next_large_cell_idx = obs['next_large_cell']
        # if next_large_cell_idx == 27:
        #     playable_large_indices = sorted(list(set(a // 27 for a in valid_actions)))
            # print("LookaheadAgent: Selecting big cell inside minimax using heuristic to reduce search space.")
            # large_board = obs["large_board"]
            # chosen_large_idx_target = choose_action_within_board(large_board.copy(), current_player,
            #                                                      playable_large_indices, _lines, self.rng)
            # valid_actions = valid_actions[np.where(valid_actions // 27 == chosen_large_idx_target)[0]]
            # print(f"LookaheadAgent: Masked valid actions to {valid_actions} based on heuristic choice.")

        # Initialize based on whose turn it is
        if current_player == self.player_id:
            best_value = -float('inf')  # Agent wants to *minimize* moves-to-win
        else:  # Opponent's turn
            best_value = float('inf')  # Opponent wants to *maximize* agent's moves-to-win

        # Explore child nodes
        for action in valid_actions:
            # --- Create a copy of the environment state for the next level ---
            # Using deepcopy is convenient but can be slow.
            # If performance is critical, implement a faster custom copy method
            # for the environment or manually copy relevant state attributes.
            child_env_state = copy.deepcopy(env_state)

            # --- Simulate the move on the copy ---
            try:
                # We only need the state transition, ignore return values here
                child_env_state.step(action)
            except ValueError as e:
                # This might happen if the action mask was somehow incorrect
                # or if the step logic has issues not caught by validation.
                # print(f"Warning: Error stepping copied env with action {action} at depth {depth}: {e}")
                # Assign worst possible value if simulation fails for this branch
                raise RuntimeError(f"Error stepping copied env with action {action} at depth {depth}: {e}")
            else:
                # --- Recursively call minimax ---
                new_taken_moves = np.append(taken_moves, action)
                value = self._minimax_search(child_env_state, depth + 1, new_taken_moves)
                # print(f"LookaheadAgent: Move sequence {new_taken_moves} evaluated to value {value}")

            # --- Update best_value (Minimax logic) ---
            if current_player == self.player_id:
                best_value = max(best_value, value)  # Agent maximizes
            else:  # Opponent's turn
                best_value = min(best_value, value)  # Opponent minimizes

        # print(f"LookaheadAgent: Current player {current_player} after move sequence {taken_moves} - Best value: {best_value}")

        return best_value

    # --- Iterate through possible first moves using multiprocessing ---
    def evaluate_action(self, action, template_env, observation, action_mask, info):
        """Helper function to evaluate a single action in a separate process"""
        try:
            # Create a copy of the initial state
            initial_env_copy = copy.deepcopy(template_env)
            # Manually set the state of the copy to match the current observation/info
            initial_env_copy._small_boards = observation['small_boards'].copy()
            initial_env_copy._large_board = observation['large_board'].copy()
            initial_env_copy._current_player = observation['current_player']
            initial_env_copy._next_large_cell_idx = observation['next_large_cell']
            initial_env_copy._action_mask = action_mask.copy()
            initial_env_copy._game_winner = info['game_winner']
            initial_env_copy._is_terminal = info['game_winner'] is not None

            # Simulate the first move
            _, _, _, _, next_info = initial_env_copy.step(action)

            # Start the minimax search from the state after the first move
            value = self._minimax_search(initial_env_copy, depth=1, taken_moves=np.array([action]))
            # print(f"LookaheadAgent: Move {action} evaluated to value: {value}")
            return action, value
        except Exception as e:
            print(f"Warning: Error processing action {action}: {e}")
            return action, float('inf')  # Return worst possible value on error

    def choose_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Chooses the best action based on N-step lookahead minimax search.

        Args:
            observation: The current environment observation.
            info: The current environment info dictionary (must contain 'action_mask').

        Returns:
            The chosen action index.
        """
        start_time = time.time()
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask == 1)[0]
        # print(f"LookaheadAgent: Valid actions: {valid_actions}")

        if len(valid_actions) == 0:
            raise RuntimeError("LookaheadAgent.choose_action called with no valid moves!")
        if len(valid_actions) == 1:
            print("LookaheadAgent: Only one valid move, selecting it.")
            return valid_actions[0]  # No need for search

        # if first move, do random selection
        if np.all(observation['small_boards'] == 0):
            print("LookaheadAgent: First move, selecting center cell.")
            return 13 * 27 + 13  # Center cell in the first large board

        best_action = -1
        best_value = -float('inf')  # Agent maximizes

        # --- Quickly select the big cell if needed ---
        # This is suboptimal solution to the case when all big cells are available
        # to play and we have more than the standard <= 27 moves to play. To narrow
        # our search, we can quickly select the big cell using the underlying heuristic
        # This is a heuristic choice, not a minimax search, but it helps reduce the search space
        # next_large_cell_idx = observation['next_large_cell']
        # if next_large_cell_idx == 27:
        #     playable_large_indices = sorted(list(set(a // 27 for a in valid_actions)))
            # print("LookaheadAgent: Selecting big cell using heuristic to reduce search space.")
            # large_board = observation["large_board"]
            # chosen_large_idx_target = choose_action_within_board(large_board.copy(), current_player,
            #                                                      playable_large_indices, _lines, rng)
            # valid_actions = valid_actions[np.where(valid_actions // 27 == chosen_large_idx_target)[0]]
            # print(f"LookaheadAgent: Masked valid actions to {valid_actions} based on heuristic choice.")

        # Create a pool of workers
        with mp.Pool(processes=min(mp.cpu_count(), len(valid_actions))) as pool:
            # Create a partial function with fixed arguments
            eval_func = partial(self.evaluate_action, template_env=self._env_template, observation=observation,
                action_mask=action_mask, info=info)

            # Submit all actions to be evaluated in parallel
            results = pool.map(eval_func, valid_actions)

        # Process results to find the best action
        for action, value in results:
            # We want the action that leads to the maximum value (best outcome)
            if value > best_value:
                best_value = value
                best_action = action
            elif value == best_value:
                # Tie-breaking remains the same logic (random choice)
                if best_action == -1 or self.rng.random() < 0.5:
                    best_action = action

        if best_action == -1:
            print("Warning: All initial moves evaluated to negative infinity or failed. Choosing random.")
            best_action = self.rng.choice(valid_actions)

        end_time = time.time()
        print(f"LookaheadAgent: Search complete in {end_time - start_time:.2f}s. Best value: {best_value}")

        # Fallback if no move seems good (all lead to infinity)
        if best_action == -1:
            print("Warning: All initial moves evaluated to infinity. Choosing random valid move.")
            best_action = self.rng.choice(valid_actions)

        return best_action


# --- Example Usage ---
if __name__ == '__main__':
    register_env()

    print_3d_grid_indices()  # Print the index map for reference
    print("\n--- Running Sample Episode (Lookahead Agent vs Heuristic Agent) ---")

    # Create the base environment instance
    try:
        # Need render_mode=None for copying, create a separate one for rendering if needed
        env_for_logic = SuperTicTacToe3DEnv(render_mode=None)
        # Optional: Create a second env just for rendering the final output
        render_env = gym.make('SuperTicTacToe3D-v0', render_mode='human')  # print("Environments created.")
    except gym.error.Error as e:
        print(f"Failed to create environment: {e}")
        print("Ensure the environment is registered correctly.")
        exit()
    except NameError:
        print("SuperTicTacToe3DEnv class not found. Imports failed.")
        exit()

    # Use a fixed seed for reproducibility
    seed = 123
    print(f"Seed: {seed}")
    observation, info = env_for_logic.reset(seed=seed)
    # Also reset the render env to sync state initially
    render_env.reset(seed=seed)

    # Use the environment's RNG for agents
    rng = env_for_logic.np_random

    # Create Agents
    LOOKAHEAD_DEPTH = 2  # Set the desired lookahead depth (e.g., 2 or 3)
    try:
        player1_agent = LookaheadAgent(player_id=PLAYER1, lookahead_depth=LOOKAHEAD_DEPTH,
                                       env_for_copying=env_for_logic, rng=rng)
        # Player 2 uses the heuristic agent (needs RNG)
        # We don't instantiate it as a class, just use the function
        player2_agent_func = choose_action_heuristic
        print(f"Agents created: P1=Lookahead(n={LOOKAHEAD_DEPTH}), P2=Heuristic.")

    except Exception as e:
        print(f"Failed to create agents: {e}")
        exit()

    terminated = False
    truncated = False
    step_count = 0
    max_steps = 729 + 10  # Safety limit

    while not terminated and not truncated and step_count < max_steps:
        current_player = observation['current_player']
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            print("No valid actions available! Environment state likely terminal.")
            # Double check terminal status from env
            terminated = env_for_logic._is_terminal
            info = env_for_logic._get_info()  # Get final info
            break

        print(f"\n--- Step {step_count + 1} ---")
        # render_env.render() # Render the state *before* the move

        if current_player == PLAYER1:
            print(f"Player {current_player} (Lookahead) choosing move...")
            action = player1_agent.choose_action(observation, info)
            agent_type = f"Lookahead(n={LOOKAHEAD_DEPTH})"
        else:  # Player 2
            print(f"Player {current_player} (Heuristic) choosing move...")
            # Call the heuristic function directly
            action = player2_agent_func(observation, info, _lines, rng)
            agent_type = "Heuristic"

        # --- Validate chosen action (important sanity check) ---
        if action not in valid_actions:
            print(f"CRITICAL ERROR: Agent {agent_type} chose invalid action {action}!")
            print(f"Valid actions were: {valid_actions}")
            print("Choosing random valid action instead.")
            action = rng.choice(valid_actions)  # Fallback

        large_act_idx = action // 27
        small_act_idx = action % 27
        print(f"Agent: {agent_type} (Player {current_player})")
        print(f"Plays action {action} (Large: {large_act_idx}, Small: {small_act_idx})")

        # --- Step both environments to keep them in sync ---
        observation, reward, terminated, truncated, info = env_for_logic.step(action)
        render_obs, render_reward, render_terminated, render_truncated, render_info = render_env.step(
            action)  # Render env also steps

        step_count += 1
        print(f"Reward (for P{current_player}): {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")

    print("\n--- Game Over ---")
    # Render the final state
    render_env.render()

    # Get winner from the final info dictionary
    final_info = info if terminated or truncated else env_for_logic._get_info()
    winner = final_info.get('game_winner', None)

    if winner == PLAYER1:
        print(f"Winner: Player 1 (X) - Lookahead(n={LOOKAHEAD_DEPTH})")
    elif winner == PLAYER2:
        print("Winner: Player 2 (O) - Heuristic")
    elif winner == DRAW:
        print("Result: Draw")
    else:
        print("Result: Unknown or Ongoing (check max_steps)")

    print(f"Total steps: {step_count}")

    env_for_logic.close()
    render_env.close()
