import copy  # Needed for deep copying environment states
import time  # For basic timing/reporting
from typing import Dict, Any, Optional, Tuple

import gymnasium as gym
import numpy as np

# --- Assumed Imports (Place these in your actual file structure) ---
try:
    # Assuming your environment file is src/env.py
    from src.env import SuperTicTacToe3DEnv, _get_winning_lines_3x3x3
    # Assuming your heuristic agent file is src.agent.py
    from src.agent import choose_action_heuristic

    _lines = _get_winning_lines_3x3x3()  # Get winning lines once
    print("Successfully imported environment and heuristic agent.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'src/env.py' and 'src/agent.py' exist and contain")
    print("SuperTicTacToe3DEnv, _get_winning_lines_3x3x3, and choose_action_heuristic.")


    # Define dummy functions/classes if imports fail, so the rest can be parsed
    class SuperTicTacToe3DEnv:
        pass


    def _get_winning_lines_3x3x3():
        return []


    def choose_action_heuristic(*args, **kwargs):
        return 0


    _lines = []  # Exit or raise an error if essential components are missing  # raise ImportError("Could not load necessary environment/agent components.") from e

# Constants (can be fetched from env instance if needed)
PLAYER1 = 1
PLAYER2 = 2
DRAW = 0  # Assuming DRAW=0 in heuristic info['game_winner'] convention
INF_MOVES = float('inf')  # Value for losing/drawing


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
        if not _lines:
            print("Warning: Winning lines data seems missing.")
        if 'choose_action_heuristic' not in globals() or not callable(choose_action_heuristic):
            raise RuntimeError("Heuristic function 'choose_action_heuristic' not available.")

    def _heuristic_rollout(self, env_state: SuperTicTacToe3DEnv) -> float:
        """
        Plays out a game from the given state using the heuristic agent
        for BOTH players until the game ends.

        Args:
            env_state: A copy of the environment representing the state to start from.

        Returns:
            Number of moves made during the rollout if self.player_id wins.
            float('inf') if self.player_id loses or the game is a draw.
        """
        # Ensure we're working with a copy that can be modified
        current_env = copy.deepcopy(env_state)
        # We need observation and info to call the heuristic
        obs = current_env._get_obs()
        info = current_env._get_info()
        terminated = current_env._is_terminal
        truncated = False  # Assuming no truncation in the game logic
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
                return INF_MOVES  # Treat as unfavorable outcome

        # Game ended, evaluate the outcome
        winner = info.get('game_winner', None)  # Use .get for safety

        if winner == self.player_id:
            return float(rollout_moves)
        else:  # Loss or Draw
            return INF_MOVES

    def _minimax_search(self, env_state: SuperTicTacToe3DEnv, depth: int) -> float:
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
        if env_state._is_terminal:
            winner = env_state._game_winner
            if winner == self.player_id:
                return 0.0  # Won immediately (0 extra moves)
            else:
                return INF_MOVES  # Lost or drew immediately

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

        if len(valid_actions) == 0:
            # Game ends here unexpectedly (should have been caught by _is_terminal)
            # Evaluate based on current state (likely a draw)
            return INF_MOVES

        # Initialize based on whose turn it is
        if current_player == self.player_id:
            best_value = INF_MOVES  # Agent wants to minimize moves-to-win
        else:  # Opponent's turn
            best_value = 0.0  # Opponent wants to *maximize* agent's moves-to-win

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
                print(f"Warning: Error stepping copied env with action {action} at depth {depth}: {e}")
                # Assign worst possible value if simulation fails for this branch
                value = INF_MOVES if current_player == self.player_id else 0.0
            else:
                # --- Recursively call minimax ---
                value = self._minimax_search(child_env_state, depth + 1)

            # --- Update best_value (Minimax logic) ---
            if current_player == self.player_id:
                best_value = min(best_value, value)  # Agent minimizes
            else:  # Opponent's turn
                best_value = max(best_value, value)  # Opponent maximizes agent's cost

            # Optional: Alpha-Beta Pruning could be added here for efficiency  # but requires passing alpha and beta values down the recursion.

        return best_value

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

        if len(valid_actions) == 0:
            raise RuntimeError("LookaheadAgent.choose_action called with no valid moves!")
        if len(valid_actions) == 1:
            print("LookaheadAgent: Only one valid move, selecting it.")
            return valid_actions[0]  # No need for search

        best_action = -1
        best_value = INF_MOVES  # Initialize to worst case for the agent

        # --- Iterate through possible first moves ---
        for action in valid_actions:
            # --- Create a copy of the *initial* state for each first move ---
            # This uses the template env stored during init, then applies current state
            # This is safer than copying the potentially modified 'real' env instance
            initial_env_copy = copy.deepcopy(self._env_template)
            # Manually set the state of the copy to match the current observation/info
            initial_env_copy._small_boards = observation['small_boards'].copy()
            initial_env_copy._large_board = observation['large_board'].copy()
            initial_env_copy._current_player = observation['current_player']
            initial_env_copy._next_large_cell_idx = observation['next_large_cell']
            # Recompute mask and terminal state based on copied state (important!)
            initial_env_copy._action_mask = initial_env_copy._compute_action_mask()  # Need access or reimplement
            initial_env_copy._is_terminal, initial_env_copy._game_winner = self._check_terminal_status(
                initial_env_copy)  # Helper needed

            # --- Simulate the first move on the copy ---
            try:
                # We only need the state transition for the search
                _, _, _, _, next_info = initial_env_copy.step(action)
            except ValueError as e:
                print(f"Warning: Error simulating first move {action}: {e}")
                value = INF_MOVES  # Penalize if simulating the move fails
            else:
                # --- Start the minimax search from the state *after* the first move ---
                # Depth starts at 1 because we've already made one move
                value = self._minimax_search(initial_env_copy, depth=1)

            print(f"LookaheadAgent: Move {action} evaluated to value: {value}")  # Debugging

            # --- Update best action ---
            # We want the action that leads to the minimum value (fastest win)
            if value < best_value:
                best_value = value
                best_action = action
            elif value == best_value:
                # Tie-breaking: randomly choose among equally good moves
                if best_action == -1 or self.rng.random() < 0.5:  # 50% chance to switch
                    best_action = action

        end_time = time.time()
        print(f"LookaheadAgent: Search complete in {end_time - start_time:.2f}s. Best value: {best_value}")

        # Fallback if no move seems good (all lead to infinity)
        if best_action == -1:
            print("Warning: All initial moves evaluated to infinity. Choosing random valid move.")
            best_action = self.rng.choice(valid_actions)

        return best_action

    def _check_terminal_status(self, env_state: SuperTicTacToe3DEnv) -> Tuple[bool, Optional[int]]:
        """
        Helper to determine if a copied environment state is terminal.
        This might require access to env internals or replicating win checks.
        NOTE: This logic should mirror the termination checks within env.step().
        """
        # Check for large board win for either player
        for player in [PLAYER1, PLAYER2]:
            large_won, _ = env_state._check_win_or_draw(env_state._large_board, player)
            if large_won:
                return True, player

        # Check for draw (no empty cells on large board OR no valid moves left)
        # Recompute mask needed here based on env_state's current player/next cell
        current_mask = env_state._compute_action_mask()  # Assumes _compute_action_mask is available
        if not np.any(current_mask):
            # Check if large board is full *and* no winner was found above
            large_drawn = not np.any(env_state._large_board == env_state.EMPTY)
            if large_drawn:
                return True, DRAW  # Draw (full board, no winner)
            else:
                # No moves left, but board not full? Can happen if remaining cells are blocked. Treat as draw.
                return True, DRAW

        # Not terminal
        return False, None


# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Running Sample Episode (Lookahead Agent vs Heuristic Agent) ---")

    # Create the base environment instance
    try:
        # Need render_mode=None for copying, create a separate one for rendering if needed
        env_for_logic = gym.make('SuperTicTacToe3D-v0', render_mode=None)
        # Optional: Create a second env just for rendering the final output
        render_env = gym.make('SuperTicTacToe3D-v0', render_mode='human')
        print("Environments created.")
    except gym.error.Error as e:
        print(f"Failed to create environment: {e}")
        print("Ensure the environment is registered correctly.")
        exit()
    except NameError:
        print("SuperTicTacToe3DEnv class not found. Imports failed.")
        exit()

    # Use a fixed seed for reproducibility
    seed = 123
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
        render_env.render()  # Render the state *before* the move

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
