import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
from typing import Optional, Tuple, Dict, Any, List, Union


def _get_winning_lines_3x3x3():
    """Generates all winning lines (indices 0-26) for a 3x3x3 cube."""
    lines = []

    # Helper to convert 3D coords (x, y, z) to 1D index
    def to_1d(x, y, z):
        return x + y * 3 + z * 9

    # 1. Axis-aligned lines (X, Y, Z) - 9 lines per direction = 27 total
    for y in range(3):
        for z in range(3):
            lines.append([to_1d(0, y, z), to_1d(1, y, z), to_1d(2, y, z)])  # X direction
    for x in range(3):
        for z in range(3):
            lines.append([to_1d(x, 0, z), to_1d(x, 1, z), to_1d(x, 2, z)])  # Y direction
    for x in range(3):
        for y in range(3):
            lines.append([to_1d(x, y, 0), to_1d(x, y, 1), to_1d(x, y, 2)])  # Z direction

    # 2. Planar diagonals (XY, XZ, YZ planes) - 2 diagonals per slice * 3 slices per direction = 18 total
    # XY plane diagonals (varying z)
    for z in range(3):
        lines.append([to_1d(0, 0, z), to_1d(1, 1, z), to_1d(2, 2, z)])
        lines.append([to_1d(0, 2, z), to_1d(1, 1, z), to_1d(2, 0, z)])
    # XZ plane diagonals (varying y)
    for y in range(3):
        lines.append([to_1d(0, y, 0), to_1d(1, y, 1), to_1d(2, y, 2)])
        lines.append([to_1d(0, y, 2), to_1d(1, y, 1), to_1d(2, y, 0)])
    # YZ plane diagonals (varying x)
    for x in range(3):
        lines.append([to_1d(x, 0, 0), to_1d(x, 1, 1), to_1d(x, 2, 2)])
        lines.append([to_1d(x, 0, 2), to_1d(x, 1, 1), to_1d(x, 2, 0)])

    # 3. Space diagonals - 4 total
    lines.append([to_1d(0, 0, 0), to_1d(1, 1, 1), to_1d(2, 2, 2)])
    lines.append([to_1d(0, 0, 2), to_1d(1, 1, 1), to_1d(2, 2, 0)])
    lines.append([to_1d(0, 2, 0), to_1d(1, 1, 1), to_1d(2, 0, 2)])
    lines.append([to_1d(2, 0, 0), to_1d(1, 1, 1), to_1d(0, 2, 2)])

    assert len(lines) == 27 + 18 + 4 == 49, f"Expected 49 lines, found {len(lines)}"
    return [tuple(sorted(line)) for line in lines]  # Use tuples for set operations if needed


class SuperTicTacToe3DEnv(gym.Env):
    """
    Gymnasium environment for 3D Super Tic-Tac-Toe (Ultimate Tic-Tac-Toe in 3D).

    Game Rules:
    - Played on a 3x3x3 grid of large cells.
    - Each large cell contains a small 3x3x3 Tic-Tac-Toe board.
    - Players take turns placing their mark (Player 1 or Player 2) in an empty cell
      of one of the small boards.
    - The small cell's position (x, y, z) within its small board dictates the
      large cell (X, Y, Z) where the *next* player must play.
    - If the dictated large cell's small board is already won or full (drawn),
      the next player can play in any small cell of any large board that is not
      won or full.
    - Winning a small 3x3x3 board claims the corresponding large cell for that player.
    - The first player to get 3 of their large cells in a row (horizontally,
      vertically, depth-wise, or diagonally according to 3D Tic-Tac-Toe rules)
      wins the overall game.
    - A small board is drawn if it's full but no one has won it. Drawn small boards
      do not contribute to winning the large board.
    - The game is a draw if the large board is filled (or no more moves are possible)
      without a winner.

    Observation Space: Dict
        - `small_boards`: Box(0, 2, (27, 27), int8) - State of all 729 small cells.
                         Shape is (large_cell_idx, small_cell_idx).
                         0=empty, 1=player1, 2=player2.
        - `large_board`: Box(0, 3, (27,), int8) - State of the 27 large cells.
                         0=empty/ongoing, 1=won_by_p1, 2=won_by_p2, 3=drawn.
        - `current_player`: Discrete(2, start=1) - Player whose turn it is (1 or 2).
        - `next_large_cell`: Discrete(28) - Index (0-26) of the large cell where the
                             current player *must* play. 27 means the player can choose
                             any valid cell in any non-finished large board.

    Action Space: Discrete(729)
        - Represents playing in one of the 729 small cells.
        - Action `a` corresponds to playing in large cell `a // 27` and small cell `a % 27`.
        - An action mask is provided in `info['action_mask']` to indicate valid moves.
          The agent *must* select an action where the mask is 1.

    Reward:
        - +1 for winning the game.
        - -1 for losing the game.
        - 0 for all other steps (including draw).

    Termination:
        - The game ends when a player wins the large board or when no more valid
          moves can be made (draw).
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    DRAW = 3  # Used only for large_board status

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        # Precompute winning lines for efficiency
        self._winning_lines = _get_winning_lines_3x3x3()

        # --- Define Spaces ---
        # Observation Space
        self.observation_space = spaces.Dict({
            # Using int8 for memory efficiency
            "small_boards": spaces.Box(low=0, high=2, shape=(27, 27), dtype=np.int8),
            "large_board": spaces.Box(low=0, high=3, shape=(27,), dtype=np.int8),
            "current_player": spaces.Discrete(2, start=1),  # Players 1 and 2
            "next_large_cell": spaces.Discrete(28)  # 0-26 for specific, 27 for any
        })

        # Action Space: Flat index for all 729 small cells
        self.action_space = spaces.Discrete(27 * 27)

        # Internal state (initialized in reset)
        self._small_boards: Optional[np.ndarray] = None
        self._large_board: Optional[np.ndarray] = None
        self._current_player: Optional[int] = None
        self._next_large_cell_idx: Optional[int] = None  # 0-26, or 27 for "any"
        self._action_mask: Optional[np.ndarray] = None
        self._game_winner: Optional[int] = None  # 0=draw, 1=P1, 2=P2
        self._is_terminal: bool = False

    def _check_win_or_draw(self, board_1d: np.ndarray, player: int) -> Tuple[bool, bool]:
        """
        Checks if the given player has won or if the board is drawn.
        Args:
            board_1d: A flattened 1D numpy array (size 27) representing a board
                      (small or large). For large board, uses 1, 2 for players.
                      For small board, uses 1, 2 for players. DRAW state (3)
                      on large board doesn't count for winning.
            player: The player (1 or 2) whose win status to check.
        Returns:
            (has_won, is_drawn)
        """
        # Check win
        for line in self._winning_lines:
            if all(board_1d[i] == player for i in line):
                return True, False  # Player has won

        # Check draw (board full and no winner)
        # A board is full if it has no EMPTY cells (0)
        # Note: For the large board, drawn small boards (3) also mean it's not empty
        if not np.any(board_1d == self.EMPTY):
            # We already checked for a win for the current player.
            # If we are checking the *large* board for draw, we need to ensure
            # the *other* player didn't just win either. This is implicitly handled
            # because we check for win *before* checking for draw in the step function.
            # If we get here, the board is full and the current player didn't win.
            return False, True  # Board is full, no winner -> Draw

        return False, False  # Game ongoing

    def _get_obs(self) -> Dict[str, Any]:
        """Constructs the observation dictionary from the internal state."""
        return {
            "small_boards": self._small_boards.copy(),
            "large_board": self._large_board.copy(),
            "current_player": self._current_player,
            "next_large_cell": self._next_large_cell_idx
        }

    def _get_info(self) -> Dict[str, Any]:
        """Constructs the info dictionary, including the action mask."""
        return {
            "action_mask": self._action_mask.copy(),
            "game_winner": self._game_winner  # None if ongoing, 0 draw, 1 P1, 2 P2
        }

    def _compute_action_mask(self) -> np.ndarray:
        """
        Computes the legal action mask based on the current state.
        Returns:
            A numpy array of shape (729,) with 1s for valid moves, 0s otherwise.
        """
        mask = np.zeros(729, dtype=np.int8)
        can_play_anywhere = (self._next_large_cell_idx == 27)

        if can_play_anywhere:
            # Iterate through all large cells
            for large_idx in range(27):
                # Can only play if the large cell is not finished (won or drawn)
                if self._large_board[large_idx] == self.EMPTY:
                    # Iterate through small cells within this large cell
                    for small_idx in range(27):
                        if self._small_boards[large_idx, small_idx] == self.EMPTY:
                            mask[large_idx * 27 + small_idx] = 1
        else:
            # Forced to play in a specific large cell (_next_large_cell_idx)
            # Check if this target cell is actually playable (should be, unless game ended unexpectedly)
            large_idx = self._next_large_cell_idx
            if self._large_board[large_idx] == self.EMPTY:
                # Iterate through small cells within this specific large cell
                for small_idx in range(27):
                    if self._small_boards[large_idx, small_idx] == self.EMPTY:
                        mask[large_idx * 27 + small_idx] = 1
            else:
                # This case should ideally not happen if _next_large_cell_idx was set correctly
                # It implies the game sent the player to a finished board.
                # Fallback to playing anywhere (treat as if _next_large_cell_idx was 27)
                # This could indicate a bug in the _next_large_cell_idx update logic.
                # Let's recompute as if play is anywhere
                print(
                    f"Warning: Player {self._current_player} forced to play in finished large cell {large_idx}. Allowing play anywhere.")
                for l_idx in range(27):
                    if self._large_board[l_idx] == self.EMPTY:
                        for s_idx in range(27):
                            if self._small_boards[l_idx, s_idx] == self.EMPTY:
                                mask[l_idx * 27 + s_idx] = 1

        return mask

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment to the initial state."""
        super().reset(seed=seed)

        self._small_boards = np.full((27, 27), self.EMPTY, dtype=np.int8)
        self._large_board = np.full((27,), self.EMPTY, dtype=np.int8)
        self._current_player = self.PLAYER1  # Player 1 starts
        # Player 1 can initially play anywhere
        self._next_large_cell_idx = 27
        self._game_winner = None
        self._is_terminal = False

        # Compute the initial action mask
        self._action_mask = self._compute_action_mask()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
            self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Performs one step in the environment."""

        if self._is_terminal:
            # Game already ended, return current state
            # This can happen if the environment is stepped again after termination.
            # It's good practice to handle this gracefully.
            print("Warning: Stepping environment after termination.")
            obs = self._get_obs()
            info = self._get_info()
            # Return 0 reward, terminated=True, truncated=False
            return obs, 0.0, True, False, info

        # --- Action Validation ---
        if not (0 <= action < 729):
            raise ValueError(f"Invalid action {action}. Action must be in [0, 728].")
        if self._action_mask[action] == 0:
            # For debugging, print available moves if an invalid one is chosen
            # available_moves = np.where(self._action_mask == 1)[0]
            # print(f"Available moves: {available_moves}")
            # print(f"Current state: Player {self._current_player}, Next Large Cell {self._next_large_cell_idx}")
            # print(f"Large Board State: {self._large_board}")
            raise ValueError(
                f"Invalid action {action} chosen by player {self._current_player}. "
                f"Move is not allowed according to the action mask."
                f" (Large Cell: {action // 27}, Small Cell: {action % 27})"
            )

        # --- Apply Action ---
        large_idx = action // 27
        small_idx = action % 27

        # Ensure the chosen small cell is actually empty (redundant if mask is correct)
        # assert self._small_boards[large_idx, small_idx] == self.EMPTY, "Chosen cell not empty!"

        self._small_boards[large_idx, small_idx] = self._current_player

        # --- Check Small Board Win/Draw ---
        small_board_1d = self._small_boards[large_idx, :]
        small_won, small_drawn = self._check_win_or_draw(small_board_1d, self._current_player)

        if small_won:
            self._large_board[large_idx] = self._current_player
        elif small_drawn:
            self._large_board[large_idx] = self.DRAW

        # --- Check Large Board Win/Draw (Game End) ---
        terminated = False
        reward = 0.0
        self._game_winner = None  # Reset winner for this step

        if small_won:  # Only need to check large board win if a small board was just won
            large_won, large_drawn = self._check_win_or_draw(self._large_board, self._current_player)
            if large_won:
                terminated = True
                self._game_winner = self._current_player
                # Reward: +1 for winner, -1 for loser (assigned after player switch)
                reward = 1.0
            elif large_drawn:  # Check if large board is drawn
                terminated = True
                self._game_winner = 0  # Draw
                reward = 0.0

        # Check for draw condition even if no small board was won this turn
        # (i.e., the last playable cell was filled, resulting in a draw)
        if not terminated and not np.any(self._compute_action_mask()):  # Recompute mask for next player
            # If there are no valid moves left for the *next* player, it's a draw
            # This check is done *before* switching the player
            is_large_board_full = not np.any(self._large_board == self.EMPTY)
            if is_large_board_full:
                # We need to ensure the *other* player didn't win on the previous turn
                # This is tricky. Let's rely on the action mask: if no moves are possible, it's a draw.
                # We check if any EMPTY cells exist on the large board; if not, and no winner, it's a draw.
                # The most robust check is if the *next* action mask is all zeros.

                # Re-calculate potential next player's mask
                potential_next_player = self.PLAYER1 if self._current_player == self.PLAYER2 else self.PLAYER2
                potential_next_large_idx = small_idx  # Where the next player *would* be sent

                # Determine where the next player *must* play
                if self._large_board[potential_next_large_idx] != self.EMPTY:
                    potential_next_large_idx = 27  # Can play anywhere

                # Create a temporary mask based on this potential next state
                temp_mask = np.zeros(729, dtype=np.int8)
                if potential_next_large_idx == 27:
                    for l_idx in range(27):
                        if self._large_board[l_idx] == self.EMPTY:
                            for s_idx in range(27):
                                if self._small_boards[l_idx, s_idx] == self.EMPTY:
                                    temp_mask[l_idx * 27 + s_idx] = 1
                else:
                    if self._large_board[potential_next_large_idx] == self.EMPTY:
                        for s_idx in range(27):
                            if self._small_boards[potential_next_large_idx, s_idx] == self.EMPTY:
                                temp_mask[potential_next_large_idx * 27 + s_idx] = 1

                # Check if any valid moves exist for the next player
                if not np.any(temp_mask):
                    terminated = True
                    self._game_winner = 0  # Draw
                    reward = 0.0

        self._is_terminal = terminated  # Update terminal state flag

        # --- Prepare for Next Turn ---
        if not terminated:
            # Switch player
            self._current_player = self.PLAYER1 if self._current_player == self.PLAYER2 else self.PLAYER2

            # Determine where the next player must play
            next_large_forced_idx = small_idx  # Based on the small cell just played
            if self._large_board[next_large_forced_idx] == self.EMPTY:
                self._next_large_cell_idx = next_large_forced_idx
            else:
                # The target large cell is finished (won or drawn), player can go anywhere
                self._next_large_cell_idx = 27

            # Compute action mask for the *new* current player
            self._action_mask = self._compute_action_mask()

            # Check again if the new player has any moves (redundant draw check?)
            if not np.any(self._action_mask):
                # This means the game ended in a draw *after* switching players
                # (e.g. P1 makes last move, P2 has nowhere to go)
                terminated = True
                self._is_terminal = True
                self._game_winner = 0  # Draw
                reward = 0.0  # No reward change needed for draw
                # Ensure mask is all zeros if terminal
                self._action_mask.fill(0)


        else:  # Game has terminated
            # Assign reward based on who won/lost relative to the *current* player at start of step
            if self._game_winner == self._current_player:
                reward = 1.0
            elif self._game_winner == (self.PLAYER1 if self._current_player == self.PLAYER2 else self.PLAYER2):
                # The *other* player won
                reward = -1.0
            else:  # Draw
                reward = 0.0

            # No valid moves left if game is over
            self._action_mask = np.zeros(729, dtype=np.int8)
            self._next_large_cell_idx = -1  # Indicate no next move possible / relevant

        # --- Return Results ---
        observation = self._get_obs()
        info = self._get_info()
        truncated = False  # Board games usually don't truncate unless step limit imposed

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Renders the environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()
            return None  # Human rendering typically prints to console
        else:
            # For render_mode=None or other modes like "rgb_array",
            # we are not implementing graphical rendering here.
            return None

    def _render_human(self):
        """Prints a textual representation to the console."""
        print(self._render_ansi())

    def _render_ansi(self) -> str:
        """Returns an ANSI string representation of the state."""
        # This is complex to display nicely. We'll show the large board
        # and the currently active small board(s).

        player_map = {0: ".", 1: "X", 2: "O", 3: "D"}  # D for Drawn large cell

        s = f"--- 3D Super Tic-Tac-Toe ---\n"
        s += f"Current Player: {player_map[self._current_player]}\n"

        # Large Board Representation (3 layers)
        s += "Large Board State (Layers Z=0, Z=1, Z=2):\n"
        s += "[.] signifies the next playable large cell(s)\n"
        for z in range(3):
            s += f" Z={z}\n"
            for y in range(2, -1, -1):  # Print Y from top (2) to bottom (0)
                row_str = "  "
                for x in range(3):
                    idx_1d = x + y * 3 + z * 9
                    cell_state = self._large_board[idx_1d]
                    # Highlight the next playable large cell(s)
                    is_next_cell = False
                    if self._next_large_cell_idx == 27:  # Any cell is possible target
                        if cell_state == self.EMPTY: is_next_cell = True
                    elif self._next_large_cell_idx == idx_1d:
                        is_next_cell = True

                    marker = player_map[cell_state]
                    if is_next_cell and not self._is_terminal:
                        row_str += f"[{marker}] "  # Indicate potential target
                    else:
                        row_str += f" {marker}  "
                s += row_str + "\n"
            s += "\n"

        # Active Small Board(s)
        s += "Active Small Board(s):\n"
        if self._is_terminal:
            s += "  Game Over.\n"
        elif self._next_large_cell_idx == 27:
            s += "  Player can play in any non-finished large board.\n"
        else:
            large_idx = self._next_large_cell_idx
            lx, ly, lz = large_idx % 3, (large_idx // 3) % 3, large_idx // 9
            s += f"  Player must play in Large Cell {large_idx} (Coords: x={lx}, y={ly}, z={lz})\n"
            # Print the specific small board
            for z in range(3):
                s += f"    Small Board Z={z} Slice:\n"
                for y in range(3):
                    row_str = "      "
                    for x in range(3):
                        small_1d = x + y * 3 + z * 9
                        cell_state = self._small_boards[large_idx, small_1d]
                        row_str += f"{player_map[cell_state]} "
                    s += row_str + "\n"
                s += "\n"

        # Winner Info
        if self._is_terminal:
            if self._game_winner == 1:
                s += "Result: Player X (1) Wins!\n"
            elif self._game_winner == 2:
                s += "Result: Player O (2) Wins!\n"
            elif self._game_winner == 0:
                s += "Result: Draw!\n"

        return s

    def close(self):
        """Clean up any resources (not needed for this simple env)."""
        pass


# --- Registration ---
# You would typically put the registration in your package's __init__.py
# For this example, we'll just call it here.

def register_env():
    try:
        gym.register(
            id='SuperTicTacToe3D-v0',
            entry_point=f'{__name__}:SuperTicTacToe3DEnv',  # Dynamically get module name
            max_episode_steps=None,  # Game has natural termination
        )
        print("Registered SuperTicTacToe3D-v0 environment.")
    except gym.error.Error as e:
        # Environment might already be registered if script is run multiple times
        print(f"Environment registration skipped: {e}")


register_env()


def print_3d_grid_indices():
    """
    Prints the 1D index (0-26) for each cell in a 3x3x3 grid,
    visualized layer by layer (along the Z-axis).
    Uses the standard indexing: index = x + y*3 + z*9
    """
    print("--- 3x3x3 Grid Cell Indices ---")
    print("Mapping: index = x + y*3 + z*9")
    print("(Where x, y, z range from 0 to 2)")
    print("-" * 30)

    for z in range(3):  # Iterate through layers (depth)
        print(f"\nLayer Z = {z}:")
        print("        <---- X ---->")
        print("         (0) (1) (2)")
        print("        +---+---+---+")
        # Iterate through rows (Y), printing from top (y=2) to bottom (y=0)
        # for a more conventional grid display.
        for y in range(2, -1, -1):
            row_str = f"Y={y} |"
            # Iterate through columns (X)
            for x in range(3):
                index = x + y * 3 + z * 9
                # Format index to take 2 spaces for alignment
                row_str += f" {index:2d}|"
            print(f"  ^ {row_str}")
            print("  |     +---+---+---+")
        print("  Y")  # Label Y axis direction

    print("\n--- End of Grid Indices ---")

# --- Example Usage and Validation ---
if __name__ == '__main__':
    print("\n--- Validating Environment ---")
    try:
        # Use the registered ID
        env = gym.make('SuperTicTacToe3D-v0', render_mode='human')
        # Check the environment using Gymnasium's tool
        from gymnasium.utils.env_checker import check_env

        check_env(env.unwrapped)  # Use unwrapped to bypass wrappers for the check
        print("Environment validation successful!")
    except Exception as e:
        print(f"Environment validation failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Running Sample Episode (Random Agent) ---")
    print_3d_grid_indices()
    print()
    env = gym.make('SuperTicTacToe3D-v0', render_mode='human')  # Create instance with rendering
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        action_mask = info['action_mask']
        # Choose a random valid action
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            print("No valid actions available! Should be terminal.")
            break
        action = env.np_random.choice(valid_actions)  # Use env's RNG

        print(f"\n--- Step {step_count + 1} ---")
        # Decode action for clarity
        large_act_idx = action // 27
        small_act_idx = action % 27
        print(
            f"Player {obs['current_player']} selects action {action} (Large Cell: {large_act_idx}, Small Cell: {small_act_idx})")

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        # Render is called inside step when render_mode='human'

        if terminated:
            print("\n--- Game Over ---")
            if info['game_winner'] == 1:
                print("Winner: Player 1 (X)")
            elif info['game_winner'] == 2:
                print("Winner: Player 2 (O)")
            elif info['game_winner'] == 0:
                print("Result: Draw")
            else:
                print("Result: Unknown (Error?)")
            print(f"Total steps: {step_count}")

    env.close()