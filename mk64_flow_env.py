from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gym
from gym import spaces
from gym.envs.registration import register

from gym_mupen64plus.envs.MarioKart64.mario_kart_env import MarioKartEnv
from gym_mupen64plus.envs.mupen64plus_env import ControllerState


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class MenuScriptedMarioKartEnv(MarioKartEnv):
    """
    Mario Kart 64 environment that scripts everything up to the character select screen,
    lets the agent pick a driver, auto-selects Mushroom Cup / Luigi Raceway, then restricts
    driving to a minimal, stable action set.
    """

    # Action ids used in both menu (character select) and driving phases.
    # Driving phase only meaning is documented in DRIVE_ACTIONS.
    MENU_ACTIONS: Dict[int, str] = {
        0: "noop",
        1: "confirm",
        2: "left",
        3: "right",
        4: "up",
        5: "down",
    }

    DRIVE_ACTIONS: List[Tuple[str, List[int]]] = [
        ("noop", [0, 0, 0, 0, 0]),
        ("accel", [0, 0, 1, 0, 0]),
        ("accel_left", [-40, 0, 1, 0, 0]),
        ("accel_right", [40, 0, 1, 0, 0]),
        ("brake", [0, 0, 0, 1, 0]),
        ("coast", [0, 0, 0, 0, 0]),  # alias to keep a down action during menus
    ]

    CHARACTER_GRID: List[List[str]] = [
        ["mario", "luigi", "peach", "toad"],
        ["yoshi", "d.k.", "wario", "bowser"],
    ]

    def __init__(self, character_reward: float = 1.0):
        # Default selections are placeholders; the agent will override character choice.
        super().__init__(character="mario", course="LuigiRaceway")
        self.character_reward = float(character_reward)
        self.action_space = spaces.Discrete(len(self.DRIVE_ACTIONS))

        # State used during the character selection phase.
        self.waiting_for_character = True
        self.cursor_row = 0
        self.cursor_col = 0
        self.selected_character: Optional[str] = None

    # --- Navigation overrides ---
    def _navigate_menu(self):
        """
        Script through the title and mode selection until the player select screen.
        """
        self._wait(count=10, wait_for="Nintendo screen")
        self._press_button(ControllerState.START_BUTTON)

        self._wait(count=68, wait_for="Mario Kart splash screen")
        self._press_button(ControllerState.START_BUTTON)

        self._wait(count=68, wait_for="Game Select screen")
        self._navigate_game_select()

        self._wait(count=14, wait_for="Player Select screen")
        self._prepare_character_cursor()

    def _navigate_game_select(self):
        """
        Force 1 Player -> Time Trial -> Begin -> OK.
        """
        # 1 Player is highlighted by default; confirm.
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=3, wait_for="confirm 1P")

        # Move to Time Trial and confirm.
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._wait(count=3, wait_for="highlight Time Trial")
        self._press_button(ControllerState.A_BUTTON)

        # Begin and OK.
        self._press_button(ControllerState.A_BUTTON)
        self._press_button(ControllerState.A_BUTTON)

    def _navigate_map_select(self):
        """
        Force Mushroom Cup -> Luigi Raceway regardless of prior choice.
        """
        # Ensure Mushroom Cup (left-most) is active, then pick Luigi Raceway (top-left).
        self.MAP_SERIES, self.MAP_CHOICE = (0, 0)
        # Always reset selection to the left/top first for determinism.
        self._press_button(ControllerState.JOYSTICK_LEFT, times=3)
        self._press_button(ControllerState.JOYSTICK_RIGHT, times=self.MAP_SERIES)
        self._press_button(ControllerState.A_BUTTON)

        self._press_button(ControllerState.JOYSTICK_UP, times=3)
        self._press_button(ControllerState.JOYSTICK_DOWN, times=self.MAP_CHOICE)
        self._press_button(ControllerState.A_BUTTON)
        self._press_button(ControllerState.A_BUTTON)

    def _prepare_character_cursor(self):
        """
        Land on the player select screen with the cursor at the upper-left character.
        """
        self.waiting_for_character = True
        self.selected_character = None
        self.cursor_row = 0
        self.cursor_col = 0

        # Force cursor to the upper-left for a predictable starting point.
        self._press_button(ControllerState.JOYSTICK_UP)
        self._press_button(ControllerState.JOYSTICK_LEFT, times=3)

    def _reset_after_race(self):
        """
        After a race finishes, return to driver select instead of map select.
        """
        self._wait(count=275, wait_for="times screen")
        self._navigate_post_race_menu()
        self._wait(count=14, wait_for="Player Select screen")
        self._prepare_character_cursor()

    def _reset_during_race(self):
        """
        Pause and back out to character select; used if reset happens mid-race.
        """
        if (self.step_count * self.controller_server.frame_skip) < 120:
            steps_to_wait = 100 - (self.step_count * self.controller_server.frame_skip)
            self._wait(count=steps_to_wait, wait_for="green light so we can pause")
        self._press_button(ControllerState.START_BUTTON)
        # Menu order: Retry, Course Change, Driver Change, ...
        self._press_button(ControllerState.JOYSTICK_DOWN, times=2)
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=76, wait_for="return to player select")
        self._prepare_character_cursor()

    def _navigate_post_race_menu(self):
        """
        From the post-race menu, move to Driver Change to return to character select.
        """
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=13, wait_for="Post race menu")
        self._press_button(ControllerState.JOYSTICK_UP, times=5)
        self._press_button(ControllerState.JOYSTICK_DOWN, times=2)  # Driver Change
        self._press_button(ControllerState.A_BUTTON)

    # --- Core step/reset overrides ---
    def _reset(self):
        self.waiting_for_character = True
        self.selected_character = None
        self.cursor_row = 0
        self.cursor_col = 0
        obs = super()._reset()
        self.step_count = 0
        return obs

    def step(self, action):
        action_int = int(action)
        if self.waiting_for_character:
            reward = self._handle_character_select_action(action_int)
            obs = self._observe()
            done = False
            info = {
                "phase": "character_select",
                "cursor": (self.cursor_row, self.cursor_col),
                "selected_character": self.selected_character,
            }
            return obs, reward, done, info

        controls = self._drive_controls(action_int)
        obs, reward, done, info = super()._step(controls)
        info = info or {}
        info.update(
            {
                "phase": "driving",
                "selected_character": self.selected_character,
                "action_name": self.DRIVE_ACTIONS[_clamp(action_int, 0, len(self.DRIVE_ACTIONS) - 1)][0],
            }
        )
        return obs, reward, done, info

    # --- Helpers ---
    def _drive_controls(self, action: int) -> List[int]:
        idx = _clamp(action, 0, len(self.DRIVE_ACTIONS) - 1)
        return self.DRIVE_ACTIONS[idx][1]

    def _handle_character_select_action(self, action: int) -> float:
        """
        Map the small discrete action set to menu navigation and character confirmation.
        """
        reward = 0.0
        if action == 1:  # confirm / A
            character = self.CHARACTER_GRID[self.cursor_row][self.cursor_col]
            self._press_button(ControllerState.A_BUTTON)
            self._press_button(ControllerState.A_BUTTON)
            self.selected_character = character
            reward += self.character_reward
            self._advance_to_race()
        elif action == 2:  # left
            self.cursor_col = _clamp(self.cursor_col - 1, 0, len(self.CHARACTER_GRID[0]) - 1)
            self._press_button(ControllerState.JOYSTICK_LEFT)
        elif action == 3:  # right
            self.cursor_col = _clamp(self.cursor_col + 1, 0, len(self.CHARACTER_GRID[0]) - 1)
            self._press_button(ControllerState.JOYSTICK_RIGHT)
        elif action == 4:  # up
            self.cursor_row = _clamp(self.cursor_row - 1, 0, len(self.CHARACTER_GRID) - 1)
            self._press_button(ControllerState.JOYSTICK_UP)
        elif action == 5:  # down
            self.cursor_row = _clamp(self.cursor_row + 1, 0, len(self.CHARACTER_GRID) - 1)
            self._press_button(ControllerState.JOYSTICK_DOWN)
        else:
            # noop/coast
            self._act(ControllerState.NO_OP)
        return reward

    def _advance_to_race(self):
        """
        After a character is selected, script cup/track selection and get into the race.
        """
        # Small wait for the character confirm animation.
        self._wait(count=6, wait_for="character confirm")

        # OK to continue.
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=14, wait_for="Map Select screen")

        self._navigate_map_select()
        self._wait(count=46, wait_for="race to load")

        # Consistent HUD then start driving.
        self._cycle_hud_view(times=2)
        self.waiting_for_character = False
        self.step_count = 0


def register_menu_restricted_env(env_id: str = "Mario-Kart-Menu-Restricted-v0"):
    """
    Idempotent registration helper so gym.make can find the custom flow env.
    """
    try:
        registry = gym.envs.registry  # type: ignore[attr-defined]
        specs = getattr(registry, "env_specs", registry)
        if env_id in specs:
            return
    except Exception:
        pass
    register(
        id=env_id,
        entry_point="mk64_flow_env:MenuScriptedMarioKartEnv",
        max_episode_steps=1250,
    )


# Register on import for convenience.
register_menu_restricted_env()
