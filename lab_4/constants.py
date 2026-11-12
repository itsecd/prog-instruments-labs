"""Shared UI constants for the game collection."""

# --- Colors ---
BG_COLOR = "#90EE90"  # light green background used by every game screen

# Escape room ("dark" theme)
ADV_BG = "#2C3E50"
ADV_PANEL_BG = "#34495E"
ADV_TEXT = "#ECF0F1"
ADV_ACCENT = "#3498DB"
ADV_ACCENT_ACTIVE = "#2980B9"
ADV_INVENTORY = "#F39C12"
ADV_DANGER = "#E74C3C"
ADV_SUCCESS = "#27AE60"

# Battleship ship placement (per player)
PLACEMENT_BG = {1: "#E6F7FF", 2: "#FFE6E6"}
PLACEMENT_CELL_BG = {1: "lightblue", 2: "lightpink"}

# --- Fonts ---
FONT_TITLE = ('Arial', 16, 'bold')
FONT_SUBTITLE = ('Arial', 14, 'bold')
FONT_TEXT = ('Arial', 12)
FONT_BODY = ('Arial', 11)
FONT_SMALL_BOLD = ('Arial', 10, 'bold')
FONT_BOARD_LABEL = ('Arial', 8, 'bold')
FONT_BOARD_CELL = ('Arial', 8)

# --- Window sizes ---
SIZE_MENU = "600x500"
SIZE_BATTLESHIP = "1000x900"
SIZE_TEXT_ADVENTURE = "900x750"

# Games that need a bigger window than the menu (frame name -> size)
GAME_WINDOW_SIZES = {
    "BattleshipGUI": SIZE_BATTLESHIP,
    "TextAdventureGUI": SIZE_TEXT_ADVENTURE,
}

# --- Battleship ---
MODE_COMPUTER = "COMPUTER"
MODE_TWO_PLAYERS = "TWOPLAYER"