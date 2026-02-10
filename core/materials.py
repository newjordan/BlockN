from typing import Tuple, List

# Define a type hint for colors (e.g., RGBA)
ColorTuple = Tuple[int, int, int, int]

# Basic color palette (Moved from core/scene.py)
DEFAULT_COLORS: List[ColorTuple] = [
    (255, 0, 0, 255),    # Red
    (0, 255, 0, 255),    # Green
    (0, 0, 255, 255),    # Blue
    (255, 255, 0, 255),  # Yellow
    (255, 165, 0, 255), # Orange
    (128, 0, 128, 255), # Purple
    (255, 255, 255, 255),# White
    (128, 128, 128, 255),# Gray
    (0, 0, 0, 255),      # Black
]

# Future: Could define Material objects here with names, properties, etc.
