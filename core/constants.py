"""
LegoGen Constants Module

Centralized constants for the LegoGen application.
Extracts magic numbers and repeated values from across the codebase.

Usage:
    from core.constants import BLOCK_SIZE, CLASSICAL_COLUMN_RADIUS
"""

# =============================================================================
# Block Dimensions
# =============================================================================

# Standard block dimensions (unit size)
BLOCK_SIZE = 1.0  # Standard unit block dimension
BLOCK_WIDTH = 2.0  # Default block width
BLOCK_HEIGHT = 2.0  # Default block height
BLOCK_DEPTH = 2.0  # Default block depth

# Half dimensions (for center-based positioning)
HALF_BLOCK = BLOCK_SIZE / 2.0

# =============================================================================
# Geometry Parameters
# =============================================================================

# Cylinder defaults
DEFAULT_CYLINDER_RADIUS = 0.4
DEFAULT_CYLINDER_HEIGHT = 1.0
DEFAULT_CYLINDER_SECTIONS = 16  # Number of segments in cylinder

# Wedge slope directions
WEDGE_DIRECTION_POS_X = "+X"
WEDGE_DIRECTION_NEG_X = "-X"
WEDGE_DIRECTION_POS_Z = "+Z"
WEDGE_DIRECTION_NEG_Z = "-Z"

# =============================================================================
# Classical Architecture Constants
# =============================================================================

# Column parameters
CLASSICAL_COLUMN_RADIUS = 0.4
CLASSICAL_COLUMN_BASE_HEIGHT = 1.0
CLASSICAL_COLUMN_CAPITAL_HEIGHT = 1.0
CLASSICAL_MIN_COLUMNS = 2
CLASSICAL_MAX_COLUMNS = 12

# Entablature proportions (heights in blocks)
ENTABLATURE_ARCHITRAVE_HEIGHT = 1
ENTABLATURE_FRIEZE_HEIGHT = 1
ENTABLATURE_CORNICE_HEIGHT = 2

# Podium parameters
PODIUM_MIN_LAYERS = 1
PODIUM_MAX_LAYERS = 5
PODIUM_DEFAULT_LAYERS = 3

# Pediment parameters
PEDIMENT_MIN_HEIGHT = 2
PEDIMENT_MAX_HEIGHT = 6
PEDIMENT_SLOPE_RATIO = 0.5  # Height to base ratio

# Classical orders (for reference)
ORDER_DORIC = "DORIC"
ORDER_IONIC = "IONIC"
ORDER_CORINTHIAN = "CORINTHIAN"

# =============================================================================
# Generation Parameters
# =============================================================================

# Default scene dimensions
DEFAULT_SCENE_WIDTH = 10
DEFAULT_SCENE_HEIGHT = 10
DEFAULT_SCENE_DEPTH = 10

# Generation limits
MAX_BLOCKS_DEFAULT = 5000
MAX_BLOCKS_WARNING = 10000
MIN_DIMENSION = 1
MAX_DIMENSION = 100

# Random seed
DEFAULT_SEED = None  # None means random
MIN_SEED = 0
MAX_SEED = 999999

# =============================================================================
# Collision and Support
# =============================================================================

# Support check modes
SUPPORT_MODE_HYBRID = "HYBRID"
SUPPORT_MODE_GRID = "GRID"
SUPPORT_MODE_OFF = "OFF"

# Collision tolerance
COLLISION_TOLERANCE = 1e-6  # Small epsilon for floating point comparisons
MIN_OVERLAP_THRESHOLD = 0.01  # Minimum overlap to consider as collision

# Support detection parameters
GROUND_LEVEL = 0.0  # Y-coordinate of ground plane
SUPPORT_RAY_OFFSET = 0.1  # Offset for ray-casting support checks
SUPPORT_GRID_RESOLUTION = 1.0  # Grid cell size for support checking

# =============================================================================
# Mesh Optimization
# =============================================================================

# Boolean operation parameters
BOOLEAN_UNION_BACKEND = "auto"  # Can be: "auto", "blender", "trimesh", "manifold"
BOOLEAN_TOLERANCE = 1e-5

# Optimization thresholds
OPTIMIZE_MIN_BLOCKS = 2  # Minimum blocks to trigger optimization
OPTIMIZE_BATCH_SIZE = 100  # Process meshes in batches

# =============================================================================
# Blender Integration
# =============================================================================

# Bevel defaults
DEFAULT_BEVEL_AMOUNT = 0.1
DEFAULT_BEVEL_SEGMENTS = 3
DEFAULT_BEVEL_PROFILE = 0.5

# Blender operation timeouts (seconds)
BLENDER_TIMEOUT_SHORT = 60  # 1 minute
BLENDER_TIMEOUT_MEDIUM = 120  # 2 minutes
BLENDER_TIMEOUT_LONG = 300  # 5 minutes

# Blender boolean operation modes
BLENDER_LIMIT_METHOD_ANGLE = "ANGLE"
BLENDER_LIMIT_METHOD_WEIGHT = "WEIGHT"
BLENDER_LIMIT_METHOD_VGROUP = "VGROUP"

BLENDER_MITER_OUTER_SHARP = "SHARP"
BLENDER_MITER_OUTER_PATCH = "PATCH"
BLENDER_MITER_OUTER_ARC = "ARC"

BLENDER_MITER_INNER_SHARP = "SHARP"
BLENDER_MITER_INNER_ARC = "ARC"

# =============================================================================
# UI Constants
# =============================================================================

# Window dimensions
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_WINDOW_X = 100
DEFAULT_WINDOW_Y = 100

# Viewport camera defaults
VIEWPORT_INITIAL_ZOOM = 1.5
VIEWPORT_INITIAL_CENTER = (0.5, 0.5, 0.5)

# Status bar message durations (milliseconds)
STATUS_MESSAGE_SHORT = 3000  # 3 seconds
STATUS_MESSAGE_MEDIUM = 5000  # 5 seconds
STATUS_MESSAGE_PERSISTENT = 0  # Persistent (until replaced)

# Panel dimensions
PANEL_MIN_WIDTH = 200
PANEL_DEFAULT_WIDTH = 250
PANEL_MAX_WIDTH = 400

# =============================================================================
# File I/O
# =============================================================================

# File extensions
FILE_EXT_GLB = ".glb"
FILE_EXT_JSON = ".json"
FILE_EXT_OBJ = ".obj"
FILE_EXT_STL = ".stl"

# Export formats
EXPORT_FORMAT_GLB = "glb"
EXPORT_FORMAT_JSON = "json"
EXPORT_FORMAT_OBJ = "obj"

# Default file names
DEFAULT_EXPORT_NAME = "building"
DEFAULT_SAVE_NAME = "scene"

# =============================================================================
# AI Architect Constants
# =============================================================================

# OpenAI parameters
AI_MAX_TOKENS = 4000
AI_TEMPERATURE = 0.7
AI_DEFAULT_MODEL = "gpt-4"

# AI generation limits
AI_MAX_BLOCKS = 1000
AI_MIN_BLOCKS = 10
AI_TIMEOUT = 60  # seconds

# =============================================================================
# Color Constants (references to materials.py)
# =============================================================================

# Default alpha value
DEFAULT_ALPHA = 255

# Color categories
COLOR_PRIMARY = "primary"
COLOR_SECONDARY = "secondary"
COLOR_ACCENT = "accent"
COLOR_NEUTRAL = "neutral"

# =============================================================================
# Performance Constants
# =============================================================================

# Threading
GENERATION_THREAD_TIMEOUT = 300000  # 5 minutes in milliseconds
THREAD_CLEANUP_TIMEOUT = 3000  # 3 seconds

# Memory management
MAX_MESH_VERTICES = 1_000_000  # Warn if mesh exceeds this
MAX_SCENE_MEMORY_MB = 500  # Rough memory limit estimate

# Rendering
MAX_ACTORS_BEFORE_SIMPLIFY = 1000  # Simplify rendering above this
RENDER_UPDATE_INTERVAL = 100  # milliseconds

# =============================================================================
# Testing Constants
# =============================================================================

# Test tolerances
TEST_POSITION_TOLERANCE = 1e-6
TEST_VOLUME_TOLERANCE = 1e-3
TEST_COLOR_TOLERANCE = 1  # For uint8 colors

# Test dimensions
TEST_SMALL_SCENE = (5, 5, 5)
TEST_MEDIUM_SCENE = (10, 10, 10)
TEST_LARGE_SCENE = (20, 20, 20)

# =============================================================================
# Validation Constants
# =============================================================================

# Dimension validation
MIN_BLOCK_DIMENSION = 0.1
MAX_BLOCK_DIMENSION = 100.0

# Position validation
MIN_POSITION_VALUE = -1000.0
MAX_POSITION_VALUE = 1000.0

# Color validation
MIN_COLOR_VALUE = 0
MAX_COLOR_VALUE = 255

# =============================================================================
# Utility Functions
# =============================================================================

def get_column_count_for_width(width: int) -> int:
    """
    Calculate appropriate number of columns based on facade width.

    Args:
        width: Facade width in blocks

    Returns:
        Recommended number of columns
    """
    if width < 5:
        return 2
    elif width < 10:
        return 4
    elif width < 15:
        return 6
    else:
        return min(8, width // 2)


def get_pediment_height_for_width(width: int) -> int:
    """
    Calculate pediment height based on facade width.

    Args:
        width: Facade width in blocks

    Returns:
        Recommended pediment height
    """
    height = int(width * PEDIMENT_SLOPE_RATIO)
    return max(PEDIMENT_MIN_HEIGHT, min(height, PEDIMENT_MAX_HEIGHT))


def validate_dimensions(width: int, height: int, depth: int) -> bool:
    """
    Validate scene dimensions are within acceptable range.

    Args:
        width, height, depth: Dimensions to validate

    Returns:
        True if valid, False otherwise
    """
    return (MIN_DIMENSION <= width <= MAX_DIMENSION and
            MIN_DIMENSION <= height <= MAX_DIMENSION and
            MIN_DIMENSION <= depth <= MAX_DIMENSION)


def validate_seed(seed: int) -> bool:
    """
    Validate seed value is within acceptable range.

    Args:
        seed: Seed value to validate

    Returns:
        True if valid, False otherwise
    """
    return MIN_SEED <= seed <= MAX_SEED


# =============================================================================
# Version Information
# =============================================================================

LEGOGEN_VERSION = "0.5.0-beta"
LEGOGEN_BUILD_DATE = "2025-11-17"
LEGOGEN_AUTHOR = "LegoGen Team"
