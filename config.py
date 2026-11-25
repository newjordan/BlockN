"""
BlockN configuration.

Loads settings from environment variables and an optional ``.env`` file.
The classical generator and workbench read the dimension defaults. The OpenAI
and Blender values are accepted so existing environment files keep loading;
this branch does not call either service.

Usage:
    from config import config
    api_key = config.OPENAI_API_KEY
    blender_path = config.BLENDER_PATH
"""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, just use os.environ
    pass


def _read_int(name: str, default: int) -> int:
    """Return a positive integer setting, or ``default`` when the value is unusable."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _int_issue(name: str, default: int) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return f"{name} must be an integer, got {raw!r}. Using {default}."
    if value <= 0:
        return f"{name} must be positive, got {value}. Using {default}."
    return None


class Config:
    """
    Configuration for the BlockN classical workbench.

    Attributes are loaded from environment variables with fallback defaults.
    """

    # =============================================================================
    # API Configuration
    # =============================================================================

    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        """OpenAI API key for AI Architect mode"""
        return os.getenv('OPENAI_API_KEY', None)

    @property
    def OPENAI_MODEL(self) -> str:
        """OpenAI model to use (default: gpt-4)"""
        return os.getenv('OPENAI_MODEL', 'gpt-4')

    # =============================================================================
    # External Tool Paths
    # =============================================================================

    @property
    def BLENDER_PATH(self) -> Optional[str]:
        """Path to Blender executable for advanced boolean operations"""
        return os.getenv('BLENDER_PATH', None)

    # =============================================================================
    # Application Defaults
    # =============================================================================

    @property
    def DEFAULT_WIDTH(self) -> int:
        """Default scene width in blocks"""
        return _read_int('DEFAULT_WIDTH', 5)

    @property
    def DEFAULT_HEIGHT(self) -> int:
        """Default scene height in blocks"""
        return _read_int('DEFAULT_HEIGHT', 5)

    @property
    def DEFAULT_DEPTH(self) -> int:
        """Default scene depth in blocks"""
        return _read_int('DEFAULT_DEPTH', 5)

    @property
    def DEFAULT_SUPPORT_MODE(self) -> str:
        """Default support checking mode (HYBRID, GRID, OFF)"""
        mode = os.getenv('DEFAULT_SUPPORT_MODE', 'HYBRID').upper()
        if mode not in ['HYBRID', 'GRID', 'OFF']:
            return 'HYBRID'
        return mode

    @property
    def DEFAULT_COLLISION_CHECK(self) -> bool:
        """Default collision checking enabled state"""
        return os.getenv('DEFAULT_COLLISION_CHECK', 'True').lower() in ('true', '1', 'yes')

    # =============================================================================
    # File Paths
    # =============================================================================

    @property
    def PROJECT_ROOT(self) -> Path:
        """Project root directory"""
        return Path(__file__).parent

    @property
    def EXPORT_DIR(self) -> Path:
        """Default directory for GLB exports. The directory is not created until export."""
        export_dir = Path(os.getenv('EXPORT_DIR', 'exports'))
        if not export_dir.is_absolute():
            export_dir = self.PROJECT_ROOT / export_dir
        return export_dir

    @property
    def SAVE_DIR(self) -> Path:
        """Default directory for JSON scene saves. The directory is not created until a save."""
        save_dir = Path(os.getenv('SAVE_DIR', 'saves'))
        if not save_dir.is_absolute():
            save_dir = self.PROJECT_ROOT / save_dir
        return save_dir

    # =============================================================================
    # Performance Settings
    # =============================================================================

    @property
    def USE_PYEMBREE(self) -> bool:
        """Enable pyembree for faster ray tracing (if installed)"""
        return os.getenv('USE_PYEMBREE', 'True').lower() in ('true', '1', 'yes')

    @property
    def MAX_BLOCKS_WARNING(self) -> int:
        """Warn user if block count exceeds this threshold"""
        return _read_int('MAX_BLOCKS_WARNING', 5000)

    # =============================================================================
    # Debug/Development
    # =============================================================================

    @property
    def DEBUG(self) -> bool:
        """Enable debug mode"""
        return os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

    @property
    def LOG_LEVEL(self) -> str:
        """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return level if level in valid_levels else 'INFO'

    @property
    def VERBOSE(self) -> bool:
        """Enable verbose output during generation"""
        return os.getenv('VERBOSE', 'False').lower() in ('true', '1', 'yes')

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of warning/error messages (empty if all OK)
        """
        issues = []

        # Check Blender path if specified
        if self.BLENDER_PATH:
            blender_path = Path(self.BLENDER_PATH)
            if not blender_path.exists():
                issues.append(f"Blender path does not exist: {self.BLENDER_PATH}")
            elif not blender_path.is_file():
                issues.append(f"Blender path is not a file: {self.BLENDER_PATH}")

        # Check OpenAI API key format if specified
        if self.OPENAI_API_KEY:
            if len(self.OPENAI_API_KEY) < 20:
                issues.append("OpenAI API key looks too short (possible misconfiguration)")

        for name, default in (
            ("DEFAULT_WIDTH", 5),
            ("DEFAULT_HEIGHT", 5),
            ("DEFAULT_DEPTH", 5),
            ("MAX_BLOCKS_WARNING", 5000),
        ):
            issue = _int_issue(name, default)
            if issue:
                issues.append(issue)

        return issues

    def get_summary(self) -> str:
        """
        Get human-readable configuration summary.

        Returns:
            Formatted configuration summary string
        """
        lines = [
            "BlockN configuration:",
            f"  Project Root: {self.PROJECT_ROOT}",
            f"  Export Dir: {self.EXPORT_DIR}",
            f"  Save Dir: {self.SAVE_DIR}",
            "",
            "API Configuration:",
            f"  OpenAI API Key: {'✓ Set' if self.OPENAI_API_KEY else '✗ Not set'}",
            f"  OpenAI Model: {self.OPENAI_MODEL}",
            "",
            "External Tools:",
            f"  Blender Path: {self.BLENDER_PATH if self.BLENDER_PATH else '✗ Not set'}",
            "",
            "Defaults:",
            f"  Dimensions: {self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}x{self.DEFAULT_DEPTH}",
            f"  Support Mode: {self.DEFAULT_SUPPORT_MODE}",
            f"  Collision Check: {self.DEFAULT_COLLISION_CHECK}",
            "",
            "Performance:",
            f"  Use pyembree: {self.USE_PYEMBREE}",
            f"  Max blocks warning: {self.MAX_BLOCKS_WARNING}",
            "",
            "Debug:",
            f"  Debug mode: {self.DEBUG}",
            f"  Log level: {self.LOG_LEVEL}",
            f"  Verbose: {self.VERBOSE}",
        ]
        return "\n".join(lines)


# Global config instance
config = Config()


# Convenience function for validation on import
def check_config():
    """
    Check configuration on module import and print warnings.
    Call this at application startup.
    """
    issues = config.validate()
    if issues:
        print("⚠️  Configuration Warnings:")
        for issue in issues:
            print(f"   - {issue}")
        print()


if __name__ == "__main__":
    # Allow running as script to check configuration
    print(config.get_summary())
    print()

    issues = config.validate()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Configuration valid!")
