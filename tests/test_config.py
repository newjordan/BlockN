import shutil
import unittest
from pathlib import Path

from config import Config


class TestConfig(unittest.TestCase):
    def test_invalid_dimension_falls_back_and_is_reported(self):
        import os

        previous = os.environ.get("DEFAULT_WIDTH")
        os.environ["DEFAULT_WIDTH"] = "wide"
        try:
            settings = Config()
            self.assertEqual(settings.DEFAULT_WIDTH, 5)
            issues = settings.validate()
        finally:
            if previous is None:
                os.environ.pop("DEFAULT_WIDTH", None)
            else:
                os.environ["DEFAULT_WIDTH"] = previous

        self.assertTrue(any("DEFAULT_WIDTH" in issue for issue in issues))

    def test_output_directories_are_not_created_on_read(self):
        import os

        export_key = os.environ.get("EXPORT_DIR")
        save_key = os.environ.get("SAVE_DIR")
        missing_export = Path("/tmp/blockn-missing-exports")
        missing_save = Path("/tmp/blockn-missing-saves")
        for path in (missing_export, missing_save):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        os.environ["EXPORT_DIR"] = str(missing_export)
        os.environ["SAVE_DIR"] = str(missing_save)
        try:
            settings = Config()
            export_dir = settings.EXPORT_DIR
            save_dir = settings.SAVE_DIR
            self.assertEqual(export_dir, missing_export)
            self.assertEqual(save_dir, missing_save)
            self.assertFalse(export_dir.exists())
            self.assertFalse(save_dir.exists())
        finally:
            if export_key is None:
                os.environ.pop("EXPORT_DIR", None)
            else:
                os.environ["EXPORT_DIR"] = export_key
            if save_key is None:
                os.environ.pop("SAVE_DIR", None)
            else:
                os.environ["SAVE_DIR"] = save_key


if __name__ == "__main__":
    unittest.main()
