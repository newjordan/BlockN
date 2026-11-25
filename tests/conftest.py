import os

# Qt widgets created by pytest-qt need a platform plugin. offscreen works
# without a display, including in GitHub Actions.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
