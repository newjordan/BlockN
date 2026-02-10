import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
