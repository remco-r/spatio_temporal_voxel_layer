"""Entry point for the FOV Boundary Tool."""

import sys


def main():
    from PyQt5.QtWidgets import QApplication
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("FOV Boundary Tool")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
