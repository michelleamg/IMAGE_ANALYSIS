import sys
from PyQt5.QtWidgets import QApplication

from models.image_model import ImageModel
from views.main_view import MainView
from controllers.main_controller import MainController

def main():
    app = QApplication(sys.argv)

    model = ImageModel()
    view = MainView()
    controller = MainController(model, view)

    view.set_controller(controller)
    view.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()