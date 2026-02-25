import sys
from PyQt5.QtWidgets import QApplication
from model.colormap import ImageModel
from view.view import MainWindow
from controller.controller import ImageController

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    model = ImageModel()
    view = MainWindow()
    controller = ImageController(model, view)
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()