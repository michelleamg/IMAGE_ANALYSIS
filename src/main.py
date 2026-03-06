"""
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 2.0
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python"
               Implementación completa incluyendo:
               ✓ Lectura de imágenes (OpenCV)
               ✓ Separación de componentes RGB
               ✓ Escalamiento a grises
               ✓ Mapas de color (OpenCV + personalizado morado-fucsia)
               ✓ Binarización (fijo, Otsu, adaptativo)
               ✓ Histogramas de intensidad
               ✓ Interfaz gráfica con PyQt5 (MVC)
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
| Librerias utilizadas:
- PyQt5: Para la construcción de la interfaz gráfica (widgets, layouts, señales)
- Matplotlib: Para la visualización de histogramas dentro de la aplicación
- OpenCV: Para la manipulación y procesamiento de imágenes (convertir a QImage,
    aplicar mapas de color, etc.)
"""
import sys
from PyQt5.QtWidgets import QApplication
from model.colormap import ImageModel
from model.color_models import ColorModels  
from view.view import MainWindow
from view.rgb_window import RGBComponentsWindow
from view.color_models_window import ColorModelWindow
from controller.controller import ImageController

def main():
    # Arranca la aplicación Qt y conecta modelo, vista y controlador
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    model = ImageModel()
    view = MainWindow()
    controller = ImageController(model, view)
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
