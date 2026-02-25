import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class ImageModel:
    def __init__(self):
        self.images = {}#diccionario para almacenar imágenes cargadas
        self.results = {}#diccionario para almacenar resultados de mapas aplicados
        
        # Mapas de OpenCV
        self.colormaps = {
            "TWILIGHT": cv2.COLORMAP_TWILIGHT, #1
            "TURBO": cv2.COLORMAP_TURBO,#2
            "VIRDIS": cv2.COLORMAP_VIRIDIS,#3
            "PINK": cv2.COLORMAP_PINK,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "WINTER": cv2.COLORMAP_WINTER,
            "HSV": cv2.COLORMAP_HSV,
            "PARULA": cv2.COLORMAP_PARULA
        }
        
        # Mapa 
        colores_personalizado = [
            (0.4, 0.0, 0.6),   # Morado profundo
            (0.6, 0.2, 0.8),   # Morado medio
            (0.8, 0.3, 0.9),   # Morado claro
            (1.0, 0.3, 0.6),   # Fucsia
        ]
        self.mapa_personalizado = LinearSegmentedColormap.from_list("Personalizado", colores_personalizado, N=256) #Crear mapa personalizado con 256 colores para una transición suave
    
    def load_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # Cargar en escala de grises
        if img is None:#Intentar cargar como imagen a color si falla la carga en escala de grises
            img_color = cv2.imread(filepath)#Cargar a color
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        name = filepath.split('/')[-1].split('.')[0]#Usar solo el nombre del archivo sin extensión
        self.images[name] = img
        return name
    
    def apply_colormap(self, image_name, map_name):
        img = self.images[image_name]
        result = cv2.applyColorMap(img, self.colormaps[map_name]) #Aplicar mapa de OpenCV
        self.results[map_name] = result #Guardar resultado en formato compatible con OpenCV para guardarlo como JPEG o PNG
        return result
    
    def apply_personalized_map(self, image_name):
        """Aplica el mapa personalizado a la imagen dada y guarda el resultado en un formato compatible con OpenCV. y para guardarla como JPEG o PNG."""
        img = self.images[image_name]#Imagen a escala de grises
        img_norm = img / 255.0 # Normalizar a [0, 1] de matplotlib
        colored = self.mapa_personalizado(img_norm) # Aplicar mapa personalizado 
        colored_255 = (colored[:, :, :3] * 255).astype(np.uint8)#Convertir a RGB de 8 bits para OpenCV 
        result = cv2.cvtColor(colored_255, cv2.COLOR_RGB2BGR)#Convertir a BGR para OpenCV
        self.results['Personalizado'] = result #Guardar resultado en formato compatible con OpenCV para guardarlo como JPEG o PNG
        return result
    
    def get_image(self, image_name):
        img = self.images[image_name]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)#Convertir a RGB para mostrar en PyQt
    
    def get_result(self, image_name, map_name):
        if map_name in self.results:
            img = self.results[map_name]
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convertir a RGB para mostrar en PyQt
        return None