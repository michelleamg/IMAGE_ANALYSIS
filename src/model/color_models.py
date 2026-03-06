"""Modelos de color para transformaciones de imágenes"""
import cv2
import numpy as np

class ColorModels:
    """Clase para manejar diferentes modelos de color y sus transformaciones"""
    
    @staticmethod
    def rgb_to_cmyk(img_rgb):
        """
        Convierte RGB a CMYK
        CMYK: Cyan, Magenta, Yellow, Key (Black)
        """
        if img_rgb is None:
            return None
        
        # Normalizar a [0,1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Separar canales RGB
        r, g, b = cv2.split(img_normalized)
        
        # Calcular K (negro)
        k = 1 - np.maximum(np.maximum(r, g), b)
        
        # Evitar división por cero
        k = np.clip(k, 1e-10, 1)
        
        # Calcular C, M, Y
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
        
        # Escalar a 0-255 para visualización
        c = (c * 255).astype(np.uint8)
        m = (m * 255).astype(np.uint8)
        y = (y * 255).astype(np.uint8)
        k = (k * 255).astype(np.uint8)
        
        return {
            'c': c, 'm': m, 'y': y, 'k': k,
            'combined': cv2.merge([c, m, y, k])
        }
    
    @staticmethod
    def cmyk_to_rgb(cmyk_data):
        """Convierte CMYK a RGB (simplificado)"""
        if cmyk_data is None:
            return None
        
        c = cmyk_data['c'].astype(np.float32) / 255.0
        m = cmyk_data['m'].astype(np.float32) / 255.0
        y = cmyk_data['y'].astype(np.float32) / 255.0
        k = cmyk_data['k'].astype(np.float32) / 255.0
        
        # Convertir a RGB
        r = (1 - c) * (1 - k) * 255
        g = (1 - m) * (1 - k) * 255
        b = (1 - y) * (1 - k) * 255
        
        return cv2.merge([
            r.astype(np.uint8),
            g.astype(np.uint8),
            b.astype(np.uint8)
        ])
    
    @staticmethod
    def rgb_to_hsv(img_rgb):
        """
        Convierte RGB a HSV usando OpenCV
        HSV: Hue (Matiz), Saturation (Saturación), Value (Valor/Brillo)
        """
        if img_rgb is None:
            return None
        
        # OpenCV usa BGR por defecto
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Separar canales
        h, s, v = cv2.split(img_hsv)
        
        return {
            'h': h, 's': s, 'v': v,
            'combined': img_hsv
        }
    
    @staticmethod
    def hsv_to_rgb(img_hsv):
        """Convierte HSV a RGB"""
        if img_hsv is None:
            return None
        
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def rgb_to_hsi(img_rgb):
        """
        Convierte RGB a HSI (Hue, Saturation, Intensity)
        Implementación manual siguiendo fórmulas estándar
        """
        if img_rgb is None:
            return None
        
        # Normalizar a [0,1]
        img = img_rgb.astype(np.float32) / 255.0
        r, g, b = cv2.split(img)
        
        # Calcular Intensidad (I)
        i = (r + g + b) / 3.0
        
        # Calcular Saturación (S)
        # S = 1 - (3 * min(r,g,b)) / (r+g+b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        suma = r + g + b
        s = np.zeros_like(suma)
        mask = suma > 0
        s[mask] = 1 - (3 * min_rgb[mask]) / suma[mask]
        
        # Calcular Matiz (H)
        h = np.zeros_like(suma)
        
        # Para píxeles con saturación > 0
        mask_s = s > 0
        
        # Calcular theta
        numerador = 0.5 * ((r - g) + (r - b))
        denominador = np.sqrt((r - g)**2 + (r - b)*(g - b))
        denominador = np.clip(denominador, 1e-10, None)  # Evitar división por cero
        
        theta = np.arccos(numerador / denominador)
        theta = theta * 180 / np.pi  # Convertir a grados
        
        # Asignar H según la relación entre canales
        h = theta.copy()
        mask_b_gt_g = b > g
        h[mask_b_gt_g] = 360 - theta[mask_b_gt_g]
        
        # Normalizar H a [0,179] para OpenCV
        h_norm = (h * 179 / 360).astype(np.uint8)
        s_norm = (s * 255).astype(np.uint8)
        i_norm = (i * 255).astype(np.uint8)
        
        return {
            'h': h_norm, 's': s_norm, 'i': i_norm,
            'combined': cv2.merge([h_norm, s_norm, i_norm])
        }
    
    @staticmethod
    def rgb_to_yuv(img_rgb):
        """
        Convierte RGB a YUV
        Y: Luminancia, U: Crominancia azul, V: Crominancia roja
        """
        if img_rgb is None:
            return None
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        
        y, u, v = cv2.split(img_yuv)
        
        return {
            'y': y, 'u': u, 'v': v,
            'combined': img_yuv
        }
    
    @staticmethod
    def yuv_to_rgb(img_yuv):
        """Convierte YUV a RGB"""
        if img_yuv is None:
            return None
        
        img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def rgb_to_lab(img_rgb):
        """
        Convierte RGB a LAB (CIE L*a*b*)
        L: Luminosidad, a: Verde-Rojo, b: Azul-Amarillo
        """
        if img_rgb is None:
            return None
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        l, a, b = cv2.split(img_lab)
        
        return {
            'l': l, 'a': a, 'b': b,
            'combined': img_lab
        }
    
    @staticmethod
    def lab_to_rgb(img_lab):
        """Convierte LAB a RGB"""
        if img_lab is None:
            return None
        
        img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def rgb_to_xyz(img_rgb):
        """
        Convierte RGB a XYZ (CIE 1931)
        """
        if img_rgb is None:
            return None
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_xyz = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2XYZ)
        
        x, y, z = cv2.split(img_xyz)
        
        return {
            'x': x, 'y': y, 'z': z,
            'combined': img_xyz
        }
    
    @staticmethod
    def get_channel_visualization(channel_data, channel_name):
        """
        Prepara un canal para visualización (normalización si es necesario)
        """
        if channel_data is None:
            return None
        
        # Normalizar para visualización
        if channel_data.dtype != np.uint8:
            channel_norm = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            channel_norm = channel_data.copy()
        
        return channel_norm