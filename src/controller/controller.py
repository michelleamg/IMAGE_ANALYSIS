"""Controlador del programa, maneja la lógica de interacción entre el modelo y la vista.
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 2.3
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python"
               Versión mejorada con estadísticas de histograma
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
"""
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from view.rgb_window import RGBComponentsWindow
from model.color_models import ColorModels
from view.color_models_window import ColorModelWindow

class ImageController:
    def __init__(self, model, view):
        # Configura el vínculo MVC y estado base de la sesión
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.current_image = None
        self.current_map = None
    
    def load_image(self):
        # Abre un diálogo de archivo y registra la imagen seleccionada
        filepath, _ = QFileDialog.getOpenFileName(
            self.view, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if filepath:
            try:
                name = self.model.load_image(filepath)
                self.current_image = name
                
                # Mostrar imagen original a color
                img = self.model.get_image(name)
                self.view.show_original(img)
                self.view.show_status(f"Imagen a color cargada: {name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))
    
    def convert_to_gray(self):
        # Convierte la imagen cargada a escala de grises y la muestra
        """Convierte la imagen a color a escala de grises"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        # Convertir a grises
        self.model.convert_to_gray(self.current_image)
        
        # Mostrar resultado en grises
        img_gray = self.model.get_gray_image(self.current_image)
        if img_gray is not None:
            self.view.show_result(img_gray)
            self.view.show_status("Imagen convertida a escala de grises")
    
    def show_rgb_components(self):
        """Muestra los componentes RGB por separado en ventanas de PyQt5"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        r, g, b = self.model.get_rgb_components(self.current_image)
        
        # Crear y mostrar la ventana de componentes RGB
        self.rgb_window = RGBComponentsWindow(r, g, b, self.current_image)
        self.rgb_window.show()
        
        self.view.show_status("Componentes RGB mostrados en ventanas separadas")
    
    def apply_map(self):
        # Aplica el mapa de color elegido (OpenCV o personalizado)
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        map_name = self.view.map_combo.currentText()
        
        if "PERSONALIZADO" in map_name:
            result = self.model.apply_personalized_map(self.current_image)
            self.current_map = 'Personalizado'
            map_display = "Personalizado (Morado-Fucsia)"
        else:
            result = self.model.apply_colormap(self.current_image, map_name)
            self.current_map = map_name
            map_display = map_name
        
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Mapa {map_display} aplicado")
    
    def apply_binary(self, method):
        # Ejecuta binarización con el método seleccionado
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        threshold = self.view.threshold_slider.value()
        result = self.model.apply_threshold(self.current_image, threshold, method)
        self.current_map = f'BINARIA_{method}'
        
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Binarización {method} aplicada")
    
    def save_result(self):
        # Exporta a disco el resultado actualmente mostrado
        if not self.current_image or not self.current_map:
            QMessageBox.warning(self.view, "Aviso", "No hay resultado para guardar")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self.view, "Guardar imagen", 
            f"{self.current_image}_{self.current_map}.jpg",
            "JPEG (*.jpg);;PNG (*.png)"
        )
        
        if filepath:
            cv_img = self.model.get_result(self.current_image, self.current_map)
            if cv_img is not None:
                cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, cv_img_bgr)
                self.view.show_status(f"Guardado: {os.path.basename(filepath)}")
    
    # ===== NUEVA FUNCIÓN: Calcular estadísticas del histograma =====
    def calculate_histogram_stats(self, hist_data):
        """
        Calcula estadísticas descriptivas a partir de los datos del histograma
        Retorna: media, varianza, desviación estándar, asimetría, curtosis,
                energía, entropía, y percentiles
        """
        if hist_data is None:
            return None
        
        # Asegurar que sea un array de numpy
        if isinstance(hist_data, dict):
            # Para histogramas RGB, usar el canal de intensidad promedio
            if 'r' in hist_data and 'g' in hist_data and 'b' in hist_data:
                # Promediar los histogramas de los tres canales
                hist = (hist_data['r'] + hist_data['g'] + hist_data['b']) / 3
            else:
                return None
        else:
            hist = hist_data
        
        # Normalizar el histograma para obtener probabilidades
        hist = hist.astype(np.float64)
        total_pixels = np.sum(hist)
        if total_pixels == 0:
            return None
        
        prob = hist / total_pixels
        niveles = np.arange(256)
        
        # Media
        media = np.sum(niveles * prob)
        
        # Varianza y desviación estándar
        varianza = np.sum(((niveles - media) ** 2) * prob)
        desviacion = np.sqrt(varianza)
        
        # Asimetría (skewness)
        asimetria = np.sum(((niveles - media) ** 3) * prob) / (desviacion ** 3) if desviacion > 0 else 0
        
        # Curtosis (kurtosis)
        curtosis = np.sum(((niveles - media) ** 4) * prob) / (desviacion ** 4) if desviacion > 0 else 0
        curtosis = curtosis - 3  # Curtosis con referencia a la normal
        
        # Energía (uniformidad)
        energia = np.sum(prob ** 2)
        
        # Entropía
        entropia = -np.sum(prob * np.log2(prob + 1e-10))  # +1e-10 para evitar log(0)
        
        # Percentiles importantes
        prob_acum = np.cumsum(prob)
        percentil_25 = np.searchsorted(prob_acum, 0.25)
        percentil_50 = np.searchsorted(prob_acum, 0.50)  # Mediana
        percentil_75 = np.searchsorted(prob_acum, 0.75)
        
        # Moda (valor más frecuente)
        moda = np.argmax(hist)
        
        # Rango dinámico
        niveles_no_cero = niveles[hist > 0]
        rango_dinamico = niveles_no_cero[-1] - niveles_no_cero[0] if len(niveles_no_cero) > 0 else 0
        
        stats = {
            'media': media,
            'mediana': percentil_50,
            'varianza': varianza,
            'desviacion': desviacion,
            'asimetria': asimetria,
            'curtosis': curtosis,
            'energia': energia,
            'entropia': entropia,
            'moda': moda,
            'percentil_25': percentil_25,
            'percentil_75': percentil_75,
            'rango_dinamico': rango_dinamico,
            'min': niveles_no_cero[0] if len(niveles_no_cero) > 0 else 0,
            'max': niveles_no_cero[-1] if len(niveles_no_cero) > 0 else 0
        }
        
        return stats
    
    # ===== NUEVA FUNCIÓN: Formatear estadísticas para mostrar =====
    def format_stats_text(self, stats, channel_name=""):
        """Convierte las estadísticas en texto formateado para mostrar"""
        if stats is None:
            return "No hay datos suficientes"
        
        text = f"📊 Estadísticas {channel_name}:\n"
        text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"Media: {stats['media']:.2f}\n"
        text += f"Mediana: {stats['mediana']:.1f}\n"
        text += f"Moda: {stats['moda']}\n"
        text += f"Varianza: {stats['varianza']:.2f}\n"
        text += f"Desviación estándar: {stats['desviacion']:.2f}\n"
        text += f"Asimetría: {stats['asimetria']:.3f}\n"
        text += f"Curtosis: {stats['curtosis']:.3f}\n"
        text += f"Energía: {stats['energia']:.6f}\n"
        text += f"Entropía: {stats['entropia']:.3f} bits\n"
        text += f"Rango dinámico: [{stats['min']}, {stats['max']}]\n"
        text += f"Percentil 25: {stats['percentil_25']}\n"
        text += f"Percentil 75: {stats['percentil_75']}\n"
        
        return text
    
    def show_histogram(self):
        # Variante optimizada que obtiene histogramas según pestaña y tipo de mapa
        """Muestra el histograma según el tipo de imagen actual con estadísticas"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        # Verificar qué pestaña está activa
        current_tab = self.view.tabs.currentIndex()
        
        if current_tab == 0:  # Pestaña Original
            # Mostrar histograma RGB de la imagen original a color
            hists = self.model.get_histogram_rgb(self.current_image)
            if hists is None:
                return
                
            img = self.model.get_image(self.current_image)
            
            # Calcular estadísticas para cada canal
            stats_r = self.calculate_histogram_stats(hists['r'])
            stats_g = self.calculate_histogram_stats(hists['g'])
            stats_b = self.calculate_histogram_stats(hists['b'])
            
            # Crear figura con 2 filas: imagen + histograma y estadísticas
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f"Análisis de Histograma - {self.current_image}", fontsize=14, fontweight='bold')
            
            # Grid: 2 filas, 3 columnas
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
            
            # Fila 1: Imagen original y histogramas
            ax_img = fig.add_subplot(gs[0, :2])
            ax_img.imshow(img)
            ax_img.set_title('Imagen original a color')
            ax_img.axis('off')
            
            ax_hist = fig.add_subplot(gs[0, 2])
            ax_hist.plot(hists['r'], color='red', alpha=0.7, label='Rojo', linewidth=1.5)
            ax_hist.plot(hists['g'], color='green', alpha=0.7, label='Verde', linewidth=1.5)
            ax_hist.plot(hists['b'], color='blue', alpha=0.7, label='Azul', linewidth=1.5)
            ax_hist.set_title('Histograma RGB')
            ax_hist.set_xlabel('Nivel de intensidad')
            ax_hist.set_ylabel('Frecuencia')
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)
            
            # Fila 2: Estadísticas para cada canal
            ax_stats_r = fig.add_subplot(gs[1, 0])
            ax_stats_r.axis('off')
            ax_stats_r.text(0.1, 0.95, self.format_stats_text(stats_r, "Canal Rojo"), 
                          transform=ax_stats_r.transAxes, fontsize=9, fontfamily='monospace',
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.8))
            
            ax_stats_g = fig.add_subplot(gs[1, 1])
            ax_stats_g.axis('off')
            ax_stats_g.text(0.1, 0.95, self.format_stats_text(stats_g, "Canal Verde"), 
                          transform=ax_stats_g.transAxes, fontsize=9, fontfamily='monospace',
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#eeffee', alpha=0.8))
            
            ax_stats_b = fig.add_subplot(gs[1, 2])
            ax_stats_b.axis('off')
            ax_stats_b.text(0.1, 0.95, self.format_stats_text(stats_b, "Canal Azul"), 
                          transform=ax_stats_b.transAxes, fontsize=9, fontfamily='monospace',
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#eeeeff', alpha=0.8))
            
        else:  # Pestaña Resultado
            if self.current_map:
                # Obtener histograma del resultado
                hists = self.model.get_histogram_result(self.current_image, self.current_map)
                img_result = self.model.get_result(self.current_image, self.current_map)
                
                if hists is None or img_result is None:
                    QMessageBox.warning(self.view, "Aviso", "No hay resultado para mostrar")
                    return
                
                # Determinar título según tipo de resultado
                if 'BINARIA' in self.current_map:
                    titulo = f"Análisis de Histograma - {self.current_map}"
                else:
                    titulo = f"Análisis de Histograma RGB - {self.current_map}"
                
                # Crear figura
                fig = plt.figure(figsize=(14, 8))
                fig.suptitle(titulo, fontsize=14, fontweight='bold')
                
                if 'BINARIA' in self.current_map:
                    # Para binarizada, usar solo estadísticas del histograma de intensidad
                    hist = hists['r']  # Usamos cualquier canal (son iguales)
                    stats = self.calculate_histogram_stats(hist)
                    
                    # Grid: 2 filas, 2 columnas
                    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
                    
                    # Imagen resultado
                    ax_img = fig.add_subplot(gs[0, 0])
                    ax_img.imshow(img_result)
                    ax_img.set_title(f'Imagen binarizada ({self.current_map.split("_")[1]})')
                    ax_img.axis('off')
                    
                    # Histograma
                    ax_hist = fig.add_subplot(gs[0, 1])
                    ax_hist.bar([0, 255], [hist[0], hist[255]], 
                              color=['black', 'white'], edgecolor='black', alpha=0.7, width=50)
                    ax_hist.set_title('Histograma (0 = Negro, 255 = Blanco)')
                    ax_hist.set_xlabel('Valor de píxel')
                    ax_hist.set_ylabel('Frecuencia')
                    ax_hist.grid(True, alpha=0.3)
                    
                    # Estadísticas
                    ax_stats = fig.add_subplot(gs[1, :])
                    ax_stats.axis('off')
                    ax_stats.text(0.5, 0.5, self.format_stats_text(stats, ""), 
                                transform=ax_stats.transAxes, fontsize=10, fontfamily='monospace',
                                horizontalalignment='center', verticalalignment='center',
                                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
                    
                else:
                    # Para mapas de color, mostrar estadísticas RGB
                    stats_r = self.calculate_histogram_stats(hists['r'])
                    stats_g = self.calculate_histogram_stats(hists['g'])
                    stats_b = self.calculate_histogram_stats(hists['b'])
                    
                    # Grid: 2 filas, 3 columnas
                    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
                    
                    # Imagen resultado
                    ax_img = fig.add_subplot(gs[0, :2])
                    ax_img.imshow(img_result)
                    ax_img.set_title(f'Imagen con mapa {self.current_map}')
                    ax_img.axis('off')
                    
                    # Histograma RGB
                    ax_hist = fig.add_subplot(gs[0, 2])
                    ax_hist.plot(hists['r'], color='red', alpha=0.7, label='Rojo', linewidth=1.5)
                    ax_hist.plot(hists['g'], color='green', alpha=0.7, label='Verde', linewidth=1.5)
                    ax_hist.plot(hists['b'], color='blue', alpha=0.7, label='Azul', linewidth=1.5)
                    ax_hist.set_title('Histograma RGB')
                    ax_hist.set_xlabel('Nivel de intensidad')
                    ax_hist.set_ylabel('Frecuencia')
                    ax_hist.legend(fontsize=8)
                    ax_hist.grid(True, alpha=0.3)
                    
                    # Estadísticas para cada canal
                    ax_stats_r = fig.add_subplot(gs[1, 0])
                    ax_stats_r.axis('off')
                    ax_stats_r.text(0.1, 0.95, self.format_stats_text(stats_r, "Canal Rojo"), 
                                  transform=ax_stats_r.transAxes, fontsize=9, fontfamily='monospace',
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.8))
                    
                    ax_stats_g = fig.add_subplot(gs[1, 1])
                    ax_stats_g.axis('off')
                    ax_stats_g.text(0.1, 0.95, self.format_stats_text(stats_g, "Canal Verde"), 
                                  transform=ax_stats_g.transAxes, fontsize=9, fontfamily='monospace',
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#eeffee', alpha=0.8))
                    
                    ax_stats_b = fig.add_subplot(gs[1, 2])
                    ax_stats_b.axis('off')
                    ax_stats_b.text(0.1, 0.95, self.format_stats_text(stats_b, "Canal Azul"), 
                                  transform=ax_stats_b.transAxes, fontsize=9, fontfamily='monospace',
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#eeeeff', alpha=0.8))
                
            else:
                # Si no hay mapa seleccionado, mostrar histograma de grises
                hist = self.model.get_histogram_gray(self.current_image)
                img_gray = self.model.get_gray_image(self.current_image)
                
                # Calcular estadísticas
                stats = self.calculate_histogram_stats(hist)
                
                # Crear figura
                fig = plt.figure(figsize=(14, 8))
                fig.suptitle("Análisis de Histograma - Escala de Grises", fontsize=14, fontweight='bold')
                
                # Grid: 2 filas, 2 columnas
                gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
                
                # Imagen en grises
                ax_img = fig.add_subplot(gs[0, 0])
                ax_img.imshow(img_gray, cmap='gray')
                ax_img.set_title('Imagen en escala de grises')
                ax_img.axis('off')
                
                # Histograma
                ax_hist = fig.add_subplot(gs[0, 1])
                ax_hist.plot(hist, color='black', linewidth=1.5)
                ax_hist.fill_between(range(256), hist, alpha=0.3, color='gray')
                ax_hist.set_title('Histograma de intensidades')
                ax_hist.set_xlabel('Nivel de gris')
                ax_hist.set_ylabel('Frecuencia')
                ax_hist.grid(True, alpha=0.3)
                
                # Líneas verticales para estadísticas importantes
                if stats:
                    ax_hist.axvline(x=stats['media'], color='red', linestyle='--', linewidth=2, 
                                   label=f"Media: {stats['media']:.1f}")
                    ax_hist.axvline(x=stats['mediana'], color='green', linestyle=':', linewidth=2, 
                                   label=f"Mediana: {stats['mediana']}")
                    ax_hist.axvline(x=stats['moda'], color='blue', linestyle='-.', linewidth=2, 
                                   label=f"Moda: {stats['moda']}")
                    ax_hist.legend(fontsize=8)
                
                # Estadísticas detalladas
                ax_stats = fig.add_subplot(gs[1, :])
                ax_stats.axis('off')
                ax_stats.text(0.5, 0.5, self.format_stats_text(stats, "Intensidad"), 
                            transform=ax_stats.transAxes, fontsize=10, fontfamily='monospace',
                            horizontalalignment='center', verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
        
        plt.tight_layout()
        plt.show()
        self.view.show_status("Histograma con estadísticas mostrado")
    
    def apply_color_model(self):
        """Aplica transformación a diferentes modelos de color"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        model_name = self.view.models_combo.currentText()
        img_rgb = self.model.get_image(self.current_image)
        
        if img_rgb is None:
            return
        
        model_data = None
        
        try:
            if "RGB" in model_name:
                # Para RGB, solo mostrar los canales originales
                r, g, b = cv2.split(img_rgb)
                model_data = {
                    'r': r, 'g': g, 'b': b,
                    'combined': img_rgb.copy()
                }
                display_name = "RGB"
            
            elif "CMYK" in model_name:
                model_data = ColorModels.rgb_to_cmyk(img_rgb)
                display_name = "CMYK"
            
            elif "HSV" in model_name:
                model_data = ColorModels.rgb_to_hsv(img_rgb)
                display_name = "HSV"
            
            elif "HSI" in model_name:
                model_data = ColorModels.rgb_to_hsi(img_rgb)
                display_name = "HSI"
            
            elif "YUV" in model_name:
                model_data = ColorModels.rgb_to_yuv(img_rgb)
                display_name = "YUV"
            
            elif "LAB" in model_name:
                model_data = ColorModels.rgb_to_lab(img_rgb)
                display_name = "LAB"
            
            elif "XYZ" in model_name:
                model_data = ColorModels.rgb_to_xyz(img_rgb)
                display_name = "XYZ"
            
            if model_data:
                # Crear y mostrar ventana del modelo
                self.model_window = ColorModelWindow(model_data, display_name, img_rgb)
                self.model_window.show()
                self.view.show_status(f"Modelo {display_name} aplicado correctamente")
            else:
                QMessageBox.warning(self.view, "Error", f"No se pudo aplicar el modelo {model_name}")
        
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Error al aplicar modelo: {str(e)}")