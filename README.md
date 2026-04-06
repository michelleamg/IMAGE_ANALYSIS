# Análisis de Imagen Digital — ESCOM IPN

> **Práctica 1 · Práctica 3-a · Práctica 3-b · Práctica 3-c**  
> Materia: Análisis de Imágenes · Academia de Inteligencia Artificial  
> Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria  
> Marzo 2026

Aplicación de escritorio en Python + PyQt5 para exploración interactiva de imágenes digitales. Incluye pseudocolor, binarización, modelos de color, ruido, operaciones lógico-aritméticas y conteo de objetos por etiquetado de componentes conexas.

---

## Índice

1. [Captura de pantalla](#captura-de-pantalla)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Ejecución](#ejecución)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Funcionalidades](#funcionalidades)
   - [Práctica 1 — Análisis de imagen](#práctica-1--análisis-de-imagen)
   - [Práctica 3-a — Ruido](#práctica-3-a--ruido)
   - [Práctica 3-b — Operaciones](#práctica-3-b--operaciones-lógicas-aritméticas-y-relacionales)
   - [Práctica 3-c — Conteo de objetos](#práctica-3-c--conteo-de-objetos)
7. [Exportar resultados](#exportar-resultados)
8. [Arquitectura MVC](#arquitectura-mvc)
9. [Correcciones de bugs](#correcciones-de-bugs)

---

## Captura de pantalla

```
┌─────────────────────────────┬──────────────────────────────────────────────┐
│  Sidebar azul oscuro         │  Original | Resultado | Comparar             │
│                             │                                              │
│  📁 Archivo                 │                                              │
│  ┄ Cargar imagen            │         [imagen activa]                      │
│  ┄ Guardar resultado        │                                              │
│                             │                                              │
│  🎨 Mapa de color           │                                              │
│  ┄ [combo] TURBO ▾          │                                              │
│  ┄ [Aplicar mapa]           ├──────────────────────────────────────────────┤
│                             │  Histograma en vivo (matplotlib embebido)    │
│  ⚡ 3-b Operaciones         │                                              │
│  ┄ [+ Suma] [− Resta] [×]   └──────────────────────────────────────────────┘
│  ┄ [AND] [OR] [XOR] [NOT]
│  ┄ [> Mayor] [< Menor] [≈]
│
│  🔬 3-c Conteo de objetos
│  ┄ [Vecindad 4] [Vecindad 8] [Comparar]
└─────────────────────────────
```

---

## Requisitos

| Dependencia | Versión mínima | Uso |
|-------------|---------------|-----|
| Python | 3.9+ | Lenguaje base |
| PyQt5 | 5.15+ | Interfaz gráfica |
| opencv-python | 4.5+ | Procesamiento de imagen |
| numpy | 1.22+ | Operaciones matriciales |
| matplotlib | 3.5+ | Histogramas embebidos |
| scipy | 1.8+ | (Opcional) etiquetado alternativo |

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/analisis-imagenes-escom.git
cd analisis-imagenes-escom

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install PyQt5 opencv-python numpy matplotlib scipy
```

---

## Ejecución

```bash
# Desde la raíz del proyecto
python main.py
```

> ⚠️ Ejecutar siempre desde la raíz para que las importaciones relativas funcionen correctamente.

---

## Estructura del proyecto

```
analisis-imagenes-escom/
│
├── main.py                          # Punto de entrada
│
├── model/
│   ├── colormap.py                  # ImageModel — carga, histogramas, mapas, binarización
│   ├── color_models.py              # ColorModels — HSV, CMYK, YCbCr, LAB, XYZ…
│   └── practica3.py                 # Módulo 3-a/b/c — ruido, operaciones, etiquetado
│
├── controller/
│   └── controller.py                # ImageController — lógica y coordinación MVC
│
└── view/
    ├── view.py                      # MainWindow — ventana principal con sidebar
    ├── rgb_window.py                # RGBComponentsWindow — canales R, G, B
    ├── color_models_window.py       # ColorModelWindow — modelos de color
    ├── histogram_window.py          # HistogramWindow — histograma detallado Qt
    └── results_window.py            # ResultsWindow — grid de resultados con export PNG
```

---

## Funcionalidades

### Práctica 1 — Análisis de imagen

#### Interfaz principal
- **Sidebar** azul oscuro (320 px) con secciones colapsables tipo VS Code
- **Tres pestañas**: Original · Resultado · Comparar (splitter lado a lado)
- **Histograma en vivo** embebido (matplotlib en canvas Qt), se refresca automáticamente

#### Mapas de pseudocolor

| Nombre | Origen | Descripción |
|--------|--------|-------------|
| `TWILIGHT` | OpenCV | Azules y violetas |
| `TURBO` | OpenCV | Alta saturación, arco iris mejorado |
| `VIRDIS` | OpenCV | Perceptualmente uniforme |
| `PINK` | OpenCV | Tonos rosados |
| `INFERNO` | OpenCV | Negro → rojo → amarillo |
| `WINTER` | OpenCV | Azul frío → verde |
| `HSV` | OpenCV | Ciclo completo de tono |
| `PARULA` | OpenCV | Azul → amarillo (estilo MATLAB) |
| `PERSONALIZADO` | Matplotlib | Morado → Fucsia |

#### Binarización

| Método | Descripción |
|--------|-------------|
| Fijo | Umbral manual (slider 0–255) |
| Otsu | Umbral óptimo automático por varianza intraclase |
| Adaptativo | Umbralización local 11×11 px (iluminación desigual) |

#### Modelos de color
Ventana dedicada con tarjetas por canal, estadísticas min/max/media y guardado en PNG:
`RGB · HSV · CMYK · HSI · YUV · LAB · XYZ`

#### Histograma detallado
Ventana Qt dedicada con 4 modos:

| Modo | Contenido |
|------|-----------|
| `original` | Pestaña RGB superpuesto + pestaña individual por canal |
| `gray` | Curva gris con líneas media/mediana/moda e IQR sombreado |
| `colormap` | Igual que original sobre resultado pseudocoloreado |
| `binary` | Barras con conteo de píxeles negros y blancos |

12 métricas estadísticas por canal: media, mediana, moda, varianza, desv. estándar, asimetría, curtosis, entropía, energía, rango dinámico, P25, P75.

---

### Práctica 3-a — Ruido

Módulo: `model/practica3.py`

#### Tipos de ruido

| Tipo | Función | Parámetro |
|------|---------|-----------|
| Sal y pimienta | `add_salt_pepper(img, amount)` | `amount` = fracción de píxeles afectados (0–1) |
| Gaussiano | `add_gaussian_noise(img, sigma)` | `sigma` = desviación estándar (0.5–50) |

#### Filtros suavizadores

| Filtro | Función | Uso recomendado |
|--------|---------|-----------------|
| Mediana | `apply_median_filter(img, ksize=3)` | Eliminar ruido sal y pimienta |
| Gaussiano | `apply_gaussian_filter(img, ksize=5, sigma=1)` | Suavizar ruido gaussiano |

**Flujo recomendado según las prácticas:**
```
imagen original → agregar ruido → etiquetado (antes) → filtro → etiquetado (después)
```

La ventana de resultados muestra la imagen antes y después con tarjeta exportable en PNG.

---

### Práctica 3-b — Operaciones lógicas, aritméticas y relacionales

Módulo: `model/practica3.py`

#### Operaciones aritméticas (escalar)

```python
arith_add_scalar(img, valor)        # img + valor  (saturación en 255)
arith_subtract_scalar(img, valor)   # img − valor  (saturación en 0)
arith_multiply_scalar(img, factor)  # img × factor (saturación en 255)
```

También disponibles entre dos imágenes:
```python
arith_add_images(img1, img2)
arith_subtract_images(img1, img2)
arith_multiply_images(img1, img2)
```

#### Operaciones lógicas

| Operación | Función | Descripción |
|-----------|---------|-------------|
| AND | `logic_and(img1, img2)` | Solo píxeles activos en **ambas** imágenes |
| OR | `logic_or(img1, img2)` | Píxeles activos en **al menos una** imagen |
| XOR | `logic_xor(img1, img2)` | Píxeles que **difieren** entre las imágenes |
| NOT | `logic_not(img)` | **Inversión** bit a bit (unaria) |

> AND, OR y XOR solicitan una segunda imagen mediante diálogo de archivo al ejecutarse.  
> Se recomienda aplicar umbralizado simple antes de estas operaciones.

#### Operaciones relacionales

| Operación | Función | Resultado |
|-----------|---------|-----------|
| `> umbral` | `relational_greater(img, threshold)` | Máscara: píxeles más claros que el umbral |
| `< umbral` | `relational_less(img, threshold)` | Máscara: píxeles más oscuros que el umbral |
| `≈ umbral` | `relational_equal(img, threshold, tol=10)` | Máscara: píxeles ≈ umbral ±10 |

El umbral se toma del slider de binarización del sidebar.

---

### Práctica 3-c — Conteo de objetos

Módulo: `model/practica3.py`

#### Etiquetado de componentes conexas

```python
result = connected_components(binary_img, connectivity=8)
# result["num_objects"]  → número de objetos detectados
# result["labels_img"]   → imagen RGB coloreada por etiqueta
# result["contours_img"] → imagen con contornos verdes numerados
# result["stats"]        → lista de {"id", "area", "cx", "cy", "x","y","w","h"}
```

| Vecindad | Conectividad | Efecto |
|----------|-------------|--------|
| Vecindad-4 | Solo ortogonal (↑↓←→) | Más estricta; objetos con conexión diagonal se separan |
| Vecindad-8 | Ortogonal + diagonal | Más inclusiva; detecta formas curvas completas |

#### Comparación V4 vs V8

```python
r4, r8 = compare_connectivity(binary_img)
```

La ventana **Comparar** muestra los 4 resultados (etiquetado y contornos de cada vecindad) en un grid 2×2 con las diferencias numéricas en el panel izquierdo.

**Flujo completo de la práctica:**
```
imagen → binarizar (Otsu/Adaptativo) → etiquetar V4 → etiquetar V8
       → agregar ruido sal & pimienta → etiquetar V4/V8 → filtro mediana
       → etiquetar V4/V8 (comparar antes/después del filtro)
```

---

## Exportar resultados

### Desde ResultsWindow (prácticas 3-a / 3-b / 3-c)
- **Botón "💾 Guardar PNG"** en cada tarjeta → guarda esa imagen individual
- **Botón "📁 Exportar todo como PNG"** → elige carpeta y guarda todos los resultados visibles

Todas las exportaciones son PNG sin pérdida (`cv2.imwrite` con extensión `.png`).

### Desde la ventana principal
- **Guardar resultado**: exporta el resultado activo como JPG o PNG

### Desde ColorModelWindow
- **Botón Guardar [canal] como PNG**: guarda el canal individual en escala de grises

---

## Arquitectura MVC

```
Usuario → View (señales Qt) → Controller → Model (procesa)
                           ← Controller ← Model (datos)
          View ← Controller (show_result / show_status / ResultsWindow)
```

| Componente | Responsabilidad |
|-----------|----------------|
| `ImageModel` | Almacena imágenes, calcula histogramas, aplica transformaciones |
| `practica3` | Funciones puras de ruido, operaciones y etiquetado |
| `ImageController` | Coordina modelo ↔ vista, abre ventanas secundarias |
| `MainWindow` | Sidebar + área principal; no conoce el modelo |
| `ResultsWindow` | Grid genérico de resultados; recibe lista de `{"title", "img", "badge"}` |
| `HistogramWindow` | Histograma detallado con estadísticas por canal |

---

## Correcciones de bugs

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `model/colormap.py` | `get_histogram_result` ignoraba `image_name`; corregido con clave compuesta `(image_name, map_name)` |
| 2 | `controller/controller.py` | `rgb_window` y `model_window` eran variables locales; destruidas por el garbage collector. Ahora son `self.*` |
| 3 | `view/color_models_window.py` | División por cero en `cmyk_to_rgb_display` cuando `k ≈ 1`. Corregido con `np.clip` |

---

## Licencia

Proyecto académico — ESCOM · IPN · Marzo 2026.
