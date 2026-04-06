# Análisis de Imagen Digital — ESCOM IPN

> **Práctica 1 · Práctica 3-a · Práctica 3-b · Práctica 3-c · Práctica 4**  
> Materia: Análisis de Imágenes · Academia de Inteligencia Artificial  
> Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria  
> Mayo 2026

Aplicación de escritorio en Python + PyQt5 para exploración interactiva de imágenes digitales. Incluye pseudocolor, binarización, modelos de color, ruido, operaciones lógico-aritméticas, conteo de objetos por etiquetado de componentes conexas y morfología matemática binaria y en grises (Latticce).

---

## Índice

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Ejecución](#ejecución)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Funcionalidades](#funcionalidades)
   - [Práctica 1 — Análisis de imagen](#práctica-1--análisis-de-imagen)
   - [Práctica 3-a — Ruido](#práctica-3-a--ruido)
   - [Práctica 3-b — Operaciones](#práctica-3-b--operaciones-lógicas-aritméticas-y-relacionales)
   - [Práctica 3-c — Conteo de objetos](#práctica-3-c--conteo-de-objetos)
   - [Práctica 4 — Morfología Matemática](#práctica-4--morfología-matemática)
6. [Exportar resultados](#exportar-resultados)
7. [Arquitectura MVC](#arquitectura-mvc)
8. [Correcciones de bugs](#correcciones-de-bugs)

---

## Requisitos

| Dependencia | Versión mínima | Uso |
|-------------|---------------|-----|
| Python | 3.9+ | Lenguaje base |
| PyQt5 | 5.15+ | Interfaz gráfica |
| opencv-python | 4.5+ | Procesamiento de imagen y morfología |
| numpy | 1.22+ | Operaciones matriciales |
| matplotlib | 3.5+ | Histogramas embebidos |
| scipy | 1.8+ | (Opcional) etiquetado alternativo |

---

## Instalación

```bash
git clone https://github.com/tu-usuario/analisis-imagenes-escom.git
cd analisis-imagenes-escom

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install PyQt5 opencv-python numpy matplotlib scipy
```

---

## Ejecución

```bash
python main.py
```

> ⚠️ Ejecutar siempre desde la raíz del proyecto para que las importaciones relativas funcionen correctamente.

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
│   ├── practica3.py                 # Módulo 3-a/b/c — ruido, operaciones, etiquetado
│   └── practica4.py                 # Módulo 4 — morfología matemática binaria y en grises
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
`RGB · HSV · CMYK · HSI · YUV · LAB · XYZ` — ventana dedicada con tarjetas por canal, estadísticas y guardado en PNG.

#### Histograma detallado
Ventana Qt con 4 modos (original, gray, colormap, binary) y 12 métricas estadísticas por canal.

---

### Práctica 3-a — Ruido

Todas las operaciones trabajan sobre **imagen binaria**. Si no hay binarización activa, se aplica Otsu automáticamente antes de operar.

#### Tipos de ruido

| Tipo | Parámetro controlado |
|------|---------------------|
| Sal y pimienta | fracción de píxeles afectados (slider 1–100 %) |
| Gaussiano | σ proporcional al slider (0.5–50) |

#### Filtros suavizadores

| Filtro | Uso |
|--------|-----|
| Mediana (3×3) | Eliminar ruido sal y pimienta |
| Gaussiano (5×5, σ=1) | Suavizar ruido gaussiano |

Después del filtro, la imagen se re-binariza automáticamente (umbral 127) para mantener la naturaleza binaria.

**Flujo completo documentado en la ventana de resultados:**
```
Original color  →  Binaria (Otsu)  →  Binaria + ruido
Binaria + ruido →  filtrado        →  re-binarizada (limpia)
```

---

### Práctica 3-b — Operaciones lógicas, aritméticas y relacionales

Todas las operaciones trabajan sobre **imagen binaria** (auto-binarización Otsu si no hay activa).

#### Operaciones aritméticas (escalar)

| Operación | Función |
|-----------|---------|
| Suma | `img_bin + escalar` (saturación en 255) |
| Resta | `img_bin − escalar` (saturación en 0) |
| Multiplicación | `img_bin × factor` |

#### Operaciones lógicas

| Operación | Resultado |
|-----------|-----------|
| AND | Solo píxeles activos en **ambas** imágenes binarias |
| OR | Píxeles activos en **al menos una** imagen |
| XOR | Píxeles que **difieren** entre las dos imágenes |
| NOT | **Inversión** bit a bit (unaria, una sola imagen) |

> AND, OR y XOR solicitan una segunda imagen mediante diálogo de archivo. Esa imagen también se binariza con Otsu automáticamente.

#### Operaciones relacionales

| Operación | Resultado sobre imagen binaria |
|-----------|-------------------------------|
| `> umbral` | Máscara: píxeles con valor > umbral |
| `< umbral` | Máscara: píxeles con valor < umbral |
| `≈ umbral` | Máscara: píxeles con valor ≈ umbral ±10 |

---

### Práctica 3-c — Conteo de objetos

Trabaja siempre sobre **imagen binaria**.

| Modo | Conectividad | Efecto |
|------|-------------|--------|
| Vecindad-4 | Solo ortogonal (↑↓←→) | Estricta; diagonales no conectan |
| Vecindad-8 | Ortogonal + diagonal | Inclusiva; formas curvas completas |
| Comparar | Ambas | Grid 2×2 con las diferencias numéricas |

La ventana de resultados muestra: Original color · Binaria · Etiquetado coloreado · Contornos numerados, con estadísticas por objeto (área, centroide, bounding box).

---

### Práctica 4 — Morfología Matemática

Módulo: `model/practica4.py`  
Referencia teórica: [HIPR2 Morphology](https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm)

#### Elemento estructurante (EE)

Configurable desde el sidebar antes de cualquier operación:

| Forma | Función | Notas |
|-------|---------|-------|
| Rect | `kernel_rect(N)` | Cuadrado N×N |
| Cruz | `kernel_cross(N)` | Solo filas/columnas centrales |
| Elipse | `kernel_ellipse(N)` | Forma oval inscrita |

Tamaño disponible: 3, 5, 7, 9, 11 px.

#### Operaciones básicas (binaria y grises)

| Operación | Función | Efecto binario | Efecto grises |
|-----------|---------|----------------|---------------|
| Erosión | `erosion(img, k)` | Reduce regiones blancas | Oscurece (mínimo local) |
| Dilatación | `dilation(img, k)` | Expande regiones blancas | Aclara (máximo local) |
| Apertura | `opening(img, k)` | Erosión → Dilatación; elimina ruido pequeño | Suaviza picos brillantes |
| Cierre | `closing(img, k)` | Dilatación → Erosión; rellena agujeros | Rellena valles oscuros |

> Apertura y Cierre se implementan en modo **tradicional** (composición explícita de erosión y dilatación) para cumplir con el requisito de la práctica.

#### Morfología Binaria avanzada

| Operación | Función | Descripción |
|-----------|---------|-------------|
| Frontera | `boundary(img, k)` | `img − erosión(img)` — extrae contorno interior |
| Hit-or-Miss | `hit_or_miss(img, se_fg, se_bg)` | Detecta patrones específicos de forma |
| Adelgazamiento | `thinning(img, iters)` | Reduce objetos a 1 px de grosor (Zhang-Suen) |
| Esqueleto | `skeleton(img)` | `∪ [Eᵏ(img) − open(Eᵏ(img))]` — Medial Axis Transform |

#### Morfología en Grises — Latticce

| Operación | Función | Descripción |
|-----------|---------|-------------|
| Gradiente simétrico | `gradient_morph(img, k, "symmetric")` | `dilatación − erosión` — todos los bordes |
| Gradiente por erosión | `gradient_morph(img, k, "erosion")` | `img − erosión` — bordes internos |
| Gradiente por dilatación | `gradient_morph(img, k, "dilation")` | `dilatación − img` — bordes externos |
| Top Hat | `top_hat(img, k)` | `img − apertura(img)` — detecta picos brillantes |
| Bot Hat | `bot_hat(img, k)` | `cierre(img) − img` — detecta valles oscuros |
| Suavizado | `morph_smooth(img, k)` | `cierre(apertura(img))` — elimina ruido claro y oscuro |

#### Botones "Todo bin." y "Todo gris"

Ejecutan **todas** las operaciones del modo de un jalón y abren un `ResultsWindow` con el grid completo (2 + N tarjetas), donde cada una tiene su botón individual de guardado PNG.

```
[Todo bin.]  →  Erosión · Dilatación · Apertura · Cierre ·
               Frontera · Hit-or-Miss · Adelgazamiento · Esqueleto

[Todo gris]  →  Erosión · Dilatación · Apertura · Cierre ·
               Grad. simétrico · Grad. erosión · Grad. dilatación ·
               Top Hat · Bot Hat · Suavizado morfológico
```

---

## Exportar resultados

Desde cualquier `ResultsWindow` (prácticas 3-a / 3-b / 3-c / 4):

- **Botón "💾 Guardar PNG"** en cada tarjeta → guarda esa imagen individual sin pérdida.
- **Botón "📁 Exportar todo como PNG"** → elige carpeta y guarda todas las imágenes visibles.

Desde la ventana principal → **Guardar resultado**: exporta el resultado activo como JPG o PNG.

---

## Arquitectura MVC

```
Usuario → View (señales Qt) → Controller → Model (procesa)
                           ← Controller ← Model (datos)
          View ← Controller (show_result / show_status / ResultsWindow)
```

| Componente | Responsabilidad |
|-----------|----------------|
| `ImageModel` (`colormap.py`) | Almacena imágenes, calcula histogramas, aplica transformaciones |
| `practica3.py` | Funciones puras: ruido, operaciones lógicas/aritméticas/relacionales, etiquetado |
| `practica4.py` | Funciones puras: todas las operaciones morfológicas binarias y en grises |
| `ImageController` | Coordina modelo ↔ vista; gestiona `_get_binary_img()` para garantizar base binaria |
| `MainWindow` | Sidebar + área principal; no conoce el modelo |
| `ResultsWindow` | Grid genérico de resultados; recibe lista de `{"title", "img", "badge"}` |
| `HistogramWindow` | Histograma detallado con 12 métricas estadísticas por canal |

### Flujo de binarización automática

Todas las operaciones de las prácticas 3 y 4 que requieren imagen binaria llaman internamente a `_get_binary_img()`:

```python
# Prioridad:
# 1. Si current_map contiene "BINARIA_*" → usa esa binarización
# 2. Si no → aplica Otsu automáticamente, muestra en Resultado y avisa en status bar
```

---

## Correcciones de bugs

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `model/colormap.py` | `get_histogram_result` ignoraba `image_name`; corregido con clave compuesta `(image_name, map_name)` |
| 2 | `controller/controller.py` | `rgb_window` y `model_window` eran variables locales destruidas por el GC; ahora son `self.*` |
| 3 | `view/color_models_window.py` | División por cero en `cmyk_to_rgb_display` cuando `k ≈ 1`; corregido con `np.clip` |

---

## Licencia

Proyecto académico — ESCOM · IPN · Mayo 2026.
