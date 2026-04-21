# Ver 1.0
# Autor: mcruzm@ipn.mx
# -*- coding: utf-8 -*-
"""
Práctica de laboratorio: Transformaciones en el dominio de la frecuencia (FFT y DCT)
ISC 7º semestre — Python

Este script incluye:
- Carga de imagen (PIL) y conversión a escala de grises.
- Cálculo y visualización del espectro de magnitud y fase (FFT 2D).
- Filtros en el dominio de la frecuencia: ideal, gaussiano y Butterworth (pasa bajas / pasa altas).
- Compresión simple basada en DCT por bloques de 8x8 con cuantización tipo JPEG.
- Reconstrucción y métricas (PSNR).

Uso rápido (ejemplos):
python practica_frecuencia_ISC.py --imagen data/ejemplo.jpg --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --dct_q 0.5
python practica_frecuencia_ISC.py --filtro ideal --tipo highpass --cutoff 0.05 --dct_q 0.8

Si no se proporciona imagen, se genera una imagen de prueba (tablero damero + círculos).
"""

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ----------------------------- Utilidades de imagen -----------------------------

def cargar_imagen(ruta=None, tamaño_max=512):
    """Carga una imagen y la convierte a escala de grises float32 [0,1].
    Si no hay ruta, genera una imagen sintética de prueba.
    """
    if ruta and os.path.exists(ruta):
        img = Image.open(ruta).convert('L')
    else:
        # Imagen sintética: damero + formas
        n = tamaño_max
        img = Image.new('L', (n, n), color=0)
        # damero
        tile = n // 16
        for i in range(0, n, tile):
            for j in range(0, n, tile):
                if ((i//tile) + (j//tile)) % 2 == 0:
                    ImageDraw.Draw(img).rectangle([i, j, i+tile-1, j+tile-1], fill=180)
        # círculos
        draw = ImageDraw.Draw(img)
        for r,c in [(n//4, n//4), (3*n//4, 3*n//4), (n//4, 3*n//4)]:
            draw.ellipse([r-40, c-40, r+40, c+40], outline=255, width=3)
    # redimensionar si es muy grande
    if max(img.size) > tamaño_max:
        scale = tamaño_max / float(max(img.size))
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# ----------------------------- FFT y filtros -----------------------------

def fft2_imagen(img):
    """Calcula FFT 2D, espectros de magnitud (log) y fase, y devuelve el fft desplazado."""
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitud = np.log(1 + np.abs(Fshift))
    fase = np.angle(Fshift)
    return F, Fshift, magnitud, fase


def crear_mascara(img_shape, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2):
    """Crea una máscara de filtro en el dominio de la frecuencia.
    - filtro: 'ideal', 'gaussiano', 'butterworth'
    - tipo: 'lowpass' o 'highpass'
    - cutoff: radios fraccionarios (0-0.5 aprox), relativo al tamaño mínimo.
    - orden: solo usado para Butterworth.
    """
    rows, cols = img_shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    # distancia al centro
    D = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    Dnorm = D / float(min(crow, ccol))  # normalizar por semitamaño mínimo

    if filtro == 'ideal':
        H = (Dnorm <= cutoff).astype(np.float32)
    elif filtro == 'gaussiano':
        # H = exp(-(D^2)/(2*Dc^2)) con Dc = cutoff
        H = np.exp(-(Dnorm**2) / (2 * (cutoff**2)))
    elif filtro == 'butterworth':
        # H = 1 / (1 + (D/Dc)^(2*orden))
        H = 1 / (1 + (Dnorm / (cutoff + 1e-8))**(2*orden))
    else:
        raise ValueError('Filtro desconocido')

    if tipo == 'lowpass':
        mask = H
    elif tipo == 'highpass':
        mask = 1 - H
    else:
        raise ValueError('Tipo de filtro debe ser lowpass o highpass')

    return mask.astype(np.float32)


def aplicar_filtro_fft(img, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2):
    """Aplica el filtro elegido en el dominio de la frecuencia y reconstruye la imagen."""
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    mask = crear_mascara(img.shape, filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden)
    Gshift = Fshift * mask
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G)
    g = np.real(g)
    g = np.clip(g, 0, 1)
    return g, mask

# ----------------------------- DCT por bloques 8x8 -----------------------------

def dct_matrix(N=8):
    """Genera la matriz de transformada DCT tipo II (ortogonal) de tamaño N."""
    C = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
        for n in range(N):
            C[k, n] = alpha * math.cos(((2*n + 1) * k * math.pi) / (2*N))
    return C

C8 = dct_matrix(8)


def dct_bloque_2d(b):
    """DCT 2D (tipo II) por multiplicación matricial: D = C * b * C^T."""
    return C8 @ b @ C8.T


def idct_bloque_2d(D):
    """IDCT 2D (tipo III equivalente al inverso ortogonal): b = C^T * D * C."""
    return C8.T @ D @ C8

# Tabla de cuantización luminancia estándar JPEG (aproximada)
Q_JPEG = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float64)


def pad_a_multiplo(img, N=8):
    """Rellena (padding) la imagen para que ambos ejes sean múltiplos de N."""
    h, w = img.shape
    nh = ((h + N - 1)//N)*N
    nw = ((w + N - 1)//N)*N
    padded = np.zeros((nh, nw), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w


def dct_compresion(img, q_factor=0.5):
    """Aplica compresión DCT por bloques 8x8 con cuantización.
    q_factor en [0.1, 2.0] aprox: menor -> más compresión (más pérdida).
    Devuelve: reconstruida, psnr, img_padded, reconstruida_padded
    """
    img_p = img.copy()
    padded, h, w = pad_a_multiplo(img_p, 8)
    H, W = padded.shape

    Q = Q_JPEG * q_factor

    # contenedores
    coef_cuant = np.zeros_like(padded, dtype=np.float64)
    recon = np.zeros_like(padded, dtype=np.float64)

    # procesar bloques
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            b = padded[i:i+8, j:j+8]
            # centrar señal a rango [-0.5, 0.5]
            b_shift = b - 0.5
            D = dct_bloque_2d(b_shift)
            # cuantización
            Dq = np.round(D / Q)
            # almacenamiento (opcional)
            coef_cuant[i:i+8, j:j+8] = Dq
            # de-cuantización y reconstrucción
            Dr = Dq * Q
            br = idct_bloque_2d(Dr) + 0.5
            recon[i:i+8, j:j+8] = br

    # recortar a tamaño original y saturar a [0,1]
    recon = np.clip(recon[:h, :w], 0, 1)

    psnr = calcular_psnr(img[:h, :w], recon)
    return recon, psnr, padded, recon


def calcular_psnr(img_ref, img_rec):
    mse = np.mean((img_ref - img_rec)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)

# ----------------------------- Visualización -----------------------------

def mostrar_fft(img, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2, guardar=None):
    F, Fshift, magnitud, fase = fft2_imagen(img)
    filtrada, mask = aplicar_filtro_fft(img, filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden)

    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen original (escala de grises)')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(magnitud, cmap='gray')
    plt.title('Espectro de magnitud (log)')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(fase, cmap='twilight')
    plt.title('Espectro de fase')
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Máscara {filtro} {"pasa bajas" if tipo=="lowpass" else "pasa altas"}\ncutoff={cutoff}, orden={orden}')
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.imshow(filtrada, cmap='gray')
    plt.title('Imagen filtrada (IFFT)')
    plt.axis('off')

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120)
    plt.show()


def mostrar_dct(img, q_factor=0.5, guardar=None):
    rec, psnr, _, _ = dct_compresion(img, q_factor=q_factor)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rec, cmap='gray')
    plt.title(f'Reconstruida DCT (q={q_factor})\nPSNR={psnr:.2f} dB')
    plt.axis('off')

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120)
    plt.show()

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Práctica de Transformaciones en Frecuencia (FFT y DCT)')
    parser.add_argument('--imagen', type=str, default=None, help='Ruta a imagen .jpg/.png (opcional)')
    parser.add_argument('--filtro', type=str, default='butterworth', choices=['ideal','gaussiano','butterworth'], help='Tipo de filtro en frecuencia')
    parser.add_argument('--tipo', type=str, default='lowpass', choices=['lowpass','highpass'], help='Tipo de filtro: pasa bajas o pasa altas')
    parser.add_argument('--cutoff', type=float, default=0.15, help='Radio de corte normalizado (0-0.5 aprox)')
    parser.add_argument('--orden', type=int, default=2, help='Orden (solo Butterworth)')
    parser.add_argument('--dct_q', type=float, default=0.5, help='Factor de cuantización DCT (≈0.3-1.0)')
    parser.add_argument('--salidas', type=str, default='salidas', help='Carpeta donde guardar figuras')

    args = parser.parse_args()

    os.makedirs(args.salidas, exist_ok=True)

    img = cargar_imagen(args.imagen)

    # Mostrar FFT y filtrado
    mostrar_fft(img, filtro=args.filtro, tipo=args.tipo, cutoff=args.cutoff, orden=args.orden,
                guardar=os.path.join(args.salidas, 'fft_filtrado.png'))

    # Mostrar DCT y reconstrucción
    mostrar_dct(img, q_factor=args.dct_q, guardar=os.path.join(args.salidas, 'dct_reconstruccion.png'))

    print('Listo. Figuras guardadas en:', args.salidas)

if __name__ == '__main__':
    main()
