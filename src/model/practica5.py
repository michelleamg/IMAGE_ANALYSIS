"""Práctica 5 — Transformaciones en el dominio de la frecuencia.

Parte A: FFT 2D, espectro de magnitud/fase, filtros en frecuencia
  fft2_imagen(img)                         → F, Fshift, magnitud, fase
  crear_mascara(shape, filtro, tipo, cutoff, orden) → mask
  aplicar_filtro_fft(img, filtro, tipo, cutoff, orden) → img_filtrada, mask

  Filtros disponibles: 'ideal', 'gaussiano', 'butterworth'
  Tipos:              'lowpass', 'highpass'

Parte B: DCT por bloques 8×8 con cuantización tipo JPEG
  dct_compresion(img, q_factor)  → rec, psnr
  calcular_psnr(ref, rec)        → float dB

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Mayo 2026
"""
from __future__ import annotations
import math
import numpy as np
import cv2


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convierte imagen RGB uint8 → float32 [0, 1] en escala de grises."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
    return gray


def _to_display_rgb(arr: np.ndarray) -> np.ndarray:
    """Convierte array float32 [0,1] → uint8 RGB para visualización."""
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr_u8.ndim == 2:
        return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
    return arr_u8


def _spectrum_to_display(mag: np.ndarray) -> np.ndarray:
    """Normaliza espectro log para visualización como imagen RGB uint8."""
    mn, mx = mag.min(), mag.max()
    if mx > mn:
        norm = ((mag - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(mag, dtype=np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)


def _phase_to_display(phase: np.ndarray) -> np.ndarray:
    """Mapea fase [-π, π] a imagen RGB usando colormap twilight."""
    norm = ((phase + math.pi) / (2 * math.pi) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TWILIGHT_SHIFTED)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _mask_to_display(mask: np.ndarray) -> np.ndarray:
    """Máscara float32 [0,1] → RGB uint8."""
    u8 = (mask * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# PARTE A — FFT Y FILTROS
# ══════════════════════════════════════════════════════════════════════════════

def fft2_imagen(img: np.ndarray):
    """Calcula FFT 2D sobre imagen en escala de grises.

    Devuelve:
        F        : array complejo (sin desplazar) — para filtrado
        Fshift   : array complejo centrado — para visualización
        magnitud : espectro log(1+|Fshift|) float32
        fase     : ángulo de Fshift en [-π, π] float32
    """
    gray = _to_gray_float(img)
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    magnitud = np.log(1 + np.abs(Fshift)).astype(np.float32)
    fase = np.angle(Fshift).astype(np.float32)
    return F, Fshift, magnitud, fase


def crear_mascara(img_shape: tuple,
                  filtro: str = 'butterworth',
                  tipo: str = 'lowpass',
                  cutoff: float = 0.15,
                  orden: int = 2) -> np.ndarray:
    """Construye máscara de filtro centrada en el espectro.

    Args:
        img_shape : (rows, cols) de la imagen
        filtro    : 'ideal' | 'gaussiano' | 'butterworth'
        tipo      : 'lowpass' | 'highpass'
        cutoff    : radio de corte normalizado (≈0.05–0.45)
        orden     : solo Butterworth — controla pendiente de la transición

    Returns:
        mask float32 [0, 1] del mismo tamaño que img_shape
    """
    rows, cols = img_shape[:2]
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    D = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    Dnorm = D / float(min(crow, ccol))

    if filtro == 'ideal':
        H = (Dnorm <= cutoff).astype(np.float32)
    elif filtro == 'gaussiano':
        H = np.exp(-(Dnorm ** 2) / (2 * (cutoff ** 2))).astype(np.float32)
    elif filtro == 'butterworth':
        H = (1 / (1 + (Dnorm / (cutoff + 1e-8)) ** (2 * orden))).astype(np.float32)
    else:
        raise ValueError(f"Filtro desconocido: {filtro!r}")

    mask = H if tipo == 'lowpass' else (1.0 - H).astype(np.float32)
    return mask


def aplicar_filtro_fft(img: np.ndarray,
                       filtro: str = 'butterworth',
                       tipo: str = 'lowpass',
                       cutoff: float = 0.15,
                       orden: int = 2):
    """Aplica filtro en dominio de la frecuencia y reconstruye por IFFT.

    Returns:
        img_filtrada : np.ndarray RGB uint8
        mask         : np.ndarray float32 [0,1]
        magnitud_orig: espectro de magnitud de la imagen original (display)
        fase_orig    : espectro de fase de la imagen original (display)
    """
    gray = _to_gray_float(img)
    _, _, magnitud, fase = fft2_imagen(gray)

    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mask = crear_mascara(gray.shape, filtro=filtro, tipo=tipo,
                         cutoff=cutoff, orden=orden)
    Gshift = Fshift * mask
    G = np.fft.ifftshift(Gshift)
    g = np.real(np.fft.ifft2(G))
    g = np.clip(g, 0.0, 1.0).astype(np.float32)

    return _to_display_rgb(g), mask, magnitud, fase


def espectro_todas_combinaciones(img: np.ndarray,
                                 cutoff: float = 0.15,
                                 orden: int = 2) -> dict[str, np.ndarray]:
    """Aplica los 6 filtros (3 tipos × 2 polaridades) y devuelve dict de resultados.

    Útil para mostrar una comparativa completa en ResultsWindow.
    """
    resultados = {}
    for filtro in ('ideal', 'gaussiano', 'butterworth'):
        for tipo in ('lowpass', 'highpass'):
            img_f, _, _, _ = aplicar_filtro_fft(
                img, filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden
            )
            key = f"{filtro.capitalize()} {'PB' if tipo == 'lowpass' else 'PA'}"
            resultados[key] = img_f
    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# PARTE B — DCT POR BLOQUES 8×8
# ══════════════════════════════════════════════════════════════════════════════

# ── Matriz DCT ortogonal de tamaño 8 (calculada una sola vez) ────────────────

def _dct_matrix(N: int = 8) -> np.ndarray:
    C = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        alpha = math.sqrt(1 / N) if k == 0 else math.sqrt(2 / N)
        for n in range(N):
            C[k, n] = alpha * math.cos(((2 * n + 1) * k * math.pi) / (2 * N))
    return C


_C8 = _dct_matrix(8)

# ── Tabla de cuantización luminancia estándar JPEG (aproximada) ──────────────
_Q_JPEG = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float64)


def _pad_multiplo(img: np.ndarray, N: int = 8):
    h, w = img.shape
    nh = ((h + N - 1) // N) * N
    nw = ((w + N - 1) // N) * N
    padded = np.zeros((nh, nw), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w


def calcular_psnr(img_ref: np.ndarray, img_rec: np.ndarray) -> float:
    """PSNR entre dos imágenes float32 [0,1]. Devuelve inf si son idénticas."""
    mse = np.mean((img_ref.astype(np.float64) - img_rec.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return -10.0 * math.log10(mse)   # PIXEL_MAX=1 → 20·log10(1)=0


def dct_compresion(img: np.ndarray, q_factor: float = 0.5):
    """Compresión DCT por bloques 8×8 con cuantización escalada.

    Args:
        img      : imagen RGB uint8 o float32 [0,1]
        q_factor : factor de escala de la tabla Q_JPEG.
                   Menor → menos pérdida, mayor PSNR.
                   Mayor → más pérdida (artefactos de bloque), menor PSNR.
                   Rango recomendado: 0.3 – 1.5

    Returns:
        rec_rgb  : imagen reconstruida RGB uint8
        psnr     : PSNR en dB (float)
    """
    gray = _to_gray_float(img)          # float32 [0,1]
    padded, h, w = _pad_multiplo(gray, 8)
    H, W = padded.shape
    Q = _Q_JPEG * q_factor
    recon = np.zeros_like(padded, dtype=np.float64)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            b = padded[i:i+8, j:j+8].astype(np.float64)
            b_shift = b - 0.5
            D  = _C8 @ b_shift @ _C8.T
            Dq = np.round(D / Q)
            Dr = Dq * Q
            br = _C8.T @ Dr @ _C8 + 0.5
            recon[i:i+8, j:j+8] = br

    recon_crop = np.clip(recon[:h, :w], 0.0, 1.0).astype(np.float32)
    psnr = calcular_psnr(gray, recon_crop)
    return _to_display_rgb(recon_crop), round(psnr, 2)


def dct_comparativa(img: np.ndarray,
                    q_values: tuple = (0.3, 0.5, 1.0, 1.5)) -> dict:
    """Aplica DCT con varios q_factor y devuelve dict {q: (img_rgb, psnr)}."""
    resultados = {}
    for q in q_values:
        rec, psnr = dct_compresion(img, q_factor=q)
        resultados[q] = (rec, psnr)
    return resultados
