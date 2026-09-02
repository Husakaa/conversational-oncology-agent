"""Genera un código QR de acceso rápido a la app (útil en demos/presentaciones).

Uso:
    python -m src.ui.gen_qr <url> [--out output/acceso_tfg.png]
"""
import argparse
import os

import qrcode


def generar_qr(url: str, ruta_salida: str = "output/acceso_tfg.png") -> str:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(ruta_salida)
    return ruta_salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL a codificar (p. ej. la de un túnel público a la app)")
    parser.add_argument("--out", default="output/acceso_tfg.png", help="Ruta de la imagen PNG de salida")
    args = parser.parse_args()

    ruta = generar_qr(args.url, args.out)
    print(f"Código QR guardado como '{ruta}'")
