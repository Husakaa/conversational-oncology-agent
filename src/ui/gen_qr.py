import qrcode

# Dirección IP de mi portátil
url = "http://192.168.1.67:8501"

# Configuración del QR
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

qr.add_data(url)
qr.make(fit=True)

# Crear la imagen
img = qr.make_image(fill_color="black", back_color="white")
img.save("output/acceso_tfg.png")

print("Código QR guardado como 'output/acceso_tfg.png'")