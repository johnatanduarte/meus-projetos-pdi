import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Carregar a Imagem
image = cv2.imread('mulher.jpg', cv2.IMREAD_COLOR)

# 2. Converter para escala de cinza (para simplificar a compressão)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Compressão usando o método JPEG
# Qualidade da compressão (0 a 100, onde 100 é a melhor qualidade)
compression_quality = 50
cv2.imwrite('compressed_image.jpg', gray_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])

# 4. Carregar a imagem comprimida para visualização
compressed_image = cv2.imread('compressed_image.jpg', cv2.IMREAD_GRAYSCALE)

# 5. Compressão usando o método PNG (sem perdas, mas com compactação)
cv2.imwrite('compressed_image.png', gray_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# 6. Carregar a imagem comprimida PNG para visualização
compressed_image_png = cv2.imread('compressed_image.png', cv2.IMREAD_GRAYSCALE)

# 7. Visualização das imagens
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Imagem Original')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Imagem Comprimida (JPEG)')
plt.imshow(compressed_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Imagem Comprimida (PNG)')
plt.imshow(compressed_image_png, cmap='gray')

plt.show()

# 8. Comparação de tamanhos de arquivo
import os
original_size = os.path.getsize('mulher.jpg') / 1024  # Tamanho em KB
compressed_jpeg_size = os.path.getsize('compressed_image.jpg') / 1024
compressed_png_size = os.path.getsize('compressed_image.png') / 1024

print(f"Tamanho original: {original_size:.2f} KB")
print(f"Tamanho comprimido (JPEG): {compressed_jpeg_size:.2f} KB")
print(f"Tamanho comprimido (PNG): {compressed_png_size:.2f} KB")