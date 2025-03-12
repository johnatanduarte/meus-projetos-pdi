import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar a Imagem
image = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Binarização (Thresholding)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 3. Operações Morfológicas
kernel = np.ones((5, 5), np.uint8)

# Erosão
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Dilatação
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Abertura (Erosão seguida de Dilatação)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Fechamento (Dilatação seguida de Erosão)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# 4. Segmentação por Crescimento de Região
def region_growing(img, seed):
    # Cria uma máscara para armazenar a região segmentada
    mask = np.zeros_like(img)
    # Define o valor do pixel semente
    seed_value = img[seed]
    # Lista de pixels a serem verificados
    pixels_to_check = [seed]
    
    while len(pixels_to_check) > 0:
        x, y = pixels_to_check.pop()
        # Verifica se o pixel está dentro da imagem e se ainda não foi processado
        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1] and mask[x, y] == 0:
            # Verifica se o pixel é similar ao pixel semente
            if abs(int(img[x, y]) - int(seed_value)) < 10:
                mask[x, y] = 255
                # Adiciona os vizinhos à lista de pixels a serem verificados
                pixels_to_check.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return mask

# Define um ponto semente para o crescimento de região
seed_point = (100, 100)
segmented_image = region_growing(binary_image, seed_point)

# 5. Visualização
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title('Imagem Original')
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Imagem Binária')
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Erosão')
plt.imshow(eroded_image, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Dilatação')
plt.imshow(dilated_image, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Abertura')
plt.imshow(opened_image, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('Fechamento')
plt.imshow(closed_image, cmap='gray')

plt.show()

# Mostrar a imagem segmentada
plt.figure(figsize=(6, 6))
plt.title('Segmentação por Crescimento de Região')
plt.imshow(segmented_image, cmap='gray')
plt.show()