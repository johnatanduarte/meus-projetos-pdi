import cv2
import numpy as np

# Carregar a imagem de Lena em tons de cinza
img = cv2.imread('lion.jpg', cv2.IMREAD_GRAYSCALE)

# Criar uma máscara circular
mask = np.zeros(img.shape, dtype="uint8")
cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

# Aplicar diferentes operações lógicas
and_img = cv2.bitwise_and(img, mask)
or_img = cv2.bitwise_or(img, mask)
xor_img = cv2.bitwise_xor(img, mask)
not_img = cv2.bitwise_not(img)

# Exibir os resultados
cv2.imshow("Original", img)
cv2.imshow("AND", and_img)
cv2.imshow("OR", or_img)
cv2.imshow("XOR", xor_img)
cv2.imshow("NOT", not_img)
cv2.waitKey(0)
cv2.destroyAllWindows()