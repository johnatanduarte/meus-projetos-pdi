import cv2

# Carregar a imagem de Lena em tons de cinza
img = cv2.imread('gray_image.jpg', cv2.IMREAD_GRAYSCALE)

# Testar diferentes tamanhos de kernel na suavização
for kernel_size in [(3, 3), (5, 5), (7, 7), (11, 11)]:
    smooth_img = cv2.GaussianBlur(img, kernel_size, 0)
    cv2.imshow(f"Kernel Size: {kernel_size}", smooth_img)
    cv2.waitKey(2000)
    cv2.destroyWindow(f"Kernel Size: {kernel_size}")