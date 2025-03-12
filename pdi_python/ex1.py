import cv2

# Carregar a imagem de Lena em tons de cinza
img = cv2.imread('gray_image.jpg', cv2.IMREAD_GRAYSCALE)

# Testar diferentes valores de alpha e beta
for alpha in [0.5, 1.0, 1.5, 2.0]:
    for beta in [0, 25, 50, 75]:
        bright_contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imshow(f"Alpha: {alpha}, Beta: {beta}", bright_contrast_img)
        cv2.waitKey(2000)
        cv2.destroyWindow(f"Alpha: {alpha}, Beta: {beta}")