import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função que realiza a Transformada Discreta de Fourier na imagem.
def fourier_transform(image):
    """Realiza a Transformada Discreta de Fourier em uma imagem."""
    f = np.fft.fft2(image)  # Aplica a Transformada de Fourier bidimensional.
    fshift = np.fft.fftshift(f)  # Desloca os componentes de baixa frequência para o centro.
    return fshift

# Função que realiza a Transformada Inversa de Fourier.
def inverse_fourier_transform(fshift):
    """Realiza a Transformada Inversa de Fourier."""
    f_ishift = np.fft.ifftshift(fshift)  # Reverte o deslocamento aplicado na transformada.
    img_back = np.fft.ifft2(f_ishift)  # Aplica a Transformada Inversa de Fourier.
    return np.abs(img_back)  # Retorna o valor absoluto para reconstruir a imagem.

# Função para aplicar um filtro passa-baixa.
def low_pass_filter(image, radius=30): #radiu define o tamanho do filtro circular
    """Aplica filtro passa-baixa gaussiano."""
    rows, cols = image.shape  # Dimensões da imagem.
    crow, ccol = rows // 2, cols // 2  # Coordenadas do centro da imagem.

    # Cria uma máscara circular para o filtro passa-baixa.
    mask = np.zeros((rows, cols), np.float32)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2  # Define a área do filtro.
    mask[mask_area] = 1  # Define o valor da máscara para as áreas permitidas.

    # Aplica a Transformada de Fourier e o filtro.
    f = fourier_transform(image)
    f_filtered = f * mask  # Aplica a máscara no domínio da frequência.

    # Retorna a imagem após a Transformada Inversa de Fourier.
    return inverse_fourier_transform(f_filtered)

# Função para aplicar um filtro passa-alta.
def high_pass_filter(image, radius=30):
    """Aplica filtro passa-alta gaussiano."""
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cria uma máscara circular inversa para o filtro passa-alta.
    mask = np.ones((rows, cols), np.float32)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
    mask[mask_area] = 0  # Bloqueia as áreas de baixa frequência.

    # Aplica a Transformada de Fourier e o filtro.
    f = fourier_transform(image)
    f_filtered = f * mask

    # Retorna a imagem após a Transformada Inversa de Fourier.
    return inverse_fourier_transform(f_filtered)

# Função para aplicar um filtro passa-banda.
def band_pass_filter(image, low_radius=20, high_radius=50):
    """Aplica filtro passa-banda."""
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cria uma máscara para permitir uma banda específica de frequências.
    mask = np.zeros((rows, cols), np.float32)
    x, y = np.ogrid[:rows, :cols]
    low_area = (x - crow)**2 + (y - ccol)**2 <= low_radius**2
    high_area = (x - crow)**2 + (y - ccol)**2 <= high_radius**2
    mask[high_area & ~low_area] = 1  # Define a banda permitida.

    # Aplica a Transformada de Fourier e o filtro.
    f = fourier_transform(image)
    f_filtered = f * mask

    # Retorna a imagem após a Transformada Inversa de Fourier.
    return inverse_fourier_transform(f_filtered)

# Função para aplicar um filtro elimina-faixa.
def notch_filter(image, notch_center=(0, 0), width=10):
    """Aplica filtro elimina-faixa."""
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cria uma máscara que elimina frequências específicas.
    mask = np.ones((rows, cols), np.float32)
    x, y = np.ogrid[:rows, :cols]
    notch_x, notch_y = notch_center
    notch_area = ((x - crow - notch_x)**2 + (y - ccol - notch_y)**2 <= width**2)
    mask[notch_area] = 0  # Remove frequências na área do notch.

    # Aplica a Transformada de Fourier e o filtro.
    f = fourier_transform(image)
    f_filtered = f * mask

    # Retorna a imagem após a Transformada Inversa de Fourier.
    return inverse_fourier_transform(f_filtered)

# Função para adicionar ruído à imagem.
def add_noise(image, noise_type='salt_and_pepper', amount=0.05):
    """Adiciona ruído à imagem."""
    if noise_type == 'salt_and_pepper':
        noisy = np.copy(image)
        salt_mask = np.random.rand(image.shape[0], image.shape[1]) < amount / 2
        pepper_mask = np.random.rand(image.shape[0], image.shape[1]) < amount / 2
        noisy[salt_mask] = 255  # Adiciona pontos brancos (sal).
        noisy[pepper_mask] = 0  # Adiciona pontos pretos (pimenta).
        return noisy

    elif noise_type == 'gaussian':
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = image + gauss
        return np.clip(noisy, 0, 255)  # Garante que os valores permaneçam entre 0 e 255.

# Função principal para processar a imagem e aplicar filtros.
def main():
    # Carregar imagem (substitua pelo caminho da sua imagem)
    image = cv2.imread('cr7.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Erro: Imagem não encontrada. Verifique o caminho.")
        return
    
    # 1. Filtro Passa-Baixa (Suavização)
    noisy_image = add_noise(image, 'salt_and_pepper') # Adiciona ruído à imagem.
    low_pass_result = low_pass_filter(noisy_image) # Aplica filtro passa-baixa.
    
    # 2. Filtro Passa-Alta (Realce de Bordas)
    high_pass_result = high_pass_filter(image)
    
    # 3. Filtro Passa-Banda
    band_pass_result = band_pass_filter(image)
    
    # 4. Filtro Elimina-Faixa
    notch_result = notch_filter(image)
    
    # Plota os resultados usando Matplotlib.
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Imagem Original')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title('Imagem com Ruído')
    plt.imshow(noisy_image, cmap='gray')
    
    plt.subplot(2, 3, 3)
    plt.title('Filtro Passa-Baixa')
    plt.imshow(low_pass_result, cmap='gray')
    
    plt.subplot(2, 3, 4)
    plt.title('Filtro Passa-Alta')
    plt.imshow(high_pass_result, cmap='gray')
    
    plt.subplot(2, 3, 5)
    plt.title('Filtro Passa-Banda')
    plt.imshow(band_pass_result, cmap='gray')
    
    plt.subplot(2, 3, 6)
    plt.title('Filtro Elimina-Faixa')
    plt.imshow(notch_result, cmap='gray')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()