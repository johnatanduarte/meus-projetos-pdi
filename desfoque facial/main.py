import cv2
import numpy as np

def detectar_e_borrar_rosto(caminho_imagem):
    """
    Função para detectar e borrar rosto em uma imagem
    
    Passos do algoritmo:
    1. Carregar imagem
    2. Carregar classificador de detecção facial
    3. Converter imagem para escala de cinza
    4. Detectar rostos na imagem
    5. Para cada rosto detectado:
       - Aplicar filtro de desfoque (Gaussian Blur)
    6. Salvar imagem processada
    """
    
    # 1. Carregar imagem
    imagem = cv2.imread(caminho_imagem)
    
    # 2. Carregar classificador pré-treinado de detecção facial
    classificador_facial = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # 3. Converter imagem para escala de cinza (necessário para detecção)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # 4. Detectar rostos na imagem
    rostos = classificador_facial.detectMultiScale(
        imagem_cinza, 
        scaleFactor=1.1,  # Reduz o tamanho da imagem em 10% a cada escala
        minNeighbors=5,   # Número mínimo de vizinhos para considerar uma detecção válida
        minSize=(30, 30)  # Tamanho mínimo da região detectada
    )
    
    # 5. Borrar cada rosto detectado
    for (x, y, w, h) in rostos:
        # Extrair região do rosto
        regiao_rosto = imagem[y:y+h, x:x+w]
        
        # Aplicar desfoque Gaussian
        # Kernel (25,25) define a intensidade do borramento
        rosto_borrado = cv2.GaussianBlur(regiao_rosto, (51, 51), sigmaX=10) 
        
        # Substituir região original pelo rosto borrado
        imagem[y:y+h, x:x+w] = rosto_borrado
    
    # 6. Salvar imagem processada
    caminho_saida = caminho_imagem.replace('.', '_borrado.')
    cv2.imwrite(caminho_saida, imagem) #salva a imagem em um arquivo
    
    print(f"Imagem processada salva em: {caminho_saida}")
    
    print(f"Número de rostos detectados: {len(rostos)}")

# Exemplo de uso
caminho_imagem = 'C:\\Users\\johna\\Desktop\\Eng Comp 10\\PROCESSAMENTO DIGITAL DE IMAGENS\\desfoque facial\\eu.jpg'
detectar_e_borrar_rosto(caminho_imagem)