import cv2  # Biblioteca OpenCV para manipulação de imagens
import dlib  # Biblioteca para detecção de rostos e landmarks (pontos de referência)
import numpy as np  # Biblioteca para manipulação de arrays numéricos

# Carregar os modelos de detecção facial do dlib
detector = dlib.get_frontal_face_detector()  # Carrega um modelo pré-treinado para detectar rostos frontais em uma imagem
predictor = dlib.shape_predictor("database/shape_predictor_68_face_landmarks.dat")  # Carrega um modelo pré-treinado para detectar 68 pontos de referência no rosto

def get_face_landmarks(image):
    """
    Obtém os pontos de referência (landmarks) do rosto na imagem.
    Esses pontos incluem características como olhos, nariz, boca e contorno do rosto.
    """
    # Converte a imagem de cores (BGR) para escala de cinza, pois a detecção de rostos funciona melhor em imagens monocromáticas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detecta os rostos na imagem em escala de cinza. O detector retorna uma lista de retângulos que delimitam os rostos encontrados
    faces = detector(gray)
    
    # Se nenhum rosto for detectado, retorna None (nada)
    if len(faces) == 0:
        return None
    
    # Usa o modelo de landmarks para detectar os 68 pontos de referência no primeiro rosto detectado
    landmarks = predictor(gray, faces[0])
    
    # Converte os pontos de referência em um array numpy, que é uma estrutura de dados eficiente para cálculos numéricos
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def warp_face(source_image, target_image, landmarks_source, landmarks_target):
    """
    Ajusta a forma do rosto de origem (source_image) para coincidir com o rosto alvo (target_image).
    Isso é feito usando uma transformação de perspectiva baseada nos pontos de referência (landmarks) dos dois rostos.
    """
    # Calcula o contorno convexo dos pontos de referência do rosto alvo. O contorno convexo é uma forma que envolve todos os pontos, como um elástico esticado ao redor do rosto
    hull = cv2.convexHull(landmarks_target)
    
    # Cria uma máscara para o rosto alvo. A máscara é uma imagem preta do mesmo tamanho da imagem alvo
    mask = np.zeros_like(target_image, dtype=np.uint8)
    
    # Preenche a máscara com branco na área do rosto, usando o contorno convexo calculado anteriormente
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    # Calcula a matriz de transformação que mapeia os pontos do rosto de origem para os pontos do rosto alvo
    # A função cv2.findHomography usa o método RANSAC para calcular a transformação de forma robusta, ignorando possíveis outliers (pontos incorretos)
    transformation_matrix, _ = cv2.findHomography(landmarks_source, landmarks_target, cv2.RANSAC)
    
    # Aplica a transformação de perspectiva ao rosto de origem, ajustando-o para coincidir com o rosto alvo
    warped_face = cv2.warpPerspective(source_image, transformation_matrix, (target_image.shape[1], target_image.shape[0]))
    
    # Retorna apenas a parte do rosto transformado que está dentro da máscara (ou seja, apenas o rosto, sem o fundo)
    return cv2.bitwise_and(warped_face, mask)

def swap_faces(image1, image2):
    """
    Realiza a substituição do rosto entre as duas imagens.
    O rosto da primeira imagem (image1) é substituído pelo rosto da segunda imagem (image2).
    """
    # Obtém os pontos de referência do rosto na primeira imagem
    landmarks1 = get_face_landmarks(image1)
    
    # Obtém os pontos de referência do rosto na segunda imagem
    landmarks2 = get_face_landmarks(image2)
    
    # Se não for possível detectar rostos em uma das imagens, exibe uma mensagem de erro e retorna None (nada)
    if landmarks1 is None or landmarks2 is None:
        print("Rosto não detectado em uma das imagens.")
        return None
    
    # Ajusta o rosto da primeira imagem para coincidir com o rosto da segunda imagem, usando a função warp_face
    swapped_face = warp_face(image1, image2, landmarks1, landmarks2)
    
    # Cria uma cópia da segunda imagem (imagem alvo), onde o rosto será substituído
    result = image2.copy()
    
    # Converte o rosto transformado para escala de cinza, criando uma máscara que define onde o rosto está localizado
    mask = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2GRAY)
    
    # Define a máscara como branco (255) onde há pixels do rosto transformado
    mask[mask > 0] = 255
    
    # Usa a técnica de clonagem suave (seamlessClone) para mesclar o rosto transformado na imagem alvo
    # O ponto central do rosto (landmarks2[30, 0], landmarks2[30, 1]) é usado como referência para a mesclagem
    result = cv2.seamlessClone(swapped_face, image2, mask, (landmarks2[30, 0], landmarks2[30, 1]), cv2.NORMAL_CLONE)
    
    # Retorna a imagem final com o rosto substituído
    return result

# Carregar as imagens
image1 = cv2.imread("johnatan2.jpeg")  # Carrega a primeira imagem (rosto que será transferido)
image2 = cv2.imread("cbum.jpeg")  # Carrega a segunda imagem (rosto que receberá a substituição)

# Substituir o rosto
output = swap_faces(image1, image2)  # Chama a função para trocar os rostos

# Se a troca foi bem-sucedida (ou seja, se a função não retornou None):
if output is not None:
    cv2.imshow("Troca de Rosto", output)  # Exibe a imagem final na tela
    cv2.waitKey(0)  # Aguarda o usuário pressionar uma tecla para continuar
    cv2.destroyAllWindows()  # Fecha a janela da imagem
    cv2.imwrite("resultado.jpg", output)  # Salva a imagem resultante como "resultado.jpg"