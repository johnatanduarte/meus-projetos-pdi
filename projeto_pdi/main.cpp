#include <opencv2/opencv.hpp>
#include <iostream>

// Função para realizar a equalização de histograma
void equalizarHistograma(const cv::Mat& imagem) {
    // Definindo os parâmetros do histograma
    int histSize = 256;  // Número de níveis de cinza
    float range[] = { 0, 256 };  // Intervalo de intensidade de pixel (0 a 255)
    const float* histRange = { range };

    // Calcula o histograma da imagem original
    cv::Mat hist;
    calcHist(&imagem, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Configura o tamanho da imagem para exibir o histograma
    int hist_w = 512, hist_h = 400;
    cv::Mat histImagem(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));  // Fundo branco

    // Normaliza o histograma para que ele caiba na imagem de histograma
    normalize(hist, hist, 0, hist_h, cv::NORM_MINMAX);

    // Desenha o histograma original
    for (int i = 0; i < histSize; i++) {
        line(histImagem,
            cv::Point(i * 2, hist_h),  // Ponto inicial (abaixo)
            cv::Point(i * 2, hist_h - cvRound(hist.at<float>(i))),  // Ponto final (acima)
            cv::Scalar(0, 0, 0),  // Cor preta
            2);  // Espessura da linha
    }
    // Exibe o histograma original
    cv::imshow("Histograma Original", histImagem);

    // Equaliza o histograma da imagem
    cv::Mat imagemEqualizada;
    cv::equalizeHist(imagem, imagemEqualizada);
    cv::imshow("Imagem Equalizada", imagemEqualizada);

    // Calcula o histograma da imagem equalizada
    cv::Mat histEqualizado;
    calcHist(&imagemEqualizada, 1, 0, cv::Mat(), histEqualizado, 1, &histSize, &histRange);
    cv::Mat histImagemEqualizada(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    normalize(histEqualizado, histEqualizado, 0, hist_h, cv::NORM_MINMAX);

    // Desenha o histograma equalizado
    for (int i = 0; i < histSize; i++) {
        line(histImagemEqualizada,
            cv::Point(i * 2, hist_h),
            cv::Point(i * 2, hist_h - cvRound(histEqualizado.at<float>(i))),
            cv::Scalar(0, 0, 0),
            2);
    }
    // Exibe o histograma equalizado
    cv::imshow("Histograma Equalizado", histImagemEqualizada);
}

// Função para aplicar filtros de suavização e realce na imagem
void aplicarFiltros(const cv::Mat& imagem) {
    // Suavização com filtro de média (média simples dos pixels vizinhos)
    cv::Mat imagemSuavizadaMedia;
    cv::blur(imagem, imagemSuavizadaMedia, cv::Size(5, 5));
    cv::imshow("Suavização com Filtro de Média", imagemSuavizadaMedia);

    // Suavização com filtro Gaussiano (suaviza com peso maior para pixels centrais)
    cv::Mat imagemSuavizadaGauss;
    cv::GaussianBlur(imagem, imagemSuavizadaGauss, cv::Size(5, 5), 1.5);
    cv::imshow("Suavização com Filtro Gaussiano", imagemSuavizadaGauss);

    // Suavização com filtro de Mediana (preserva bordas e remove ruídos tipo sal e pimenta)
    cv::Mat imagemSuavizadaMediana;
    cv::medianBlur(imagem, imagemSuavizadaMediana, 5);
    cv::imshow("Suavização com Filtro de Mediana", imagemSuavizadaMediana);

    // Realce com kernel de nitidez para destacar detalhes e bordas
    cv::Mat imagemRealcada;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::filter2D(imagem, imagemRealcada, imagem.depth(), kernel);
    cv::imshow("Imagem com Filtro de Realce", imagemRealcada);
}

int main() {
    try {
        // Carrega a imagem original em escala de cinza
        cv::Mat imagem = cv::imread("C:\\Users\\johna\\Desktop\\Eng Comp 10\\PROCESSAMENTO DIGITAL DE IMAGENS\\projeto_pdi\\projeto_pdi\\gray_image.jpg", cv::IMREAD_GRAYSCALE);

        if (imagem.empty()) {
            std::cout << "Erro ao carregar a imagem!" << std::endl;
            return -1;
        }

        // Exibe a imagem original
        cv::imshow("Imagem Original", imagem);

        // Executa a equalização de histograma
        equalizarHistograma(imagem);

        // Aplica os filtros de suavização e realce
        aplicarFiltros(imagem);

        // Espera o usuário pressionar uma tecla para fechar as janelas
        cv::waitKey(0);

        return 0;
    }
    catch (const cv::Exception& e) {
        std::cout << "Erro OpenCV: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cout << "Erro: " << e.what() << std::endl;
        return -1;
    }
}
