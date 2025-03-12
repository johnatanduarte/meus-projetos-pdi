#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // Carrega a imagem ruidosa em escala de cinza
        cv::Mat imagemRuidosa = cv::imread("C:\\Users\\johna\\Desktop\\Eng Comp 10\\PROCESSAMENTO DIGITAL DE IMAGENS\\projeto_pdi\\projeto_pdi\\imagem_equalizada.jpg", cv::IMREAD_GRAYSCALE);

        // Verifica se a imagem foi carregada corretamente
        if (imagemRuidosa.empty()) {
            std::cout << "Erro ao carregar a imagem ruidosa!" << std::endl;
            return -1;
        }

        // Exibe a imagem ruidosa original
        cv::namedWindow("Imagem Ruidosa Original", cv::WINDOW_AUTOSIZE);
        cv::imshow("Imagem Ruidosa Original", imagemRuidosa);

        // Filtro de suavização com média
        cv::Mat imagemSuavizadaMedia;
        cv::blur(imagemRuidosa, imagemSuavizadaMedia, cv::Size(5, 5));
        cv::namedWindow("Suavização com Filtro de Média", cv::WINDOW_AUTOSIZE);
        cv::imshow("Suavização com Filtro de Média", imagemSuavizadaMedia);

        // Filtro Gaussiano
        cv::Mat imagemSuavizadaGauss;
        cv::GaussianBlur(imagemRuidosa, imagemSuavizadaGauss, cv::Size(5, 5), 1.5);
        cv::namedWindow("Suavização com Filtro Gaussiano", cv::WINDOW_AUTOSIZE);
        cv::imshow("Suavização com Filtro Gaussiano", imagemSuavizadaGauss);

        // Filtro de Mediana
        cv::Mat imagemSuavizadaMediana;
        cv::medianBlur(imagemRuidosa, imagemSuavizadaMediana, 5);
        cv::namedWindow("Suavização com Filtro de Mediana", cv::WINDOW_AUTOSIZE);
        cv::imshow("Suavização com Filtro de Mediana", imagemSuavizadaMediana);

        // Filtro de realce (sharpening)
        cv::Mat imagemRealcada;
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0);
        cv::filter2D(imagemRuidosa, imagemRealcada, imagemRuidosa.depth(), kernel);
        cv::namedWindow("Imagem com Filtro de Realce", cv::WINDOW_AUTOSIZE);
        cv::imshow("Imagem com Filtro de Realce", imagemRealcada);

        // Salva as imagens resultantes
        cv::imwrite("imagem_suavizada_media.jpg", imagemSuavizadaMedia);
        cv::imwrite("imagem_suavizada_gauss.jpg", imagemSuavizadaGauss);
        cv::imwrite("imagem_suavizada_mediana.jpg", imagemSuavizadaMediana);
        cv::imwrite("imagem_realcada.jpg", imagemRealcada);

        // Aguarda o usuário pressionar uma tecla para fechar as janelas
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
