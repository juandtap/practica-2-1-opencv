#include <iostream>

#include <sstream>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


using namespace cv;
using namespace std;

// funciones prototipo para agregar ruido de sal y pimienta
void agregarRuidoSal(Mat& frame, double porcentaje);
void agregarRuidoPimienta(Mat& frame, double porcentaje);

void trackBarChange(int, void*) {
}
// trackbar para el tamanio del kernel filtro mediana
int km = 3;
void trackBarKernelMediana(int, void*) {
    if(km%2 == 0) {
        km = km+1;
    }
}

// trackbar para el tamanio del kernel filtro gauss
int kg = 3;
void trackBarKernelGauss(int, void*) {
    if(kg%2 == 0) {
        kg = kg+1;
    }
}

int main(int argc, char** argv){
    VideoCapture liveVideo("http://211.132.61.124/mjpg/video.mjpg");
    if (liveVideo.isOpened()) {
        namedWindow("VIDEO", WINDOW_AUTOSIZE);
        // ventana donde se juntan las otras ventanas
        namedWindow("PRINCIPAL", WINDOW_AUTOSIZE);
        Mat principal;

        namedWindow("PRINCIPAL2",  WINDOW_AUTOSIZE);
        Mat principal2;


        Mat frame;
        Mat frameRuido;
        // Mats filtros de reduccion de ruido
        Mat frameBilateral;
        Mat frameMediana;
        Mat frameGauss;
        namedWindow("Filtro-Bilateral", WINDOW_AUTOSIZE);
        namedWindow("Filtro-Mediana", WINDOW_AUTOSIZE);
        namedWindow("Filtro-Gauss", WINDOW_AUTOSIZE);

        // Mats para deteccion de bordes
        Mat frameGray;
        Mat frameSobel  ,gX, gY, absGX, absGY;
        Mat frameLaplace;
        Mat frameCanny;
        namedWindow("Bordes-Sobel", WINDOW_AUTOSIZE);
        namedWindow("Bordes-Laplace", WINDOW_AUTOSIZE);
        namedWindow("Bordes-Canny", WINDOW_AUTOSIZE);
        // suavizado de bordes
        Mat frameGraySuave;
        Mat frameSobelSuave;
        Mat frameLaplaceSuave;
        Mat frameCannySuave;
        namedWindow("Video-Suavizado-Gris", WINDOW_AUTOSIZE);
        namedWindow("Bordes-Sobel-Suave", WINDOW_AUTOSIZE);
        namedWindow("Bordes-Laplace-Suave", WINDOW_AUTOSIZE);
        namedWindow("Bordes-Canny-Suave", WINDOW_AUTOSIZE);


        int sliderSal = 0;
        int sliderPimienta = 0;
        double probSal;
        double probPimienta;

        createTrackbar("Ruido de Sal     ","VIDEO", &sliderSal, 100, trackBarChange, NULL);
        createTrackbar("Ruido de Pimienta","VIDEO", &sliderPimienta,100,trackBarChange,NULL);

        createTrackbar("Tamanio kernel M", "Filtro-Mediana",&km,35,trackBarKernelMediana, NULL);
        setTrackbarMin("Tamanio kernel M", "Filtro-Mediana",3);
        createTrackbar("Tamanio kernel G", "Filtro-Gauss", &kg, 35, trackBarKernelGauss, NULL);
        setTrackbarMin("Tamanio kernel G", "Filtro-Gauss",3);

        // flags para crear las ventana contenedoras
        bool framePrincipalCreado = false;
        bool framePrincipal2Creado = false;

        while(true) {
            liveVideo >> frame;
            resize(frame, frame, Size(), 0.5, 0.5);
            probSal  = sliderSal;
            probPimienta = sliderPimienta;

            frameRuido = frame.clone();

            agregarRuidoSal(frameRuido, probSal);
            agregarRuidoPimienta(frameRuido,probPimienta);

            // filtros de reduccion de ruido
            medianBlur(frameRuido,frameMediana,km);
            GaussianBlur(frameRuido, frameGauss, Size(kg,kg), 0);
            bilateralFilter(frameRuido, frameBilateral,15, 100, 100);

            putText(frameRuido,"Video Original con Ruido",Point(5,20),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2);
            putText(frameMediana,"Filtro Mediana",Point(5,20),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2);
            putText(frameGauss,"Filtro Gauss",Point(5,20),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2);
            putText(frameBilateral,"Filtro Bilateral",Point(5,20),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2);

            imshow("VIDEO", frameRuido);
            imshow("Filtro-Mediana", frameMediana);
            imshow("Filtro-Gauss", frameGauss);
            imshow("Filtro-Bilateral", frameBilateral);


            if (!framePrincipalCreado) {
                principal = Mat(Size(frameRuido.cols*2, frameRuido.rows*2), frameRuido.type());

                framePrincipalCreado = true;
            }

            // se colocan las cuatro ventanas de filtros
            frameRuido.copyTo(principal(Rect(0,0,frameRuido.cols, frameRuido.rows)));
            frameMediana.copyTo(principal(Rect(frameRuido.cols,0,frameMediana.cols, frameMediana.rows)));
            frameGauss.copyTo(principal(Rect(0,frameRuido.rows,frameGauss.cols, frameGauss.rows)));
            frameBilateral.copyTo(principal(Rect(frameRuido.cols,frameRuido.rows,frameBilateral.cols, frameBilateral.rows)));
            // Se muestra una ventana con los 3 filtros de ruido aplicados
            imshow("PRINCIPAL", principal);


            // Deteccion de bordes
            cvtColor(frame, frameGray, COLOR_BGR2GRAY);
            // resize para juntar todas las imagenes
            resize(frameGray, frameGray, Size(), 0.8, 0.8);
            //suavizado con medianBlur con Kernel = 3
            medianBlur(frameGray, frameGraySuave, 3);

            // Filtro de sobel
            Sobel(frameGray, gX, CV_16S, 1,0,3);
            Sobel(frameGray, gY, CV_16S,0,1,3);

            convertScaleAbs(gX,absGX);
            convertScaleAbs(gY, absGY);
            addWeighted(absGX, 0.5, absGY, 0.5,0,frameSobel);

            // Filtro Sobel con suavizado
            Sobel(frameGraySuave, gX, CV_16S, 1,0,3);
            Sobel(frameGraySuave, gY, CV_16S,0,1,3);

            convertScaleAbs(gX,absGX);
            convertScaleAbs(gY, absGY);
            addWeighted(absGX, 0.5, absGY, 0.5,0,frameSobelSuave);

            // Filtro de laplace
            Laplacian(frameGray, frameLaplace, CV_16S, 3);
            convertScaleAbs(frameLaplace, frameLaplace);

            // Filtro laplace suavizado
            Laplacian(frameGraySuave, frameLaplaceSuave, CV_16S, 3);
            convertScaleAbs(frameLaplaceSuave, frameLaplaceSuave);


            //Filtro Canny
            Canny(frameGray, frameCanny, 50, 150, 3);

            // Filtro Canny con suavizado
            Canny(frameGraySuave, frameCannySuave, 50, 150, 3);


            putText(frameSobel,"Bordes Sobel ",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);
            putText(frameSobelSuave,"Bordes Sobel Suavizado",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);

            putText(frameLaplace,"Bordes Laplace ",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);
            putText(frameLaplaceSuave,"Bordes Laplace Suavizado",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);

            putText(frameCanny,"Bordes Canny ",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);
            putText(frameCannySuave,"Bordes Canny Suavizado",Point(5,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);

            // Mostrar deteccion de bordes
            imshow("Bordes-Sobel", frameSobel);
            imshow("Bordes-Laplace", frameLaplace);
            imshow("Bordes-Canny", frameCanny);
            //Deteccion de bordes Con suavizado
            imshow("Bordes-Sobel-Suave", frameSobelSuave);
            imshow("Bordes-Laplace-Suave", frameLaplaceSuave);
            imshow("Bordes-Canny-Suave", frameCannySuave);

            imshow("Video-Suavizado-Gris", frameGraySuave);

            if (!framePrincipal2Creado) {
                // ventana de 2 columnas X 3 filas
                principal2 = Mat(Size(frameGray.cols*2, frameGray.rows*3), frameGray.type());

                framePrincipal2Creado = true;
            }

            // se colocan las seis ventanas de deteccion de bordes
            //todas las ventanas tienen la misma dimension, se usara frameSobel para todas
            frameSobel.copyTo(principal2(Rect(0,0,frameSobel.cols, frameSobel.rows)));
            frameSobelSuave.copyTo(principal2(Rect(frameSobel.cols,0,frameSobel.cols, frameSobel.rows)));
            frameLaplace.copyTo(principal2(Rect(0,frameSobel.rows,frameSobel.cols, frameSobel.rows)));
            frameLaplaceSuave.copyTo(principal2(Rect(frameSobel.cols,frameSobel.rows,frameSobel.cols, frameSobel.rows)));
            frameCanny.copyTo(principal2(Rect(0,frameSobel.rows*2,frameSobel.cols, frameSobel.rows)));
            frameCannySuave.copyTo(principal2(Rect(frameSobel.cols,frameSobel.rows*2,frameSobel.cols, frameSobel.rows)));




            imshow("PRINCIPAL2", principal2);



            if(waitKey(23)==27) {
                break;
            }
        }
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}

void agregarRuidoPimienta(Mat& frame, double porcentaje) {
    int total_pixeles = frame.rows * frame.cols;
    int pimienta = (int)((porcentaje / 100.0) * total_pixeles * 0.5); // x0.5 la mitad de los pixeles seran sal
    random_device random;
    mt19937 gen(random());
    uniform_int_distribution<> distribucionFilas(0, frame.rows - 1);
    uniform_int_distribution<> distribucionCols(0, frame.cols - 1);
    int x , y;
    // al recorrer el for con el numero de puntos de pimienta/sal se generan numeros random (x,y) para convertir los pixeles
    //  a negro (pimienta) o blanco (sal)
    for (int i = 0; i < pimienta; ++i) {
        x = distribucionFilas(gen);
        y = distribucionCols(gen);
        frame.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
    }
}

void agregarRuidoSal(Mat& frame, double porcentaje) {
    int totalPixeles = frame.rows * frame.cols;
    int sal = (int)((porcentaje / 100.0) * totalPixeles * 0.5);
    random_device random;
    mt19937 gen(random());
    uniform_int_distribution<> distribucionFilas(0, frame.rows - 1);
    uniform_int_distribution<> distribucionCols(0, frame.cols - 1);
    int x , y;
    for (int i = 0; i < sal; ++i) {
        x = distribucionFilas(gen);
        y = distribucionCols(gen);
        frame.at<Vec3b>(x, y) = Vec3b(255, 255, 255);
    }
}

