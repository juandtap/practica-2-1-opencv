#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

// funcion prototipo para aplicar la correcion gamma
void correccionGamma(Mat& frame, Mat& frameGamma, float gamma);


int main(int argc, char** argv) {
   VideoCapture liveVideo("http://211.132.61.124/mjpg/video.mjpg");
    //VideoCapture liveVideo(0);

    if (liveVideo.isOpened()) {
        namedWindow("LiveVideo", WINDOW_AUTOSIZE);
        namedWindow("movimiento", WINDOW_AUTOSIZE);
        namedWindow("Ecualizacion_Histograma", WINDOW_AUTOSIZE);
        namedWindow("CLAHE", WINDOW_AUTOSIZE);
        namedWindow("Gamma", WINDOW_AUTOSIZE);
        namedWindow("BACKGROUND", WINDOW_AUTOSIZE);
        namedWindow("PRINCIPAL", WINDOW_AUTOSIZE);

        Mat frame;
        Mat frameClahe;
        Mat anterior;
        Mat resta;
        Mat frameHist;
        Mat frameGamma;

        Mat videoFinal;

        // para el metodo daptive background substraction
        Mat backgroud;
        Mat frameDiff;
        Mat movimiento;
        int THRESH = 60;
        float alpha = 0.1;


        bool primerFrame = true;
        int FPS = liveVideo.get(CAP_PROP_FPS);
        string FPSs = "FPS: "+to_string(FPS);

        //filtro clahe
        Ptr<CLAHE> clahe = createCLAHE(4.0);
        clahe->setTilesGridSize(Size(4,4));

        while(true) {
            liveVideo >> frame;
            resize(frame, frame,Size(), 0.5,0.5);
            //convertir a escala de grises
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            // se obtiene el primer frame
            if (primerFrame) {
                primerFrame = false;
                backgroud = frame.clone();
            }

            // colocar texto mostrando los FPSs
            putText(frame,FPSs, Point(0, 20), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,0),2);
            //Filtros
            // Filtro de la ecualizacion del histograma
            equalizeHist(frame, frameHist);
            // Filtro CLAHE
            clahe->apply(frame, frameClahe);
            //filtro correcion gamma
            correccionGamma(frame, frameGamma, 0.55);
            // Deteccion de movimiento
            // substraccion del bg
            absdiff(backgroud, frameClahe, frameDiff);
            //aplicacion del threshold
            threshold(frameDiff,movimiento, THRESH, 255, THRESH_BINARY);
            //actualizar background
            addWeighted(frameClahe,alpha,backgroud,1-alpha,0,backgroud);


            imshow("movimiento", movimiento);
            imshow("LiveVideo", frame);
            imshow("Ecualizacion_Histograma", frameHist);
            imshow("CLAHE", frameClahe);
            imshow("BACKGROUND", backgroud);
            imshow("Gamma", frameGamma);

            hconcat(frameClahe, movimiento, videoFinal);
            imshow("PRINCIPAL", videoFinal);

            // esperar por la tecla Esc
            if (waitKey(23)==27) {
                break;
            }
        }


    }

    liveVideo.release();
    destroyAllWindows();


    return 0;
}

void correccionGamma(Mat& frame, Mat& frameGamma, float gamma) {
    frameGamma = frame.clone();
    // se normaliza la imagen [0 , 1]
    frameGamma.convertTo(frameGamma, CV_32F, 1.0 / 255.0);

    // Aplicar la corrección gamma
    pow(frameGamma, gamma, frameGamma);

    // se desnormaliza [0, 255]
    frameGamma *= 255;
    frameGamma.convertTo(frameGamma, CV_8U);
}

