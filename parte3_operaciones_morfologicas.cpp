#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    //carga de imagenes
    Mat img1 = cv::imread("../imagenes/brats-t1c-sample1_slice_100.png", IMREAD_GRAYSCALE);
    Mat img2 = cv::imread("../imagenes/brats-t1c-sample2_slice_100.png",IMREAD_GRAYSCALE);
    Mat img3 = cv::imread("../imagenes/lung_xray_sample.jpg",IMREAD_GRAYSCALE);



    // se define los tamaños de kernel y el tipo de estructura
    int kernel1 = 5;
    Mat elemento1 = getStructuringElement(MORPH_ELLIPSE, Size(kernel1, kernel1));

    int kernel2 = 25;
    Mat elemento2 = getStructuringElement(MORPH_ELLIPSE, Size(kernel2, kernel2));

    // tamaño sugerido en el artículo "Using morphological transforms to enhance the contrast of medical images" Hassanpour et al. (2015)
    int kernel3 = 37;
    Mat elemento3 = getStructuringElement(MORPH_ELLIPSE, Size(kernel3, kernel3));

    //Operaciones en imagen 1
    //Erosion
    Mat erosion1;
    erode(img1, erosion1, elemento1);

    Mat erosion2;
    erode(img1, erosion2, elemento2);

    Mat erosion3;
    erode(img1, erosion3, elemento3);

    //Dilatacion
    Mat dilatacion1;
    dilate(img1, dilatacion1, elemento1);

    Mat dilatacion2;
    dilate(img1, dilatacion2, elemento2);

    Mat dilatacion3;
    dilate(img1, dilatacion3, elemento3);

    //Top hat
    Mat topHat1;
    morphologyEx(img1, topHat1, MORPH_TOPHAT, elemento1);
    Mat topHat2;
    morphologyEx(img1, topHat2, MORPH_TOPHAT, elemento2);
    Mat topHat3;
    morphologyEx(img1, topHat3, MORPH_TOPHAT, elemento3);

    // Black hat
    Mat blackHat1;
    morphologyEx(img1, blackHat1, MORPH_BLACKHAT, elemento1);

    Mat blackHat2;
    morphologyEx(img1, blackHat2, MORPH_BLACKHAT, elemento2);

    Mat blackHat3;
    morphologyEx(img1, blackHat3, MORPH_BLACKHAT, elemento3);

    // combinada original + (tophat - blackhat)
    Mat combinada1;
    Mat resta1 = topHat1 - blackHat1;
    combinada1 = img1 + resta1;

    Mat combinada2;
    Mat resta2 = topHat2 - blackHat2;
    combinada2 = img1 + resta2;

    Mat combinada3;
    Mat resta3 = topHat3 - blackHat3;
    combinada3 = img1 + resta3;

    //namedWindow("IMG1", WINDOW_AUTOSIZE);
    //namedWindow("IMG2", WINDOW_AUTOSIZE);
    //namedWindow("IMG3", WINDOW_AUTOSIZE);

    imshow("IMG1", img1);
    imshow("IMG2", img2);
    imshow("IMG3", img3);

    //kernel = 5
    // imshow("Erosión", erosion1);
    // imshow("Dilatación", dilatacion1);
    // imshow("Top Hat", topHat1);
    // imshow("Black Hat", blackHat1);
    // imshow("Suma (TopHat - BalckHat)", combinada1);

    //kernel = 25
    // imshow("Erosión", erosion2);
    // imshow("Dilatación", dilatacion2);
    // imshow("Top Hat", topHat2);
    // imshow("Black Hat", blackHat2);
    // imshow("Suma (TopHat - BalckHat)", combinada2);

    //kernel = 37
    imshow("Erosión", erosion3);
    imshow("Dilatación", dilatacion3);
    imshow("Top Hat", topHat3);
    imshow("Black Hat", blackHat3);
    imshow("Suma (TopHat - BalckHat)", combinada3);


    //Operaciones en imagen 2
    //Erosion
    Mat erosion1_2;
    erode(img2, erosion1_2, elemento1);

    Mat erosion2_2;
    erode(img2, erosion2_2, elemento2);

    Mat erosion3_2;
    erode(img2, erosion3_2, elemento3);

    //Dilatacion
    Mat dilatacion1_2;
    dilate(img2, dilatacion1_2, elemento1);

    Mat dilatacion2_2;
    dilate(img2, dilatacion2_2, elemento2);

    Mat dilatacion3_2;
    dilate(img2, dilatacion3_2, elemento3);



    //Top hat
    Mat topHat1_2;
    morphologyEx(img2, topHat1_2, MORPH_TOPHAT, elemento1);
    Mat topHat2_2;
    morphologyEx(img2, topHat2_2, MORPH_TOPHAT, elemento2);
    Mat topHat3_2;
    morphologyEx(img2, topHat3_2, MORPH_TOPHAT, elemento3);

    // Black hat
    Mat blackHat1_2;
    morphologyEx(img2, blackHat1_2, MORPH_BLACKHAT, elemento1);

    Mat blackHat2_2;
    morphologyEx(img2, blackHat2_2, MORPH_BLACKHAT, elemento2);

    Mat blackHat3_2;
    morphologyEx(img2, blackHat3_2, MORPH_BLACKHAT, elemento3);

    // combinada original + (tophat - blackhat)
    Mat combinada1_2;
    Mat resta1_2 = topHat1_2 - blackHat1_2;
    combinada1_2 = img2 + resta1_2;

    Mat combinada2_2;
    Mat resta2_2 = topHat2_2 - blackHat2_2;
    combinada2_2 = img2 + resta2_2;

    Mat combinada3_2;
    Mat resta3_2 = topHat3_2 - blackHat3_2;
    combinada3_2 = img2 + resta3_2;

    //kernel = 5
    // imshow("Erosión img2", erosion1_2);
    // imshow("Dilatación img2", dilatacion1_2);
    // imshow("Top Hat img2", topHat1_2);
    // imshow("Black Hat img2", blackHat1_2);
    // imshow("Suma (TopHat - BalckHat) img2", combinada1_2);

    //kernel = 25
    // imshow("Erosión img2", erosion2_2);
    // imshow("Dilatación img2", dilatacion2_2);
    // imshow("Top Hat img2", topHat2_2);
    // imshow("Black Hat img2", blackHat2_2);
    // imshow("Suma (TopHat - BalckHat) img2", combinada2_2);

    //kernel = 37
    imshow("Erosión img2", erosion3_2);
    imshow("Dilatación img2", dilatacion3_2);
    imshow("Top Hat img2", topHat3_2);
    imshow("Black Hat img2", blackHat3_2);
    imshow("Suma (TopHat - BalckHat) img2", combinada3_2);

    //Operaciones en imagen 3
    //Erosion
    Mat erosion1_3;
    erode(img3, erosion1_3, elemento1);

    Mat erosion2_3;
    erode(img3, erosion2_3, elemento2);

    Mat erosion3_3;
    erode(img3, erosion3_3, elemento3);

    //Dilatacion
    Mat dilatacion1_3;
    dilate(img3, dilatacion1_3, elemento1);

    Mat dilatacion2_3;
    dilate(img3, dilatacion2_3, elemento2);

    Mat dilatacion3_3;
    dilate(img3, dilatacion3_3, elemento3);



    //Top hat
    Mat topHat1_3;
    morphologyEx(img3, topHat1_3, MORPH_TOPHAT, elemento1);
    Mat topHat2_3;
    morphologyEx(img3, topHat2_3, MORPH_TOPHAT, elemento2);
    Mat topHat3_3;
    morphologyEx(img3, topHat3_3, MORPH_TOPHAT, elemento3);

    // Black hat
    Mat blackHat1_3;
    morphologyEx(img3, blackHat1_3, MORPH_BLACKHAT, elemento1);

    Mat blackHat2_3;
    morphologyEx(img3, blackHat2_3, MORPH_BLACKHAT, elemento2);

    Mat blackHat3_3;
    morphologyEx(img3, blackHat3_3, MORPH_BLACKHAT, elemento3);

    // combinada original + (tophat - blackhat)
    Mat combinada1_3;
    Mat resta1_3 = topHat1_3 - blackHat1_3;
    combinada1_3 = img3 + resta1_3;

    Mat combinada2_3;
    Mat resta2_3 = topHat2_3 - blackHat2_3;
    combinada2_3 = img3 + resta2_3;

    Mat combinada3_3;
    Mat resta3_3 = topHat3_3 - blackHat3_3;
    combinada3_3 = img3 + resta3_3;

    //kernel = 5
    // imshow("Erosión img3", erosion1_3);
    // imshow("Dilatación img3", dilatacion1_3);
    // imshow("Top Hat img3", topHat1_3);
    // imshow("Black Hat img3", blackHat1_3);
    // imshow("Suma (TopHat - BalckHat) img3", combinada1_3);

   // kernel = 25
     // imshow("Erosión img3", erosion2_3);
     // imshow("Dilatación img3", dilatacion2_3);
     // imshow("Top Hat img3", topHat2_3);
     // imshow("Black Hat img3", blackHat2_3);
     // imshow("Suma (TopHat - BalckHat) img3", combinada2_3);

    //kernel = 37
    imshow("Erosión img3", erosion3_3);
    imshow("Dilatación img3", dilatacion3_3);
    imshow("Top Hat img3", topHat3_3);
    imshow("Black Hat img3", blackHat3_3);
    imshow("Suma (TopHat - BalckHat) img3", combinada3_3);

    waitKey(0);
    return 0;
}
