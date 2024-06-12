#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <utility>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include "functions.h"
#include "field_detection.h"


using namespace std;
namespace fs = filesystem;


/*--------------------------------------SPECIFIC FUNCTIONS---------------------------------------------*/
void EROSION(const cv::Mat& current_image,  cv::Mat& closed, int erosion_size) {

    cv::Mat erosion_element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        cv::Point(erosion_size, erosion_size));

    cv::erode(current_image, closed, erosion_element);
}
void DILATION(const cv::Mat& current_image,  cv::Mat& dilated, int dilation_size) {

    cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));

    cv::dilate(current_image, dilated, dilation_element);
}
vector<cv::Point2f> OMOGRAFIA(const vector<cv::Vec3f> circles, const Vec4Points vertices, int width, int height) {

vector<cv::Point2f> points(vertices.val, vertices.val + 4);
    //vertici devono essere messi in senso orario partendo dal vertice in alto a sinstra
    vector<cv::Point2f> dst_points(4);
    dst_points[0] = cv::Point2f(0, 0);
    dst_points[1] = cv::Point2f(width, 0);
    dst_points[2] = cv::Point2f(width, height);
    dst_points[3] = cv::Point2f(0, height);

    cv::Mat homography_matrix = cv::getPerspectiveTransform(vertices, dst_points);

    vector<cv::Point2f> points_to_map(circles.size());
    for (int i = 0; i < circles.size(); i++) {
        points_to_map[i] = cv::Point2f(circles[i][0], circles[i][1]);
    }

    vector<cv::Point2f> mapped_points;
    cv::perspectiveTransform(points_to_map, mapped_points, homography_matrix);

return mapped_points;
}
void Hough_Circles(const cv::Mat& input_img, cv::Mat& img_with_selected_circles, vector<cv::Vec3f>& circles,float min_Dist, float sensibility, int min_Radius, float TH_Circ_A, float TH_Circ_a, float TH_Circ_B, float  th_Ratio_B, float TH_Circ_C, float  th_Ratio_C) {
    cv::Mat gray;
    if (input_img.channels() != 1)
        cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input_img.clone();

    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, min_Dist, 100, sensibility, min_Radius, 16);
}

void select_Circles(vector<cv::Vec3f>& circles, float TH_Circ_A, float TH_Circ_a, float TH_Circ_B, float  TH_Ratio_B, float TH_Circ_C, float  TH_Ratio_C) {
int n_balls = circles.size();
int new_size = circles.size();
vector<bool> is_selected(n_balls, true);

    for (int i = 0; i < n_balls; i++) {
        for (int j = i + 1; j < n_balls; j++) {
            float x_first = circles[i][0], x_second = circles[j][0];
            float y_first = circles[i][1], y_second = circles[j][1];
            float r_first = circles[i][2], r_second = circles[j][2];

                                float min = r_first < r_second ? r_first : r_second;
                                float max = r_first >= r_second ? r_first : r_second;
                                float ratio = min / max;

            float distance_Centers = sqrt( pow((x_second - x_first),2) +  pow((y_second - y_first), 2)  );
            float distance_Circ = distance_Centers - (r_first + r_second);

            if (r_first < 10 && r_second < 10) {

                   if (distance_Circ < TH_Circ_a) {
                       is_selected[r_first < r_second ? i : j] = false;
                       new_size--;
                   }
                   else {
                       if (distance_Circ < 2 && ratio < 0.91) {  // NON ABBASSARE QUESTI 2 THRESHOLD: 1 , 0.91
                           is_selected[r_first < r_second ? i : j] = false;
                           new_size--;
                       }
                       else if (distance_Circ < 6 && ratio < 0.75){  // NON ABBASSARE QUESTI 2 THRESHOLD: 6 , 0.75
                           is_selected[r_first < r_second ? i : j] = false;
                           new_size--;
                       }
                   }
            }
            else {
                   if (distance_Circ < TH_Circ_A) {
                       is_selected[r_first < r_second ? i : j] = false;
                       new_size--;
                   }
                   else {
                           if (distance_Circ < TH_Circ_B && ratio < TH_Ratio_B){  // due threshold per vedere se prende rumori
                               is_selected[r_first < r_second ? i : j] = false;
                               new_size--;
                          }
                           else if (distance_Circ < TH_Circ_C && ratio < TH_Ratio_C){  // due threshold per vedere se prende rumori
                               is_selected[r_first < r_second ? i : j] = false;
                               new_size--;
                          }
                   }
            }
            
        } 

    }
    vector<cv::Vec3f> selected_circles;
    for (int i = 0; i < n_balls; ++i) {
        if (is_selected[i]) {
            selected_circles.push_back(circles[i]);
        }
    }
    /*
    int j = 0;
    vector<cv::Vec3f> selected_circles(new_size);
    for (int i = 0; i < n_balls; ++i) {
        if (is_selected[i]) {
            selected_circles[j] = circles[i];
            j++;
        }
    }*/

circles = selected_circles;
}
void design_Circles(vector<cv::Vec3f>& circles, const cv::Mat& image) {

    for (int j = 0; j < circles.size(); j++) {
        cv::Vec3f c = circles[j];
        cv::Point center = cv::Point(c[0], c[1]);
        cv::circle(image, center, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        int radius = c[2];
        cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    cv::imshow("immagine FINALE cerchiata", image);
    cv::waitKey(0); 
}
void design_Boxes(const vector <vector<cv::Point2f>>& vertices, cv::Mat& image) {

    for(int i = 0; i < vertices.size();i++){

        vector<cv::Point2f> vec = vertices[i];

        cv::Point2f vertice1(vec[0]);
        cv::Point2f vertice2(vec[1]);
        cv::Point2f vertice3(vec[2]);
        cv::Point2f vertice4(vec[3]);

        cv::line(image, vertice1, vertice2, cv::Scalar(255, 0, 255), 2);
        cv::line(image, vertice2, vertice3, cv::Scalar(255, 0, 255), 2);
        cv::line(image, vertice3, vertice4, cv::Scalar(255, 0, 255), 2);
        cv::line(image, vertice4, vertice1, cv::Scalar(255, 0, 255), 2);
    }

    cv::imshow("immagine FINALE con boxes", image);
    cv::waitKey(0);
}

void extractFrames(const string& videoPath, int frameInterval) {
    // Cartella di output per i frame
    string outputFolder = "frames";

    // Crea la cartella di output se non esiste
    filesystem::create_directories(outputFolder);

    // Apri il video
    cv::VideoCapture cap(videoPath);

    // Controlla se il video � stato aperto correttamente
    if (!cap.isOpened()) {
        cerr << "Errore nell'aprire il video" << endl;
    }

    int frameCount = 0;
    cv::Mat frame;
    while (true) {
        // Leggi un frame
        bool success = cap.read(frame);
        if (!success) {
            break; // Fine del video
        }

        // Controlla se questo frame deve essere campionato
        if (frameCount % frameInterval == 0) {
            // Elabora o salva il frame campionato
            string frameFilename = "frame_" + to_string(frameCount) + ".jpg";
            cv::imwrite(frameFilename, frame);
            cout << "Frame " << frameCount << " salvato come " << frameFilename << endl;
        }

        frameCount++;
    }

    cap.release();
}


vector <vector<cv::Point2f> > calculate_SquaresVertices(const vector<cv::Vec3f> & circless) {

    vector < vector<cv::Point2f>> Allvertices(circless.size());

    for (int i = 0; i < circless.size(); i++) {
        cv::Vec3f circle = circless[i];

            float cx = circle[0];
            float cy = circle[1];
            float radius = circle[2];

            vector<cv::Point2f> vertices(4);
            vertices[0] = cv::Point2f(cx - radius, cy - radius);  // Vertice superiore sinistro
            vertices[1] = cv::Point2f(cx + radius, cy - radius);  // Vertice superiore destro
            vertices[2] = cv::Point2f(cx + radius, cy + radius);  // Vertice inferiore destro
            vertices[3] = cv::Point2f(cx - radius, cy + radius);  // Vertice inferiore sinistro

            Allvertices[i] = vertices;
        }
    
return Allvertices;
}
vector<vector<float>> calculate_predictedBoxes(const vector<vector<cv::Point2f>>& allVertices){
int n = allVertices.size();
vector<vector<float>> predictedBoxes(n);

    for (int i = 0; i < n; i++) {
          vector<float> box(4); 
    
          box[0] = allVertices[i][0].x;  // x_min
          box[1] = allVertices[i][0].y;  // y_min
          box[2] = allVertices[i][2].x;  // x_max
          box[3] = allVertices[i][2].y;  // y_max

          predictedBoxes[i] = box;
    }

return predictedBoxes;
}





/*--------------------------------------COMMON FUNCTIONS---------------------------------------------*/
void process_general(const function<void(const cv::Mat&, cv::Mat&)>& function, const vector<cv::Mat>& inputImages, vector<cv::Mat>& output_Image) {
    for (int i = 0; i < inputImages.size(); ++i) 
        function(inputImages[i], output_Image[i]);
}
void process_general(const function<void(const cv::Mat&)>& function, const vector<cv::Mat>& inputImages) {
    for (int i = 0; i < inputImages.size(); ++i) 
        function(inputImages[i]);
}
void process_general(const function<void( cv::Mat&)>& function,  vector<cv::Mat>& inputImages) {
    for (int i = 0; i < inputImages.size(); ++i)
        function(inputImages[i]);
}

void Loading_images(vector<cv::Mat>& images, const vector<string>& imagePaths) {
    try {
        for (const string& imagePath : imagePaths) {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) 
                throw runtime_error("Error loading image file: " + imagePath);
            
            images.push_back(image);
        }
    }
    catch (const runtime_error& e) {
        cerr << "Exception caught: " << e.what() << endl;
        // Operazioni di pulizia, se necessario
        exit(EXIT_FAILURE); // Termina il programma con un codice di errore
    }
}

void show_image(const char* name_image, const cv::Mat image) {
    cv::imshow(name_image, image);
    cv::waitKey(0);
    cv::destroyWindow(name_image);
}
void show_image(const char* name_image, const vector<cv::Mat> Images) {
    int idx = 0;
    for (const cv::Mat& img : Images) {
        string unique_name = string(name_image) + "_" + to_string(idx);
        show_image(unique_name.c_str(), img);
        idx++;
    }
}
void show_image(const char* name_image, const vector<  vector<cv::Mat> > Images) {
    for (int i = 0; i < Images[0].size(); i++) {
        for (int j = 0; j < Images.size(); j++) {
            string unique_name = string(name_image) + "_" + to_string(i) +"//" + to_string(j);
            cv::imshow(unique_name.c_str(), Images[j][i]);
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}



