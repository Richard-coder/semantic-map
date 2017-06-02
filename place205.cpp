#include "Classifier.h"
int main(int argc, char** argv) {

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = "/home/richard/ros-semantic-mapper/deploy.prototxt";
  string trained_file = "/home/richard/ros-semantic-mapper/places.caffemodel";
  string mean_file    = "/home/richard/ros-semantic-mapper/places205CNN_mean.binaryproto";
  string label_file   = "/home/richard/Desktop/place205_c++_test/index.txt";

  string file = "/home/richard/Desktop/data/rgb_png/2.png";
  Classifier classifier(model_file, trained_file, mean_file, label_file);
  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
