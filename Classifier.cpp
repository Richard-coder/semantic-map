#include "Classifier.h"
#include <numeric>
SemanticLabel::SemanticLabel(const string filename)
{
  std::ifstream fin(filename.c_str());
  if (!fin)
  {
    std::cerr << "parameter file does not exist." << std::endl;
  }
  string line;
  while (std::getline(fin, line))
  {
    if (line[0] == '#')
    {
      // 以‘＃’开头的是注释
      continue;
    }
    int pos_find = line.find(' ');
    int pos_start = 0;
    if (pos_find == -1)
      continue;
    string labelname_ = line.substr(0, pos_find);
    labelname.push_back(labelname_);
    pos_start = pos_find + 1;
    pos_find = line.find(' ', pos_start);
    int labelidx_ = std::atoi(line.substr(pos_start, pos_find - pos_start).c_str());
    labelidx.push_back(labelidx_);
    pos_start = pos_find + 1;
    std::vector<int> color;
    //std::vector<unsigned char> color;
    pos_find = line.find(",", pos_start);

    color.push_back(std::atoi(line.substr(pos_start, pos_find - pos_start).c_str()));
    pos_start = pos_find + 1;
    pos_find = line.find(",", pos_start);
    color.push_back(std::atoi(line.substr(pos_start, pos_find - pos_start).c_str()));
    pos_start = pos_find + 1;
    color.push_back(std::atoi(line.substr(pos_start, line.size() + 1 - pos_start).c_str()));
    labelcolor.push_back(color);

    /*
    int icolor;
    unsigned char uccolor;

    icolor = (std::atoi(line.substr(pos_start, pos_find - pos_start).c_str()));
    uccolor = icolor;
    color.push_back(uccolor);
    pos_start = pos_find + 1;
    pos_find = line.find(",", pos_start);
    icolor = (std::atoi(line.substr(pos_start, pos_find - pos_start).c_str()));
    uccolor = icolor;
    color.push_back(uccolor);
    pos_start = pos_find + 1;
    icolor = (std::atoi(line.substr(pos_start, line.size() + 1 - pos_start).c_str()));
    uccolor = icolor;
    color.push_back(uccolor);
    labelcolor.push_back(color);
    */
  }
}
Classifier::Classifier(const string &model_file,
                       const string &trained_file,
                       const string &mean_file,
                       const SemanticLabel &semlabel)
{

  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  semlabel_ = semlabel;
  Blob<float> *output_layer = net_->output_blobs()[0];
  /*CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
*/
}

static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs)
{
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float> &v, int N)
{
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat &img, int N)
{
  std::vector<float> output = Predict(img);

  //N = std::min<int>(labels_.size(), N);
  //std::vector<int> maxN = Argmax(output, N);
  //归一化
  //float sum = accumulate(output.begin(), output.end(), 0);
  std::vector<Prediction> predictions;
  //std::cout << sum << std::endl;
  for (int idx = 0; idx < semlabel_.labelidx.size(); idx++)
  {
    //predictions.push_back(std::make_pair(idx, output[semlabel_.labelidx[idx]] / sum));
    predictions.push_back(std::make_pair(idx, output[semlabel_.labelidx[idx]]));
  }
  double presum = 0;
  for (int idx = 0; idx < predictions.size(); idx++)
  {
    presum += predictions[idx].second;
  }
  for (int idx = 0; idx < predictions.size(); idx++)
  {
    predictions[idx].second = (float)(predictions[idx].second / presum);
  }
  /*
  presum = 0;
  for (int idx = 0; idx < predictions.size(); idx++)
  {
    presum += predictions[idx].second;
  }
  std::cout<<presum<<std::endl;
*/
  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string &mean_file)
{
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float *data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i)
  {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat &img)
{
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
  Blob<float> *input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i)
  {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat &img,
                            std::vector<cv::Mat> *input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}