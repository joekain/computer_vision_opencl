#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace cv;

static int num_rho_buckets;
static int num_angle_buckets = 720;
static float max_rho = 0;

void init(Mat &img)
{
  int width = img.cols;
  int height = img.rows;
  max_rho = ceil(std::max(width, height) * sqrt(2));
  num_rho_buckets = 2 * int(max_rho) / 2;
}

float bucket_to_radians(int bucket)
{
  return M_PI * float(bucket) / float(num_angle_buckets);
}

int rho_to_bucket(float rho)
{
  // return 2 * max_rho * rho / float(num_rho_buckets);
  return int(num_rho_buckets * ((rho / (2 * max_rho)) + 0.5));
}

float bucket_to_rho(int bucket) {
  return 2 * max_rho * ((bucket) / float(num_rho_buckets) - 0.5);
}

void hough_line_acc(Mat &hs, int x, int y)
{
  for (int bucket = 0; bucket < num_angle_buckets; bucket++) {
    float theta = bucket_to_radians(bucket);
    float rho = x * cos(theta) + y * sin(theta);
    hs.at<float>(rho_to_bucket(rho), bucket)++;
  }
}

Mat hough_lines_acc(Mat &edges)
{
  Mat hs(num_rho_buckets, num_angle_buckets, CV_32FC1);

  for (int y = 0; y < edges.rows; y++) {
    for (int x = 0; x < edges.cols; x++) {
      if (edges.at<unsigned char>(y, x)) {
        hough_line_acc(hs, x, y);
      }
    }
  }
  return hs;
}

struct indices {
  int i;
  int j;
};

struct item_and_indices {
  float value;
  int i;
  int j;
};

bool comp(item_and_indices i1, item_and_indices i2)
{
  return i1.value < i2.value;
}

std::vector<indices>* hough_peaks(Mat &hs, int n)
{
  int width, height;

  width = hs.cols;
  height = hs.rows;

  std::vector<item_and_indices> v(width * height);

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      item_and_indices ii = {hs.at<float>(i, j), i, j};
      v.push_back(ii);
    }
  }
  std::make_heap(v.begin(), v.end(), comp);

  auto *result = new std::vector<indices>(n);
  for (int i = 0; i < n; i++) {
    indices ii = {v.front().i, v.front().j};
    result->push_back(ii);

    pop_heap(v.begin(), v.end(), comp);
    v.pop_back();
  }

  return result;
}

void hough_draw_peaks(Mat &img, std::vector<indices> peaks)
{
  for (auto const &indices: peaks) {
    Point2i center(indices.i, indices.j);
    circle(img, center, 4, 255);
  }
}

void hough_draw_lines(Mat &img, std::vector<indices> peaks)
{
  Scalar_<unsigned char> green(0, 255, 0);

  for (auto const &indices: peaks) {
    float theta = bucket_to_radians(indices.j);
    float rho = bucket_to_rho(indices.i);

    float s = sin(theta);
    float c = cos(theta);

    if (fabs(c) < 0.001) {
      int y = int(rho / s);
      Point2i a(1000, y), b(0, y);
      line(img, a, b, green);
    } else if (fabs(s) < 0.001) {
      int x = int(rho / c);
      Point2i a(x, 0), b(x, 1000);
      line(img, a, b, green);
    } else {
      int x = int(rho / c);
      int y = int(rho / s);
      Point2i a(x, 0), b(0, y);
      line(img, a, b, green);

      int x2 = int((rho - 1000 * s) / c);
      int y2 = int((rho - 1000 * c) / s);
      Point2i c(x2, 1000);
      line(img, a, c, green);

      Point2i d(1000, y2);
      line(img, b, d, green);
    }
  }
}

int main()
{
  Mat img = imread("input/ps1-input1.png");
  if ( !img.data ) {
    printf("No image data \n");
    return -1;
  }
  init(img);
  imshow("Input", img);


  Mat grey = img.clone();
  cvtColor(img, grey, COLOR_BGR2GRAY);
  imshow("Input - Grey", grey);

  Mat smooth = grey;  // No smoothing

  Mat edges;
  Canny(smooth, edges, 100, 200);
  imshow("Edges", edges);

  Mat houghSpace = hough_lines_acc(edges);
  imshow("Hough Space", houghSpace);

  auto peaks = hough_peaks(houghSpace, 10);

  hough_draw_lines(img, *peaks);
  imshow("Highlighted Lines", img);
  waitKey(0);

  return 0;
}
