#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <algorithm>
#include <boost/compute.hpp>
#include <boost/compute/interop/opencv/core.hpp>

namespace compute = boost::compute;
using namespace cv;

static int num_rho_buckets;
static int num_angle_buckets = 720;
static float max_rho = 0;


void init(Mat &img)
{
  int width = img.cols;
  int height = img.rows;
  max_rho = ceil(std::max(width, height) * sqrt(2));
  num_rho_buckets = 2 * int(max_rho)/ 2;
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

  float max = 0;
  for (int j = 0; j < hs.rows; j++) {
    for (int i = 0; i < hs.cols; i++) {
      max = std::max(max, hs.at<float>(j, i));
    }
  }

  for (int j = 0; j < hs.rows; j++) {
    for (int i = 0; i < hs.cols; i++) {
      hs.at<float>(j, i) = hs.at<float>(j, i) / max;
    }
  }
  return hs;
}

const char hough_transform_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE (
  __kernel void hough_transform(read_only image2d_t edges,
                                __global int *hs,
                                int num_rho_buckets,
                                float max_rho,
                                int num_angle_buckets) {
    sampler_t sampler =( CLK_NORMALIZED_COORDS_FALSE |
                         CLK_FILTER_NEAREST |
                         CLK_ADDRESS_CLAMP_TO_EDGE);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int angle_bucket = get_global_id(2);

    int2 coord = (int2)(x, y);
    int is_edge = read_imagei(edges, sampler, coord).x;
    if (!is_edge) return;

    float theta = M_PI_F * float(angle_bucket) / float(num_angle_buckets);
    float rho = x * cos(theta) + y * sin(theta);
    int   rho_bucket = int(num_rho_buckets * ((rho / (2 * max_rho)) + 0.5));
    int index = angle_bucket + num_angle_buckets * rho_bucket;
    atomic_add(&hs[index], 1);
  }
);

Mat hough_lines_acc_compute(Mat &h_edges)
{
  compute::device device = compute::system::default_device();
  compute::context context(device);
  compute::command_queue queue(context, device);

  Mat h_hs(num_rho_buckets, num_angle_buckets, CV_32FC1);
  compute::vector<int> d_hs(num_rho_buckets * num_angle_buckets, context);
  std::vector<int> h_hs_vec(num_rho_buckets * num_angle_buckets);

  compute::program hough_transform_program = compute::program::create_with_source(hough_transform_source, context);
  try {
    hough_transform_program.build();
  } catch(compute::opencl_error e) {
    std::cout << "Build Error: " << std::endl
              << hough_transform_program.build_log();
    exit(-1);
  }
  compute::kernel hough_transform(hough_transform_program, "hough_transform");

  compute::image2d d_edges = compute::opencv_create_image2d_with_mat(h_edges, compute::image2d::read_only, queue);

  hough_transform.set_arg(0, d_edges);
  hough_transform.set_arg(1, d_hs.get_buffer());
  hough_transform.set_arg(2, num_rho_buckets);
  hough_transform.set_arg(3, max_rho);
  hough_transform.set_arg(4, num_angle_buckets);

  size_t global_size[3] = { (size_t)h_edges.cols, (size_t)h_edges.rows, (size_t)num_angle_buckets };
  queue.enqueue_nd_range_kernel(hough_transform, 3, NULL, global_size, NULL);
  queue.finish();
  // compute::copy(d_hs.begin(), d_hs.end(), h_hs.data, queue);
  compute::copy(d_hs.begin(), d_hs.end(), h_hs_vec.begin(), queue);

  int max = 0;
  for (int j = 0; j < h_hs.rows; j++) {
    for (int i = 0; i < h_hs.cols; i++) {
      int index = j * h_hs.cols + i;
      max = std::max(max, h_hs_vec[index]);
    }
  }

  for (int j = 0; j < h_hs.rows; j++) {
    for (int i = 0; i < h_hs.cols; i++) {
      int index = j * h_hs.cols + i;
      h_hs.at<float>(j, i) = float(h_hs_vec[index]) / float(max);
    }
  }

  return h_hs;
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

  Mat houghSpace = hough_lines_acc_compute(edges);
  imshow("Hough Space - GPU", houghSpace);

  // Mat houghSpace = hough_lines_acc(edges);
  // imshow("Hough Space - CPU", houghSpace);

  auto peaks = hough_peaks(houghSpace, 20);

  hough_draw_lines(img, *peaks);
  imshow("Highlighted Lines", img);
  waitKey(0);

  return 0;
}
