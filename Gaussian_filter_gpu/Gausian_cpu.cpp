#include "Gaussian.h"

// cpu gaussian filtering using OpenCV library
extern "C" void blurring_gaussian(Mat src, float sigma, int ksize)
{
    if (src.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    //imshow("src", src);

    Mat dst;
    uint64_t total_time = 0;

    for (int iIdx = 0; iIdx < ITER; iIdx++) {

        uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

        GaussianBlur(src, dst, Size(ksize, ksize), (double)sigma);

        total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;

    }
    printf("======================================================\n\n");
    printf("3. OpenCV Gaussian filtering avg_dur_time=%6.3f[msec] \n\n", total_time / 1000.f / ITER);
    printf("======================================================\n");

    String desc = format("OpenCV Gaussian : sigma = %f", sigma);
    putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

    imshow("opencv", dst);
    //waitKey();
    destroyAllWindows();
}


// transfer mat to vector
extern "C" void mat_to_vector(vector<float> &out, Mat in)
{
    if (out.size() == 0) {// 벡터 크기가 0일때
        for (int i = 0; i < in.rows; i++) {
            for (int j = 0; j < in.cols; j++) {
                //cout << in.at<double>(i, j) << endl;
                out.push_back(in.at<double>(i, j));
            }
        }
    }
    else {
        for (int i = 0; i < in.rows; i++) {
            for (int j = 0; j < in.cols; j++) {
                //cout << in.at<double>(i, j) << endl;
                out[i * in.cols + j] = in.at<double>(i, j);
            }
        }
    }
}


extern "C" void mat_to_vector2(vector<float> &out, Mat in)
{
    if (out.size() == 0) {// 벡터 크기가 0일때
        for (int i = 0; i < in.rows; i++) {
            for (int j = 0; j < in.cols; j++) {
                //cout << in.at<double>(i, j) << endl;
                out.push_back(in.at<uchar>(i, j));
            }
        }
    }
    else {
        for (int i = 0; i < in.rows; i++) {
            for (int j = 0; j < in.cols; j++) {
                //cout << in.at<double>(i, j) << endl;
                out[i * in.cols + j] = in.at<uchar>(i, j);
            }
        }
    }
}

// 1d filter value check 
extern "C" void filter_check(vector<float> filter) {
    cout << "filter size(ksize) :: " << filter.size() << endl;
    for (int i = 0; i < filter.size(); i++) {
        cout << filter[i] << endl;
    }
}


extern "C" void valueCheck2d(vector<float>& valueCheckInput, int input_h, int input_w)
{
    for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++) {
        int temp4 = ⁠h_idx * input_w; cout << "  ";
        for (int w_idx = 0; w_idx < input_w; w_idx++)
        {
            int g_idx = w_idx + temp4;
            //cout.setf(ios::fixed);//cout.precision(6);
            cout << setw(8) << valueCheckInput[g_idx] << " ";
        }cout << endl;
    }cout << endl;
}

extern "C" void inititalizedData(vector<float>& container)
{
    int count = 0;
    for (vector<int>::size_type i = 0; i < container.size(); i++) {
        container[i] = count;
        count++;
    }
}

extern "C" void inititalizedDataOne(vector<float>& container)
{
    int count = 1;
    for (vector<int>::size_type i = 0; i < container.size(); i++) {
        container[i] = count;
    }
}

extern "C" void array_to_mat(Mat out, vector<float> in)
{
    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            out.at<uchar>(i, j) = (int)in[i * out.cols + j];
        }
    }
}

extern "C" void int_array_to_mat(Mat out, vector<int> in)
{
    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            out.at<uchar>(i, j) = (int)in[i * out.cols + j];
        }
    }
}


extern "C" void convolution(vector<float>& convOutput, vector<float>& convInput, vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
    int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
    int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
    //Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
    //cout << "===== Convolution ===== \n";
    int temp1i = input_h * input_w * input_c;
    int temp1o = outputHeightSize * outputWidthSize * ouput_c;
    int temp1k = kernelSize * kernelSize * input_c;
    for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
    {
        int temp2i = ⁠n_idx * temp1i;
        int temp2o = ⁠n_idx * temp1o;
        for (int k_idx = 0; k_idx < ouput_c; k_idx++)
        {
            int temp2k = k_idx * temp1k;
            int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
            for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
            {
                int temp3i = ⁠c_idx * input_w * input_h + temp2i;
                int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
                for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
                    int temp4o = rowStride * outputWidthSize + temp3o;
                    for (int colStride = 0; colStride < outputWidthSize; colStride++) {

                        float sum = 0;
                        int g_idx_o = colStride + temp4o;
                        for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
                            int temp4i = x * input_w + temp3i;
                            int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
                            for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
                                int ⁠g_idx_i = y + temp4i;
                                int g_idx_k = (y - colStride * stride) + temp4k;
                                sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
                            }
                        }
                        convOutput[g_idx_o] += sum;
                    }
                }
            }
        }
    }
}

extern "C" void gaussianFilter1d(vector <float> &output, float sigma, int k_radius, int odd) {
    float pi = 3.14159265358979f;
    float summation = 0;
    for (int y = -k_radius; y <= k_radius - odd; y++) {
        int y_idx = y + k_radius;
        int offset = (y_idx)* (k_radius * 2 + 1 - odd);
        float sum = 0;
        for (int x = -k_radius; x <= k_radius - odd; x++) {
            int x_idx = x + k_radius;
            int g_idx = offset + x_idx;
            sum += (1. / (2. * pi * pow(sigma, 2))) * exp(-((pow(x, 2) + pow(y, 2)) / (2. * pow(sigma, 2))));
        }
        output[y_idx] = sum;
        summation += output[y_idx];
    }
    for (int y = -k_radius; y <= k_radius - odd; y++) {
        int y_idx = y + k_radius;
        output[y_idx] /= summation;
    }
}



extern "C" void reflectivePadding(vector<float>& output, vector<float>& input, int h, int w, int k_radius, int odd)
{
    vector<float> data_p(h  * ((w + k_radius * 2) - odd));
    for (int ⁠h_idx = 0; ⁠h_idx < h; ⁠h_idx++) {
        int offset = ⁠h_idx * ((w + k_radius * 2) - odd);
        for (int w_idx = 0; w_idx < (w + k_radius * 2) - odd; w_idx++) {
            int g_idx = w_idx + offset;
            if (w_idx < k_radius - odd) {
                int i_idx = k_radius - odd - w_idx + ⁠h_idx * w;
                data_p[g_idx] = input[i_idx];
            }
            else if (w_idx >= w + k_radius - odd) {
                int i_idx = 2 * w - w_idx + k_radius - odd + ⁠h_idx * w - 2;
                data_p[g_idx] = input[i_idx];
            }
            else {
                int i_idx = w_idx - (k_radius - odd) + ⁠h_idx * w;
                data_p[g_idx] = input[i_idx];
            }
        }
    }

    for (int ⁠h_idx = 0; ⁠h_idx < (h + k_radius * 2) - odd; ⁠h_idx++) {
        int offset = ⁠h_idx * (w + k_radius * 2 - odd);
        for (int w_idx = 0; w_idx < w + k_radius * 2 - odd; w_idx++) {
            int g_idx = w_idx + offset;
            if (⁠h_idx < k_radius - odd) {
                int i_idx = ((k_radius - odd) - ⁠h_idx) * (w + k_radius * 2 - odd) + w_idx;
                output[g_idx] = data_p[i_idx];
            }
            else if (⁠h_idx > h + k_radius - (1 + odd)) {
                int i_idx = (h * 2 + (k_radius - odd) - (⁠h_idx + 2))* (w + k_radius * 2 - odd) + w_idx;
                output[g_idx] = data_p[i_idx];
            }
            else {
                int i_idx = g_idx - ((k_radius - odd) * ((w + k_radius * 2) - odd));
                output[g_idx] = data_p[i_idx];
            }
        }
    }
}

extern "C" void gaussianFilter2d(vector <float> &output, float sigma, int k_radius, int odd)
{
    float pi = 3.14159265358979f;
    float summation = 0;
    for (int y = -k_radius; y <= k_radius - odd; y++) {
        int y_idx = y + k_radius;
        int offset = (y_idx)* (k_radius * 2 + 1 - odd);
        for (int x = -k_radius; x <= k_radius - odd; x++) {
            int x_idx = x + k_radius;
            int g_idx = offset + x_idx;
            output[g_idx] = (1. / (2. * pi * pow(sigma, 2))) * exp(-((pow(x, 2) + pow(y, 2)) / (2. * pow(sigma, 2))));
            summation += output[g_idx];
        }
    }

    for (int y = -k_radius; y <= k_radius - odd; y++) {
        int y_idx = y + k_radius;
        int offset = (y_idx)* (k_radius * 2 + 1 - odd);
        for (int x = -k_radius; x <= k_radius - odd; x++) {
            int x_idx = x + k_radius;
            int g_idx = offset + x_idx;
            output[g_idx] /= summation;
        }
    }
}

extern "C" void valueCheck2d_6(vector<float>& valueCheckInput, int input_h, int input_w)
{
    for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++) {
        int temp4 = ⁠h_idx * input_w; cout << "  ";
        for (int w_idx = 0; w_idx < input_w; w_idx++)
        {
            int g_idx = w_idx + temp4;
            cout.setf(ios::fixed);
            cout.precision(6);
            cout << setw(9) << valueCheckInput[g_idx] << " ";
        }cout << endl;
    }cout << endl;
}