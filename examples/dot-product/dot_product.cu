#include <iostream>
#include <vector>
#include <chrono>
#include <random>
using namespace std;

float dot_product_cpu(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    const int N = 50'000'000;  // 50 million elements

    // Allocate arrays
    float *a = new float[N];
    float *b = new float[N];

    // Fill arrays with random float data
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    cout << "Running dot product on CPU..." << endl;
    auto start = chrono::high_resolution_clock::now();

    float result = dot_product_cpu(a, b, N);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Dot product result: " << result << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    delete[] a;
    delete[] b;

    return 0;
}
