#include <iostream>
#include <vector>
#include <chrono>
#include <random>
using namespace std;

int dot_product_cpu(int *a, int *b, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main(void) {
    const int N = 500'000'000;  // 500 million elements

    // Allocate arrays
    int *a = new int[N];
    int *b = new int[N];

    // Fill arrays with random data
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    cout << "Running dot product on CPU..." << endl;
    auto start = chrono::high_resolution_clock::now();

    int result = dot_product_cpu(a, b, N);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Dot product result: " << result << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    // Clean up
    delete[] a;
    delete[] b;

    return 0;
}
