#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void convolution_cpu(unsigned int *image, unsigned int *filter, unsigned int *result, 
                     int M, int N) {
    int pad = N / 2;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int sum = 0;
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    int ii = i - pad + fi;
                    int jj = j - pad + fj;
                    if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                        sum += image[ii * M + jj] * filter[fi * N + fj];
                    }
                }
            }
            result[i * M + j] = sum > 0 ? sum : 0;
        }
    }
}

void create_edge_filter(unsigned int *filter, int N) {
    if (N == 3) {
        unsigned int edge[9] = {4294967295, 4294967295, 4294967295, 4294967295, 8, 4294967295, 4294967295, 4294967295, 4294967295};
        for (int i = 0; i < 9; i++) {
            filter[i] = edge[i];
        }
    } else {
        for (int i = 0; i < N * N; i++) {
            filter[i] = 1;
        }
    }
}

void create_test_image(unsigned int *image, int M) {
    for (int i = 0; i < M * M; i++) {
        image[i] = rand() % 256;
    }
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <image_size> <filter_size>\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    
    unsigned int *image = malloc(M * M * sizeof(unsigned int));
    unsigned int *filter = malloc(N * N * sizeof(unsigned int));
    unsigned int *result = malloc(M * M * sizeof(unsigned int));
    
    srand(42);
    create_test_image(image, M);
    create_edge_filter(filter, N);
    
    double start = getTime();
    convolution_cpu(image, filter, result, M, N);
    double end = getTime();
    
    printf("%d,%d,%f\n", M, N, end - start);
    
    free(image);
    free(filter);
    free(result);
    
    return 0;
}