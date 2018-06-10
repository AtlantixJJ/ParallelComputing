#include "gen.h"

int main(int argc, char* argv[])
{
    int n, i, j, num_threads;
    float** A;
    float* s;
    int* b;

    // get problem size n and number of threads from command line
    if(argc < 3) {
        printf("Usage: ./target n size.\n");
    }
    n = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    // generating n-by-n martrix A, generating integer vector b of length n, where 0<b(i)<n
    gen(&A, &b, n);

    // do the computation of s
    s = (float*)malloc(sizeof(float) * n);
    for(i = 0; i < n; i++) {
        s[i] = 0.0;
        for(j = 0; j < b[i]; j++) {
            s[i] += A[i][j];
        }
        s[i] = s[i] * 1.0 / b[i];
    }

    printResult(s,n);
    freeRes(A, b, s, n);
    return 0;
}
