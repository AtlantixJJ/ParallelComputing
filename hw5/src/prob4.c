/*
 * File:     prog3.5_mpi_mat_vect_col.c
 *
 * Purpose:  Implement matrix vector multiplication when the matrix
 *           has a block column distribution
 *
 * Compile:  mpicc -g -Wall -o prog3.5_mpi_mat_vect_col
 * prog3.5_mpi_mat_vect_col.c Run:      mpiexec -n <comm_sz>
 * ./prog3.5_mpi_mat_vect_col
 *
 * Input:    order of matrix, matrix, vector
 * Output:   product of matrix and vector.  If DEBUG is defined, the
 *           order, the input matrix and the input vector
 *
 * Notes:
 * 1.  The matrix should be square and its order should be evenly divisible
 *     by comm_sz
 * 2.  The program stores the local matrices as one-dimensional arrays
 *     in row-major order
 * 3.  The program uses a derived datatype for matrix input and output
 *
 * Author:   Jinyoung Choi
 *
 * IPP:      Programming Assignment 3.5
 */
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

pthread_mutex_t gmutex;

struct thread_data_t {
  int pid;
  int n;
  int *vector;
  double *matrix;
  double *result;
};

int global_step = 0;
pthread_mutex_t gmutex;

/// generate m x n random matrix
void gen_matrix(int m, int n, double* matrix) {
  int i, j;
  srand((unsigned)time(0));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      matrix[i * n + j] = -1 + rand() * 1.0 / RAND_MAX * 2;
}

/// generate integer vector
void gen_vector(int n, int* vector) {
  int i;
  for (i = 0; i < n; i++) {
    vector[i] = (double)rand() / RAND_MAX * (double)n;
  }
}

/// serial summation: do not require clear the result array
void serial_sum(int n, double* matrix, int* vector, double* s) {
  int i, j, ed;
  double temp;
  for (i = 0; i < n; i++) {
    ed = vector[i];

    temp = 0;
    for (j = 0; j < ed; j++) {
      temp += matrix[i * n + j];
    }
    s[i] = temp;
  }
}

void print_matrix(int n, double* matrix) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%lf ", matrix[i * n + j]);
    }
    printf("\n");
  }
}

void openmp_sum_static(int n, int n_thread, double* matrix, int* vector, double* s) {
  int i, j, ed;
  double temp;
  #pragma omp parallel for num_threads(n_thread) private(i, j, temp, ed) schedule(static, 1024)
  for (i = 0; i < n; i++) {
    ed = vector[i];

    temp = 0;
    for (j = 0; j < ed; j++) {
      temp += matrix[i * n + j];
    }
    s[i] = temp;
  }
}

void openmp_sum_dynamic(int n, int n_thread, double* matrix, int* vector, double* s) {
  int i, j, ed;
  double temp;
  #pragma omp parallel for num_threads(n_thread) private(i, j, temp, ed) schedule(dynamic, 1024)
  for (i = 0; i < n; i++) {
    ed = vector[i];

    temp = 0;
    for (j = 0; j < ed; j++) {
      temp += matrix[i * n + j];
    }
    s[i] = temp;
  }
}

void thread_sum_worker(void *arg) {
  struct thread_data_t *parg = (struct thread_data_t*)arg;
  double *matrix = parg->matrix;
  int *vector = parg->vector;
  int cur, n = parg->n;
  // get global step
  while(global_step < n) {
    pthread_mutex_lock(&gmutex);
    if(global_step >= n) {
      pthread_mutex_unlock(&gmutex);
      break;
    }
    cur = global_step;
    ++global_step;
    pthread_mutex_unlock(&gmutex);
    
    // compute base on cur
    int i, ed;
    double temp = 0;
    ed = vector[cur];
    for (i = 0; i < ed; i++) {
      temp += matrix[cur * n + i];
    }
    parg->result[cur] = temp;
  }
}

int verify_serial(int n, double *vec1, double *vec2) {
  int i;
  double r = 0;
  for(i = 0; i < n; i ++) {
    r += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  if(r < 0.0001) return 1;
  else return 0;
}

//#define DEBUG

void run(int n, int n_thread) {
  double* matrix = (double*)malloc(n * n * sizeof(double));
  int* vector = (int*)malloc(n * sizeof(int));
  // serial, openmp static, openmp dynamic, pthread
  double* result1 = (double*)malloc(n * sizeof(double));
  double* result2 = (double*)malloc(n * sizeof(double));
  double* result3 = (double*)malloc(n * sizeof(double));
  double* result4 = (double*)malloc(n * sizeof(double));

  int i, j;
  // timing
  struct timeval daytime;
  double bg, ed;
  double serial_time = 0, openmp_time1 = 0, openmp_time2 = 0, pthread_time = 0;

  gen_matrix(n, n, matrix);
  gen_vector(n, vector);

  gettimeofday(&daytime, NULL);
  bg = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  serial_sum(n, matrix, vector, result1);
  gettimeofday(&daytime, NULL);
  ed = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  serial_time = ed - bg;

  gettimeofday(&daytime, NULL);
  bg = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  openmp_sum_static(n, n_thread, matrix, vector, result2);
  gettimeofday(&daytime, NULL);
  ed = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  openmp_time1 = ed - bg;

  gettimeofday(&daytime, NULL);
  bg = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  openmp_sum_static(n, n_thread, matrix, vector, result3);
  gettimeofday(&daytime, NULL);
  ed = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  openmp_time2 = ed - bg;

  // pthread
  pthread_mutex_init(&gmutex, NULL);  
  pthread_t *thread_pool = (pthread_t*)malloc(sizeof(pthread_t) * n_thread);
  struct thread_data_t *thread_data = (struct thread_data_t*)malloc(sizeof(struct thread_data_t) * n_thread);

  gettimeofday(&daytime, NULL);
  bg = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;

  for(i = 0; i < n_thread; i++) {
    thread_data[i].matrix = matrix;
    thread_data[i].vector = vector;
    thread_data[i].result = result4;
    thread_data[i].pid = i;
    thread_data[i].n = n;

    pthread_create(thread_pool + i, NULL, thread_sum_worker, thread_data + i);
  }

  for(i = 0; i < n_thread; i++) {
    pthread_join(thread_pool[i], NULL);
  }

  gettimeofday(&daytime, NULL);
  ed = daytime.tv_sec * 1000.0 + daytime.tv_usec / 1000.0;
  pthread_time = ed - bg;

  pthread_mutex_destroy(&gmutex);
  free(thread_pool);
  free(thread_data);

#ifdef DEBUG
  printf("Matrix:-----\n");
  print_matrix(n, matrix);
  printf("Vector:-----\n");
  for (i = 0; i < n; i++) printf("%d ", vector[i]); printf("\n");
  printf("Result:-----\n");
  for (i = 0; i < n; i++) printf("%lf ", result1[i]); printf("\n");
  for (i = 0; i < n; i++) printf("%lf ", result2[i]); printf("\n");
  for (i = 0; i < n; i++) printf("%lf ", result3[i]); printf("\n");
#endif

  int flag = 0;
  flag = verify_serial(n, result1, result2);
  if(flag != 1) printf("OpenMP static parallel error.\n");
  flag = verify_serial(n, result1, result3);
  if(flag != 1) printf("OpenMP dynamic parallel error.\n");
  flag = verify_serial(n, result1, result4);
  if(flag != 1) printf("PThread parallel error.\n");

  printf("%lf,%lf,%lf,%lf\n", serial_time, openmp_time1, openmp_time2, pthread_time);

  FILE *file = fopen("log_prob4.txt", "aw");
  fprintf(file, "%lf,%lf,%lf,%lf\n", serial_time, openmp_time1, openmp_time2, pthread_time);
  close(file);

  free(matrix);
  free(vector);
  free(result1); free(result2); free(result3); free(result4);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s n\n", argv[0]);
    return -1;
  }
  int n = atoi(argv[1]);
  int n_thread = atoi(argv[2]);

  run(n, n_thread);
}
