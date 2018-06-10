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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int verify_parallel(int n, int local_n, int my_rank, double *vec1, double *vec2, MPI_Comm comm);
int verify_serial(int n, double *vec1, double *vec2);
void serial(int n, double* matrix, double* vector, double* result);
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Get_dims(int* m_p,
              int* local_m_p,
              int* n_p,
              int* local_n_p,
              int my_rank,
              int comm_sz,
              MPI_Comm comm);
void Allocate_arrays(double** local_A_pp,
                     double** local_x_pp,
                     double** local_y_pp,
                     int m,
                     int local_m,
                     int local_n,
                     MPI_Comm comm);
void Build_derived_type(
    int m, int local_m, int n, int local_n, MPI_Datatype* block_col_mpi_t_p);
double* Read_matrix(char prompt[],
                 double local_A[],
                 int m,
                 int local_n,
                 int n,
                 MPI_Datatype block_col_mpi_t,
                 int my_rank,
                 MPI_Comm comm);
void Print_matrix(char title[],
                  double local_A[],
                  int m,
                  int local_n,
                  int n,
                  MPI_Datatype block_col_mpi_t,
                  int my_rank,
                  MPI_Comm comm);
double* Read_vector(char prompt[],
                 double local_vec[],
                 int n,
                 int local_n,
                 int my_rank,
                 MPI_Comm comm);
void Print_vector(char title[],
                  double local_vec[],
                  int n,
                  int local_n,
                  int my_rank,
                  MPI_Comm comm);
double* Save_vector(char title[],
                  double local_vec[],
                  int n,
                  int local_n,
                  int my_rank,
                  MPI_Comm comm);
void Mat_vect_mult(double local_A[],
                   double local_x[],
                   double local_y[],
                   int local_m,
                   int m,
                   int n,
                   int local_n,
                   int comm_sz,
                   MPI_Comm comm);

/*-------------------------------------------------------------------*/

int run(int m, int run_serial) {
  double* local_A;
  double* local_x;
  double* local_y;

  double *A_root, *vec_root, *res_root = (double*)malloc(m * sizeof(double)), *res_root_p;

  int n, i, j;
  int local_m, local_n;

  int my_rank, comm_sz;
  MPI_Comm comm;
  MPI_Datatype block_col_mpi_t;

  MPI_Init(NULL, NULL);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &my_rank);

  Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);
  Allocate_arrays(&local_A, &local_x, &local_y, m, local_m, local_n, comm);
  Build_derived_type(m, local_m, n, local_n, &block_col_mpi_t);
  A_root = Read_matrix("A", local_A, m, local_n, n, block_col_mpi_t, my_rank, comm);
#ifdef DEBUG
  Print_matrix("A", local_A, m, local_n, n, block_col_mpi_t, my_rank, comm);
#endif

  vec_root = Read_vector("x", local_x, n, local_n, my_rank, comm);

#ifdef DEBUG
  Print_vector("x", local_x, n, local_m, my_rank, comm);
#endif

  double serial_time = 0, para_time = MPI_Wtime();
  Mat_vect_mult(local_A, local_x, local_y, local_m, m, n, local_n, comm_sz,
                comm);
  para_time = MPI_Wtime() - para_time;
  if(my_rank == 0) printf("%lf,", para_time);

  res_root_p = Save_vector("parallel_y", local_y, m, local_m, my_rank, comm);

  free(local_A);
  free(local_x);
  free(local_y);
  MPI_Type_free(&block_col_mpi_t);

  // test serial on root
  if(my_rank == 0) {
#ifdef DEBUG
    printf("\n");
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        printf("%lf ", A_root[i*n+j]);
      }
      printf("\n");
    }
#endif

    if(run_serial == 1) {
      serial_time = MPI_Wtime(); 
      serial(n, A_root, vec_root, res_root);
      serial_time = MPI_Wtime() - serial_time;
      printf("%lf,", serial_time);
    }

#ifdef DEBUG
    printf("vec:\n");
    for(i = 0; i < n; i ++)
      printf("%lf ", vec_root[i]);
    printf("\n");

    printf("result serial:\n");
    for(i = 0; i < n; i ++)
      printf("%.10lf ", res_root[i]);
    printf("\n");

    printf("result parallel:\n");
    for(i = 0; i < n; i ++)
      printf("%.10lf ", res_root_p[i]);
    printf("\n");
#endif

    //FILE *file = fopen("serial_y", "wb");
    //fwrite(res_root, sizeof(double), n, file);
    //fclose(file);
  }
  //printf("Here\n");
  double para_l2_time = 0;
  para_l2_time = MPI_Wtime();
  int flag1;
  if(run_serial == 1)
    flag1 = verify_parallel(n, local_n, my_rank, res_root, res_root_p, comm);
  else
    flag1 = verify_parallel(n, local_n, my_rank, res_root_p, res_root_p, comm);
  para_l2_time = MPI_Wtime() - para_l2_time;

  double serial_l2_time = 0;
  if(run_serial == 1) {
    serial_l2_time = MPI_Wtime();
    int flag2 = verify_serial(n, res_root, res_root_p);
    serial_l2_time = MPI_Wtime() - serial_l2_time;
  }
  
  if(my_rank == 0) {
    if(flag1 == 1) printf("Success.\n");
    else printf("Failed.\n");

    FILE *file = fopen("log.txt", "aw");
    fprintf(file, "%lf,%lf,%lf,%lf,%d\n", para_time * 1000, serial_time * 1000, para_l2_time * 1000, serial_l2_time * 1000, flag1);
    close(file);
    
    free(A_root);
    free(vec_root);
    free(res_root);
  }

  MPI_Finalize();

  return 0;
} /* main */

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s n\n", argv[0]);
    return -1;
  }
  int n = atoi(argv[1]);
  int run_serial = atoi(argv[2]);

  run(n, run_serial);
}

void serial(int n, double* matrix, double* vector, double* result) {
  int i, j;

  for (i = 0; i < n; i++) {
    double temp = 0.0;
    for (j = 0; j < n; j++) {
      temp += matrix[i*n+j] * vector[j];
    }
    result[i] = temp;
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

int verify_parallel(int n, int local_n, int my_rank, double *vec1, double *vec2, MPI_Comm comm) {
  int i;
  double r = 0, ans = 0;

  double *local_vec1, *local_vec2;

  local_vec1 = (double*)malloc(local_n * sizeof(double));
  local_vec2 = (double*)malloc(local_n * sizeof(double));
  if(local_vec1 == NULL || local_vec2 == NULL)
    printf("Apply failed at verify parallel\n");

  MPI_Scatter(vec1, local_n, MPI_DOUBLE, local_vec1, local_n, MPI_DOUBLE, 0,
                comm);
  MPI_Scatter(vec2, local_n, MPI_DOUBLE, local_vec2, local_n, MPI_DOUBLE, 0,
                comm);
  //printf("ST %d\n", local_n);
  // compute local abs difference
  for(i = 0; i < local_n; i ++)
    r += (local_vec1[i] - local_vec2[i]) * (local_vec1[i] - local_vec2[i]);
  //printf("MID\n");
  //printf("%d: %lf ", my_rank, r);
  MPI_Reduce(&r, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  //printf("ED\n");
  free(local_vec1);
  free(local_vec2);
  
  if(ans < 0.0001) return 1;
  else {
    //printf("%d %lf\n", my_rank, ans);
    return 0;
  }
}

/*-------------------------------------------------------------------*/
void Check_for_error(int local_ok /* in */,
                     char fname[] /* in */,
                     char message[] /* in */,
                     MPI_Comm comm /* in */) {

  int ok;

  MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
  if (ok == 0) {
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
      fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
      fflush(stderr);
    }
    MPI_Finalize();
    exit(-1);
  }
} /* Check_for_error */

/*-------------------------------------------------------------------*/
void Get_dims(int* m_p /* out */,
              int* local_m_p /* out */,
              int* n_p /* out */,
              int* local_n_p /* out */,
              int my_rank /* in  */,
              int comm_sz /* in  */,
              MPI_Comm comm /* in  */) {

  int local_ok = 1;

  MPI_Bcast(m_p, 1, MPI_INT, 0, comm);

  *n_p = *m_p;
  if (*m_p <= 0 || *m_p % comm_sz != 0)
    local_ok = 0;
  Check_for_error(local_ok, "Get_dims",
                  "m and n must be positive and evenly divisible by comm_sz",
                  comm);

  *local_m_p = *m_p / comm_sz;
  *local_n_p = *n_p / comm_sz;
} /* Get_dims */

/*-------------------------------------------------------------------*/
void Allocate_arrays(double** local_A_pp /* out */,
                     double** local_x_pp /* out */,
                     double** local_y_pp /* out */,
                     int m /* in  */,
                     int local_m /* in  */,
                     int local_n /* in  */,
                     MPI_Comm comm /* in  */) {

  int local_ok = 1;

  *local_A_pp = malloc(m * local_n * sizeof(double));
  *local_x_pp = malloc(local_n * sizeof(double));
  *local_y_pp = malloc(local_m * sizeof(double));

  if (*local_A_pp == NULL || local_x_pp == NULL || local_y_pp == NULL)
    local_ok = 0;
  Check_for_error(local_ok, "Allocate_arrays", "Can't allocate local arrays",
                  comm);
} /* Allocate_arrays */

/*-------------------------------------------------------------------*/
void Build_derived_type(
    int m, int local_m, int n, int local_n, MPI_Datatype* block_col_mpi_t_p) {
  MPI_Datatype vect_mpi_t;

  /* m blocks each containing local_n elements */
  /* The start of each block is n doubles beyond the preceding block */
  MPI_Type_vector(m, local_n, n, MPI_DOUBLE, &vect_mpi_t);

  /* Resize the new type so that it has the extent of local_n doubles */
  MPI_Type_create_resized(vect_mpi_t, 0, local_n * sizeof(double),
                          block_col_mpi_t_p);
  MPI_Type_commit(block_col_mpi_t_p);
} /* Build_derived_type */

void gen_matrix(int m, int n, double* matrix) {
  int i, j;
  srand((unsigned)time(0));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      matrix[i * n + j] = -1 + rand() * 1.0 / RAND_MAX * 2;
}

void gen_vector(int n, double* vector) {
  int i;
  for (i = 0; i < n; i++) {
    vector[i] = -1 + rand() * 1.0 / RAND_MAX * 2;
  }
}

/*-------------------------------------------------------------------*/
double* Read_matrix(char prompt[] /* in  */,
                 double local_A[] /* out */,
                 int m /* in  */,
                 int local_n /* in  */,
                 int n /* in  */,
                 MPI_Datatype block_col_mpi_t /* in  */,
                 int my_rank /* in  */,
                 MPI_Comm comm /* in  */) {
  double* A = NULL;
  int local_ok = 1;
  int i, j;

  if (my_rank == 0) {
    A = malloc(m * n * sizeof(double));
    if (A == NULL)
      local_ok = 0;
    Check_for_error(local_ok, "Read_matrix", "Can't allocate temporary matrix",
                    comm);

    gen_matrix(m, n, A);

    MPI_Scatter(A, 1, block_col_mpi_t, local_A, m * local_n, MPI_DOUBLE, 0,
                comm);
    return A;
  } else {
    Check_for_error(local_ok, "Read_matrix", "Can't allocate temporary matrix",
                    comm);
    MPI_Scatter(A, 1, block_col_mpi_t, local_A, m * local_n, MPI_DOUBLE, 0,
                comm);
    return NULL;
  }
} /* Read_matrix */

/*-------------------------------------------------------------------*/
void Print_matrix(char title[],
                  double local_A[],
                  int m,
                  int local_n,
                  int n,
                  MPI_Datatype block_col_mpi_t,
                  int my_rank,
                  MPI_Comm comm) {
  double* A = NULL;
  int local_ok = 1;
  int i, j;

  if (my_rank == 0) {
    A = malloc(m * n * sizeof(double));
    if (A == NULL)
      local_ok = 0;
    Check_for_error(local_ok, "Print_matrix", "Can't allocate temporary matrix",
                    comm);

    MPI_Gather(local_A, m * local_n, MPI_DOUBLE, A, 1, block_col_mpi_t, 0,
               comm);

    printf("The matrix %s\n", title);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
        printf("%.2f ", A[i * n + j]);
      printf("\n");
    }

    free(A);
  } else {
    Check_for_error(local_ok, "Print_matrix", "Can't allocate temporary matrix",
                    comm);
    MPI_Gather(local_A, m * local_n, MPI_DOUBLE, A, 1, block_col_mpi_t, 0,
               comm);
  }
} /* Print_matrix */

/*-------------------------------------------------------------------*/
double* Read_vector(char prompt[] /* in  */,
                 double local_vec[] /* out */,
                 int n /* in  */,
                 int local_n /* in  */,
                 int my_rank /* in  */,
                 MPI_Comm comm /* in  */) {
  double* vec = NULL;
  int i, local_ok = 1;

  if (my_rank == 0) {
    vec = malloc(n * sizeof(double));
    if (vec == NULL)
      local_ok = 0;
    Check_for_error(local_ok, "Read_vector", "Can't allocate temporary vector",
                    comm);

    gen_vector(n, vec);

    MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0,
                comm);
    return vec;
  } else {
    Check_for_error(local_ok, "Read_vector", "Can't allocate temporary vector",
                    comm);
    MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, 0,
                comm);
    return NULL;
  }
} /* Read_vector */

/*-------------------------------------------------------------------*/
void Mat_vect_mult(double local_A[] /* in  */,
                   double local_x[] /* in  */,
                   double local_y[] /* out */,
                   int local_m /* in  */,
                   int m /* in  */,
                   int n /* in  */,
                   int local_n /* in  */,
                   int comm_sz,
                   MPI_Comm comm /* in  */) {

  double* my_y;
  int* recv_counts;
  int i, loc_j;
  int local_ok = 1;

  recv_counts = malloc(comm_sz * sizeof(int));
  my_y = malloc(n * sizeof(double));
  if (recv_counts == NULL || my_y == NULL)
    local_ok = 0;
  Check_for_error(local_ok, "Mat_vect_mult", "Can't allocate temporary arrays",
                  comm);

  for (i = 0; i < m; i++) {
    my_y[i] = 0.0;
    for (loc_j = 0; loc_j < local_n; loc_j++)
      my_y[i] += local_A[i * local_n + loc_j] * local_x[loc_j];
  }

  for (i = 0; i < comm_sz; i++) {
    recv_counts[i] = local_m;
  }

  MPI_Reduce_scatter(my_y, local_y, recv_counts, MPI_DOUBLE, MPI_SUM, comm);

  free(my_y);
} /* Mat_vect_mult */

/*-------------------------------------------------------------------*/
void Print_vector(char title[] /* in */,
                  double local_vec[] /* in */,
                  int n /* in */,
                  int local_n /* in */,
                  int my_rank /* in */,
                  MPI_Comm comm /* in */) {
  double* vec = NULL;
  int i, local_ok = 1;

  if (my_rank == 0) {
    vec = malloc(n * sizeof(double));
    if (vec == NULL)
      local_ok = 0;
    Check_for_error(local_ok, "Print_vector", "Can't allocate temporary vector",
                    comm);
    MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0,
               comm);
    printf("\nThe vector %s\n", title);
    for (i = 0; i < n; i++)
      printf("%f ", vec[i]);
    printf("\n");
    free(vec);
  } else {
    Check_for_error(local_ok, "Print_vector", "Can't allocate temporary vector",
                    comm);

    MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0,
               comm);
  }
} /* Print_vector */

double* Save_vector(char title[] /* in */,
                 double local_vec[] /* in */,
                 int n /* in */,
                 int local_n /* in */,
                 int my_rank /* in */,
                 MPI_Comm comm /* in */) {
  double* vec = NULL;
  int i, local_ok = 1;

  if (my_rank == 0) {
    vec = malloc(n * sizeof(double));
    if (vec == NULL)
      local_ok = 0;
    Check_for_error(local_ok, "Print_vector", "Can't allocate temporary vector",
                    comm);
    MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0,
               comm);

    //FILE *file = fopen((const char *)title, "wb");
    //fwrite(vec, sizeof(double), n, file);
    //fclose(file);

    return vec;
  } else {
    Check_for_error(local_ok, "Print_vector", "Can't allocate temporary vector",
                    comm);

    MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0,
               comm);
    return NULL;
  }
}