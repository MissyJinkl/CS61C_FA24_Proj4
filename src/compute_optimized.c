#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
  int a_rows = a_matrix->rows;
  int a_cols = a_matrix->cols;
  int b_rows = b_matrix->rows;
  int b_cols = b_matrix->cols;
  int output_rows = a_rows - b_rows + 1;
  int output_cols = a_cols - b_cols + 1;

  *output_matrix = malloc(sizeof(matrix_t));
  if (*output_matrix == NULL) return -1;

  (*output_matrix)->rows = output_rows;
  (*output_matrix)->cols = output_cols;
  (*output_matrix)->data = malloc(output_rows * output_cols * sizeof(int));
  if ((*output_matrix)->data == NULL) return -1;

  __m256i reverse_indices = _mm256_setr_epi32(7,6,5,4,3,2,1,0);

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < output_rows; i++) {
    for (int j = 0; j < output_cols; j++) {
        __m256i sum_vec = _mm256_setzero_si256();
        int sum = 0;

        for (int m = 0; m < b_rows; m++) {
          for (int n = 0; n < b_cols/8*8; n+=8) {
            int a_index = (i+m) * a_cols + (j + n);
            int b_index = (b_rows - 1 - m) * b_cols + (b_cols - 1 - n) - 7;
            __m256i a_val = _mm256_loadu_si256((__m256i *)&(a_matrix->data[a_index]));
            __m256i b_val = _mm256_loadu_si256((__m256i *)&(b_matrix->data[b_index]));
            b_val = _mm256_permutevar8x32_epi32(b_val, reverse_indices);
            __m256i product = _mm256_mullo_epi32(a_val, b_val);
            sum_vec = _mm256_add_epi32(sum_vec, product);
          }
          for (int n = b_cols/8*8; n < b_cols; n++) {
            int a_val = a_matrix->data[(i+m) * a_cols + (j + n)];
            int b_val = b_matrix->data[(b_rows - 1 - m) * b_cols + (b_cols - 1 - n)];
            sum += a_val * b_val;
          }
        }
        
        int tmp[8];
        _mm256_storeu_si256((__m256i *) tmp, sum_vec);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

        //tail cases
      /*#pragma omp parallel
        {
          int private_sum = 0;
          #pragma omp for
          for (int m = 0; m < b_rows; m++) {
            for (int n = b_cols/8*8; n < b_cols; n++) {
              int a_val = a_matrix->data[(i+m) * a_cols + (j + n)];
              int b_val = b_matrix->data[(b_rows - 1 - m) * b_cols + (b_cols - 1 - n)];
              private_sum += a_val * b_val;
            }
          }
          #pragma omp critical
          sum += private_sum;
        }
        
        for (int m = 0; m < b_rows; m++) {
            for (int n = b_cols/8*8; n < b_cols; n++) {
              int a_val = a_matrix->data[(i+m) * a_cols + (j + n)];
              int b_val = b_matrix->data[(b_rows - 1 - m) * b_cols + (b_cols - 1 - n)];
              sum += a_val * b_val;
            }
          }
        */
        (*output_matrix)->data[i * output_cols + j] = sum;
    }
  }

  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
