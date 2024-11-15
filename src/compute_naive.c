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

  for (int i = 0; i < output_rows; i++) {
    for (int j = 0; j < output_cols; j++) {
        int sum = 0;
        for (int m = 0; i < b_rows; m++) {
          for (int n = 0; j < b_cols; n++) {
            int a_val = a_matrix->data[(i+m) * a_cols + (j + n)];
            int b_val = b_matrix->data[m * b_cols + n];
            sum += a_val * b_val;
          }
        }
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
