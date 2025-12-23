#include "../algebra.h"
#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct
{
    int j;
    Matrix* L;
    Matrix* A;
    int start_i;
    int end_i;
} thread_data;

pthread_t* threads = NULL;
thread_data* thread_args = NULL;
int NUM_THREADS = 2;

void* compute_column(void* arg)
{
    thread_data* data = (thread_data*)arg;
    int j = data->j;
    Matrix* L = data->L;
    Matrix* A = data->A;

    for(int i = data->start_i; i < data->end_i; i++)
    {
        double s = 0.0;
        for(int k = 0; k < j; k++)
        {
            s += L->data[i][k] * L->data[j][k];
        }
        L->data[i][j] = (1.0 / L->data[j][j]) * (A->data[i][j] - s);
    }
    pthread_exit(NULL);
}

static Matrix create_matrix_openmp(size_t rows, size_t cols)
{
    Matrix m = {rows, cols, NULL, NULL};

    m.data = (double**)malloc(rows * sizeof *m.data);
    if(!m.data)
        die("malloc data");

    size_t total;
    if(mul_overflow_size_t(rows, cols, &total))
    {
        free(m.data);
        fprintf(stderr, "Размер матрицы слишком большой (переполнение).\n");
        exit(EXIT_FAILURE);
    }

    m.buf = (double*)calloc(total, sizeof *m.buf);
    if(!m.buf)
    {
        free(m.data);
        die("calloc buf");
    }

    // #pragma omp parallel for
    for(size_t i = 0; i < rows; ++i)
    {
        m.data[i] = m.buf + i * cols;
    }
    return m;
}


Matrix generate_general_spd(size_t n)
{
    Matrix A = create_matrix(n, n);

    for(size_t i = 0; i < n; ++i)
    {
        // главная диагональ (4 -> 6)
        double diag = 6.0 + sin((double)i) + cos((double)i * (double)i);
        A.data[i][i] = diag;

        // соседние диагонали
        if(i > 0)
        {
            double val1 = -1.0 + 0.5 * sin((double)i);
            A.data[i][i-1] = val1;
            A.data[i-1][i] = val1;
        }

        // первая и пятая диагонали
        if(i > 1)
        {
            double val2 = -0.5 + 0.2 * cos((double)i);
            A.data[i][i-2] = val2;
            A.data[i-2][i] = val2;
        }
    }

    return A;
}

Vector generate_true_x(size_t n)
{
    Vector x_star = create_vector(n);
    for(size_t i = 0; i < n; ++i)
    {
        x_star.data[i] = sin((double)i)
                       + cos(sqrt((double)i))
                       + 0.3 * (double)i;
    }
    return x_star;
}

void save_vector_to_file(const char *filename, Vector *v)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        perror("save_vector_to_file: fopen");
        return;
    }

    fprintf(f, "%d\n", v->size);

    for (int i = 0; i < v->size; ++i)
    {
        fprintf(f, "%.15g\n", v->data[i]);
    }
    fclose(f);
}

Vector read_vector_from_file(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("read_vector_from_file: fopen");
        exit(EXIT_FAILURE);
    }

    int size;
    if (fscanf(f, "%d", &size) != 1)
    {
        fprintf(stderr, "read_vector_from_file: invalid format\n");
        exit(EXIT_FAILURE);
    }

    Vector v = create_vector(size);
    for (int i = 0; i < size; ++i)
    {
        if (fscanf(f, "%lf", &v.data[i]) != 1)
        {
            fprintf(stderr, "read_vector_from_file: read error at %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(f);
    return v;
}

void multiply_and_save(Matrix *A, Vector *x, const char *out_filename)
{
    if (A->cols != (size_t)x->size)
    {
        fprintf(stderr, "multiply_and_save: dimension mismatch\n");
        return;
    }

    Vector b = create_vector((int)A->rows);

    for (size_t i = 0; i < A->rows; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; ++j)
        {
            sum += A->data[i][j] * x->data[j];
        }
        b.data[i] = sum;
    }

    // Запись в файл
    save_vector_to_file(out_filename, &b);

    free_vector(b);
}

double max_difference_between_vectors(Vector *a, Vector *b)
{
    if (a->size != b->size)
    {
        fprintf(stderr, "max_difference_between_vectors: size mismatch\n");
        return -1.0;
    }

    double max_diff = 0.0;
    for (int i = 0; i < a->size; ++i)
    {
        double diff = fabs(a->data[i] - b->data[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

double multiply_save_read_compare(Matrix *A, Vector *x,
                                  const char *filename_b)
{
    multiply_and_save(A, x, filename_b);

    Vector b_from_file = read_vector_from_file(filename_b);

    Vector b_original = create_vector((int)A->rows);
    for (size_t i = 0; i < A->rows; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; ++j)
            sum += A->data[i][j] * x->data[j];
        b_original.data[i] = sum;
    }

    double max_diff = max_difference_between_vectors(&b_original, &b_from_file);

    free_vector(b_from_file);
    free_vector(b_original);

    return max_diff;
}

Matrix cholesky(Matrix* A, int n)
{
    Matrix L = create_matrix_openmp(n, n);
    assert(A->cols == A->rows);
    assert(A->cols == n);

    pthread_t threads[NUM_THREADS];
    thread_data thread_args[NUM_THREADS];

    bool thread_created[NUM_THREADS]; 

    for(int j = 0; j < n; j++)
    {
        for(int t = 0; t < NUM_THREADS; t++)
        {
            thread_created[t] = false;
        }

        double s = 0.0;
        for(int k = 0; k < j; k++)
        {
            s += L.data[j][k] * L.data[j][k];
        }
        L.data[j][j] = sqrt(A->data[j][j] - s);

        int rows_remaining = n - j - 1;
        if(rows_remaining > 0)
        {
            int chunk_size = rows_remaining / NUM_THREADS;
            int extra = rows_remaining % NUM_THREADS;
            int current_start = j + 1;

            for(int t = 0; t < NUM_THREADS; t++)
            {
                int thread_rows = chunk_size + (t < extra ? 1 : 0);
                if(thread_rows > 0)
                {
                    thread_args[t] =
                        (thread_data){.j = j,
                                      .L = &L,
                                      .A = A,
                                      .start_i = current_start,
                                      .end_i = current_start + thread_rows};
                    if(pthread_create(&threads[t], NULL, compute_column,
                                      &thread_args[t]) == 0)
                    {
                        thread_created[t] = true;
                    }
                    current_start += thread_rows;
                }
            }
            for(int t = 0; t < NUM_THREADS; t++)
            {
                if(thread_created[t])
                {
                    pthread_join(threads[t], NULL);
                }
            }
        }
    }
    return L;
}
Vector solve_gauss_reverse(Matrix* U, Vector* b)
{
    int n = U->rows;
    Vector x = create_vector(n);

    for(int i = n - 1; i >= 0; i--)
    {
        double sum = b->data[i];

        for(int j = i + 1; j < n; j++)
        {
            sum -= U->data[i][j] * x.data[j];
        }

        if(fabs(U->data[i][i]) < 1e-12)
        {
            printf("Ошибка: нулевой диагональный элемент в строке %d!\n", i);
            free_vector(x);
            exit(EXIT_FAILURE);
        }

        x.data[i] = sum / U->data[i][i];
    }

    return x;
}

Vector solve_gauss_forward(Matrix* L, Vector* b)
{
    int n = L->rows;
    Vector x = create_vector(n);

    for(int i = 0; i < n; i++)
    {
        double sum = b->data[i];

        for(int j = 0; j < i; j++)
        {
            sum -= L->data[i][j] * x.data[j];
        }

        if(fabs(L->data[i][i]) < 1e-12)
        {
            printf("Ошибка: нулевой диагональный элемент в строке %d!\n", i);
            free_vector(x);
            exit(EXIT_FAILURE);
        }

        x.data[i] = sum / L->data[i][i];
    }

    return x;
}

Vector solve_gauss(Matrix* L, Matrix* U, Vector* b)
{
    if(L->rows != U->rows || L->rows != b->size)
    {
        printf("Ошибка: несовместимые размеры в solve_gauss!\n");
        exit(EXIT_FAILURE);
    }

    Vector y = solve_gauss_forward(L, b);

    Vector x = solve_gauss_reverse(U, &y);

    free_vector(y);

    return x;
}

Vector pcgPreconditioned(Matrix* A, Vector* b, Vector* xs, double err,
                         double* relres, int* iter, Matrix* P1, Matrix* P2)
{
    int k = 0;
    size_t n = A->cols;
    Vector x = copy_vector(xs);
    Vector r = residue(b, A, &x);

    Vector z = solve_gauss(P1, P2, &r);

    Vector p = copy_vector(&z);

    double r0norm = second_norm(&r);

    Vector current_ = create_vector(n);
    Vector newR = create_vector(n);
    Vector newZ = create_vector(n);
    Vector q = create_vector(n);
    while(second_norm(&r) / r0norm > err && k < 1000)
    {
        ++k;

        free_vector(q);
        q = matrix_vector_mult(A, &p);

        double pq = dot_product(&p, &q);

        double a = aKbyPQ(&z, &q, pq);

        free_vector(current_);
        current_ = scalar_vector_mult(&p, a);

        add_vector_self(&x, &current_);

        free_vector(current_);
        current_ = scalar_vector_mult(&q, a);

        free_vector(newR);
        newR = sub_vector(&r, &current_);

        free_vector(newZ);
        newZ = solve_gauss(P1, P2, &newR);

        double b = dot_product(&newZ, &newR) / dot_product(&z, &r);

        free_vector(z);
        z = newZ;

        free_vector(r);
        r = newR;

        free_vector(current_);
        current_ = scalar_vector_mult(&p, b);

        free_vector(p);
        p = add_vector(&r, &current_);
    }
    *iter = k;

    free_vector(z);
    free_vector(r);
    free_vector(p);
    free_vector(current_);
    free_vector(newR);
    free_vector(newZ);
    free_vector(q);
    return x;
}

void pcgChol(size_t n,
                             const char *b_filename,
                             double eps)
{
    printf("=== General SPD PCG + Cholesky ===\n");
    printf("Matrix size: %zu x %zu\n", n, n);

    Matrix A = generate_general_spd(n);

    Vector x_true = generate_true_x(n);

    Vector b = read_vector_from_file(b_filename);

    if ((size_t)b.size != n)
    {
        fprintf(stderr, "Размер b не совпадает с размером матрицы\n");
        exit(EXIT_FAILURE);
    }

    printf("Cholesky decomposition...\n");
    Matrix L = cholesky(&A, (int)n);
    Matrix Lt = transpose(L);

    Vector x0 = create_vector(n);

    double relres = 0.0;
    int iter = 0;

    Vector x_sol = pcgPreconditioned(
        &A,
        &b,
        &x0,
        eps,
        &relres,
        &iter,
        &L,
        &Lt
    );

    double max_err = max_difference_between_vectors(&x_true, &x_sol);

    printf("PCG finished\n");
    printf("Iterations: %d\n", iter);
    printf("Relative residual: %.3e\n", relres);
    printf("Max |x - x*|: %.3e\n", max_err);

    free_vector(x_sol);
    free_vector(x0);
    free_vector(b);
    free_vector(x_true);
    free_matrix(&A);
    free_matrix(&L);
    free_matrix(&Lt);
}





// int main(int argc, char** argv)
// {
//     if(argc < 2)
//     {
//         printf("Usage: %s <num_threads>\n", argv[0]);
//         return 1;
//     }

//     NUM_THREADS = atoi(argv[1]);
//     if(NUM_THREADS <= 0)
//     {
//         printf("Error: Number of threads must be positive\n");
//         return 1;
//     }

//     threads = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
//     thread_args = (thread_data*)malloc(NUM_THREADS * sizeof(thread_data));

//     if(!threads || !thread_args)
//     {
//         fprintf(stderr, "Error: memory allocation failed\n");
//         free(threads);
//         free(thread_args);
//         return 1;
//     }

//     size_t n = 4096;
//     const char *bfile = "b_vector.txt";
//     double eps = 1e-8;

//     pcgChol(n, bfile, eps);

//     free(threads);
//     free(thread_args);

//     return 0;
// }



int main(int argc, char** argv) {
    size_t n = 4096;
    Matrix A = generate_general_spd(n);   
    // print_matrix(&A);     
    Vector x = generate_true_x(n);             
    const char *fname = "b_vector.txt";

    double maxdiff = multiply_save_read_compare(&A, &x, fname);

    printf("Max difference = %.15f\n", maxdiff);

    free_matrix(&A);
    free_vector(x);
}
