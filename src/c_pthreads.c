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
    int bi;           
    int bj;           
    int bk;           
    Matrix* A;        
    Matrix* L;        
    int block_size;   
    int start_row;    
    int end_row;      
} thread_block_data;

const char* MATRIX_DEBUG = "matrix_debug.txt";

pthread_t* threads = NULL;
thread_block_data* thread_args = NULL;
int NUM_THREADS = 2;



static Matrix create_matrix_full(size_t rows, size_t cols)
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

static inline void block_indices(int nb, int *bi, int *bj, int idx) {
    int count = 0;
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j <= i; j++) {
            if(count == idx) {
                *bi = i;
                *bj = j;
                return;
            }
            count++;
        }
    }
}

static inline int owner_block(int bi, int bj, int num_threads) {
    int block_id = bi * (bi + 1) / 2 + bj;
    return block_id % num_threads;
}

void factorize_block(double** A, double** L, int b) {
    for(int i = 0; i < b; i++) {
        for(int j = 0; j < b; j++) {
            L[i][j] = 0.0;
        }
    }
    for(int i = 0; i < b; i++) {
        for(int j = 0; j <= i; j++) {
            L[i][j] = A[i][j];
        }
    }

    for(int k = 0; k < b; k++) {
        double sum = 0.0;
        for(int p = 0; p < k; p++) {
            sum += L[k][p] * L[k][p];
        }

        L[k][k] = sqrt(L[k][k] - sum);

        for(int i = k + 1; i < b; i++) {
            double sum2 = 0.0;
            for(int p = 0; p < k; p++) {
                sum2 += L[i][p] * L[k][p];
            }
            L[i][k] = (L[i][k] - sum2) / L[k][k];
        }
    }
}


void* solve_triangular_block(void* arg)
{
    thread_block_data* data = (thread_block_data*) arg;

    int bi = data->bi;
    int bk = data->bk;
    int b  = data->block_size;

    Matrix* A = data->A;
    Matrix* L = data->L;

    int row_start = data->start_row;
    int row_end   = data->end_row;

    int Ai = bi * b;
    int Ak = bk * b;

    for(int i = row_start; i < row_end; i++)
    {
        for(int j = 0; j < b; j++)
        {
            double s = 0.0;
            for(int p = 0; p < j; p++)
            {
                s += L->data[(Ai + i)][(bk * b + p)] *
                     L->data[(bk * b + j)][(bk * b + p)];
            }

            L->data[(Ai + i)][(bk * b + j)] =
                (A->data[(Ai + i)][(bk * b + j)] - s)
                / L->data[(bk * b + j)][(bk * b + j)];
        }
    }

    pthread_exit(NULL);
}

void* update_trailing_block(void* arg)
{
    thread_block_data* data = (thread_block_data*)arg;

    int bi = data->bi;
    int bj = data->bj;
    int bk = data->bk;
    int b  = data->block_size;

    Matrix* A = data->A;
    Matrix* L = data->L;

    // Смещение блока в глобальной матрице
    int Ai = bi * b;
    int Aj = bj * b;
    int Ak = bk * b;

    int row_start = data->start_row;
    int row_end   = data->end_row;

    for (int i = row_start; i < row_end; i++)
    {
        // Для каждой строки i текущего блока
        for (int j = 0; j < b; j++)
        {
            double sum = 0.0;

            // Вычисляем сумму L[i,k] * L[j,k] для k = 0..b-1
            for (int p = 0; p < b; p++)
            {
                sum += L->data[(Ai + i)][(Ak + p)] *
                       L->data[(Aj + j)][(Ak + p)];
            }

            // Вычитаем вклад из блока A
            A->data[(Ai + i)][(Aj + j)] -= sum;
        }
    }

    pthread_exit(NULL);
}


void print_flat_matrix(double** A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f ", A[i][j]);
        }
        printf("\n");
    }
}

Matrix cholesky_blocked_pthreads(Matrix* A, int n, int b)
{
    assert(A->cols == A->rows);
    assert(A->cols == n);

    Matrix L = create_matrix_full(n, n);

    int nb = (n + b - 1) / b;

    pthread_t threads[NUM_THREADS];
    thread_block_data thread_args[NUM_THREADS];
    bool thread_created[NUM_THREADS];

    for (int k = 0; k < nb; k++)
    {
        {
            int bi = k, bj = k;
            int rows = ((bi * b + b) > n ? (n - bi * b) : b);

            double** Ablock = allocate_block(rows);
            double** Lblock = allocate_block(rows);

            for (int ii = 0; ii < rows; ii++)
                for (int jj = 0; jj < rows; jj++)
                    Ablock[ii][jj] = A->data[bi*b + ii][bj*b + jj];
            // print_flat_matrix(Ablock, rows);
            

            factorize_block(Ablock, Lblock, rows);
            // printf("L block %i\n", k);
            // print_flat_matrix(Lblock, rows);

            for (int ii = 0; ii < rows; ii++)
                for (int jj = 0; jj < rows; jj++)
                    L.data[bi*b + ii][bj*b + jj] = Lblock[ii][jj];

            free_block(Ablock, rows);
            free_block(Lblock, rows);
        }

        for (int t = 0; t < NUM_THREADS; t++)
            thread_created[t] = false;

        for (int i = k + 1; i < nb; i++)
        {
            int rows = ((i * b + b) > n ? (n - i*b) : b);

            int chunk = rows / NUM_THREADS;
            int rem   = rows % NUM_THREADS;
            int start = 0;
            int thread_index = 0;

            for (int t = 0; t < NUM_THREADS; t++)
            {
                int cnt = chunk + (t < rem ? 1 : 0);
                if (cnt <= 0) continue;

                thread_args[thread_index] = (thread_block_data){
                    .bi = i, .bj = 0, .bk = k,
                    .A  = A, .L  = &L,
                    .block_size = b,
                    .start_row  = start,
                    .end_row    = start + cnt
                };
                pthread_create(&threads[thread_index], NULL,
                               solve_triangular_block,
                               &thread_args[thread_index]);
                thread_created[thread_index++] = true;
                start += cnt;
            }

            for (int t = 0; t < thread_index; t++)
                if (thread_created[t])
                    pthread_join(threads[t], NULL);
        }

        for (int i = k + 1; i < nb; i++)
        {
            for (int j = k + 1; j <= i; j++)
            {
                for (int t = 0; t < NUM_THREADS; t++)
                    thread_created[t] = false;

                int rows = ((i*b + b) > n ? (n - i*b) : b);

                int chunk = rows / NUM_THREADS;
                int rem   = rows % NUM_THREADS;
                int start = 0;
                int thread_index = 0;

                for (int t = 0; t < NUM_THREADS; t++)
                {
                    int cnt = chunk + (t < rem ? 1 : 0);
                    if (cnt <= 0) continue;

                    thread_args[thread_index] = (thread_block_data){
                        .bi = i, .bj = j, .bk = k,
                        .A  = A, .L  = &L,
                        .block_size = b,
                        .start_row  = start,
                        .end_row    = start + cnt
                    };
                    pthread_create(&threads[thread_index], NULL,
                                   update_trailing_block,
                                   &thread_args[thread_index]);
                    thread_created[thread_index++] = true;
                    start += cnt;
                }

                for (int t = 0; t < thread_index; t++)
                    if (thread_created[t])
                        pthread_join(threads[t], NULL);
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
    *relres = second_norm(&r) / r0norm;

    free_vector(z);
    free_vector(r);
    free_vector(p);
    free_vector(current_);
    free_vector(newR);
    free_vector(newZ);
    free_vector(q);
    return x;
}

void pcgCholBlocks(size_t n,
             const char *b_filename,
             double eps)
{
    printf("=== General SPD PCG + Blocked Cholesky ===\n");
    printf("Matrix size: %zu x %zu\n", n, n);

    Matrix A = generate_general_spd(n);
    // printf("Matrix A\n");
    // print_matrix(&A);

    Vector x_true = generate_true_x(n);

    Vector b = read_vector_from_file(b_filename);
    if ((size_t)b.size != n) {
        fprintf(stderr, "Размер b не совпадает с размером матрицы\n");
        exit(EXIT_FAILURE);
    }

    printf("Blocked Cholesky decomposition...\n");

    int block_size = 64;

    Matrix L = cholesky_blocked_pthreads(&A, (int)n, block_size);
    // printf("Matrix L\n");
    // print_matrix(&L);
    // Matrix trueL = read_matrix_from_file(MATRIX_DEBUG);
    // printf("Max diff between true and block L: %.15f\n", max_difference_between_matrices(&L, &trueL));
    Matrix Lt = transpose(L);

    Vector x0 = create_vector(n);
    Matrix Anew = generate_general_spd(n);

    double relres = 0.0;
    int iter = 0;
    Vector x_sol = pcgPreconditioned(
        &Anew,
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
    free_matrix(&Anew);
    free_matrix(&L);
    free_matrix(&Lt);
}




// True main
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    NUM_THREADS = atoi(argv[1]);
    if (NUM_THREADS <= 0)
    {
        fprintf(stderr, "Error: number of threads must be positive\n");
        return EXIT_FAILURE;
    }

    threads = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    thread_args = (thread_block_data*)malloc(NUM_THREADS * sizeof(thread_block_data));

    if (!threads || !thread_args)
    {
        fprintf(stderr, "Error: failed to allocate thread structures\n");
        free(threads);
        free(thread_args);
        return EXIT_FAILURE;
    }

    size_t n = 4096;
    Matrix A = generate_general_spd(n);   
    // print_matrix(&A);     
    Vector x = generate_true_x(n);             
    const char *fname = "b_vector.txt";

    double maxdiff = multiply_save_read_compare(&A, &x, fname);

    printf("Max difference = %.15f\n", maxdiff);
    const char* b_filename = "b_vector.txt";
    double eps = 1e-5;

    pcgCholBlocks(n, b_filename, eps);

    free(threads);
    free(thread_args);

    return 0;
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

//     size_t n = 15;
//     const char *bfile = "b_vector.txt";
//     double eps = 1e-8;

//     pcgChol(n, bfile, eps);

//     free(threads);
//     free(thread_args);

//     return 0;
// }





// int main(int argc, char** argv) {
//     size_t n = 4096;
//     Matrix A = generate_general_spd(n);   
//     // print_matrix(&A);     
//     Vector x = generate_true_x(n);        
//     // printf("True x:\n");
//     // print_vector(&x);
//     const char *fname = "b_vector.txt";

//     double maxdiff = multiply_save_read_compare(&A, &x, fname);

//     printf("Max difference = %.15f\n", maxdiff);

//     free_matrix(&A);
//     free_vector(x);
// }
