#include "matrix/matrix.h"
#include "utils/utils.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#include <mpi.h>

typedef struct
{
    size_t rows;
    size_t cols;
    double** data;
    double* buf;
    MPI_Win win_buf; // MPI window for buffer
} Matrix_MPI;

static Matrix_MPI create_shared_matrix(size_t rows, size_t cols, MPI_Comm comm)
{
    Matrix_MPI m;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    m.rows = rows;
    m.cols = cols;
    m.data = NULL;
    m.buf = NULL;
    m.win_buf = MPI_WIN_NULL;

    size_t total = rows * cols;
    MPI_Aint buf_size = total * sizeof(double);

    // Get the base address of rank 0's window
    MPI_Aint sz;
    int disp;
    double* base_ptr;
    // // Allocate shared memory window
    // MPI_Win_allocate_shared(buf_size, sizeof(double), MPI_INFO_NULL, comm,
    //                         &m.buf, &m.win_buf);

    // // Query rank 0's address to use as the common base
    // MPI_Win_shared_query(m.win_buf, 0, &sz, &disp, &base_ptr);

    MPI_Aint mysize = (rank == 0) ? buf_size : 0;
    MPI_Win_allocate_shared(mysize, sizeof(double), MPI_INFO_NULL, comm, &m.buf,
                            &m.win_buf);

    MPI_Win_shared_query(m.win_buf, 0, &sz, &disp, &base_ptr);

    // Use the common base pointer for all processes
    double* common_buf = base_ptr;

    // Allocate local row pointers
    m.data = (double**)malloc(rows * sizeof(double*));

    // Set up row pointers using the COMMON base address
    for(size_t i = 0; i < rows; i++)
    {
        m.data[i] = common_buf + i * cols;
    }

    // Initialize only once (by rank 0)
    MPI_Win_fence(0, m.win_buf);
    if(rank == 0)
    {
        for(size_t i = 0; i < total; i++)
        {
            common_buf[i] = 0.0;
        }
    }
    MPI_Win_fence(0, m.win_buf);

    return m;
}
// Function to free shared matrix
static void free_shared_matrix(Matrix_MPI* mat, MPI_Comm comm)
{
    if(mat->data)
    {
        free(mat->data);
        mat->data = NULL;
    }

    if(mat->win_buf != MPI_WIN_NULL)
    {
        MPI_Win_free(&mat->win_buf);
    }

    mat->buf = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

Matrix_MPI generate_five_diag_mpi(size_t xn, size_t yn, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t n, nn;
    if(mul_overflow_size_t(xn, yn, &n) || mul_overflow_size_t(n, n, &nn))
    {
        if(rank == 0)
        {
            fprintf(stderr, "Слишком большая сетка (переполнение).\n");
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Create shared matrix
    Matrix_MPI A = create_shared_matrix(n, n, comm);

    MPI_Win_fence(0, A.win_buf);

    // Distribute rows among processes
    size_t rows_per_proc = n / size;
    size_t remainder = n % size;

    size_t start_row =
        rank * rows_per_proc + (rank < remainder ? rank : remainder);
    size_t end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

    // Each process fills its assigned rows
    // for(size_t i = start_row; i < end_row; ++i)
    // {
    //     // центр
    //     A.data[i][i] = -4.0;

    //     // вверх (i - xn)
    //     if(i >= xn)
    //     {
    //         A.data[i][i - xn] = 1.0;
    //     }
    //     // вниз (i + xn)
    //     if(i + xn < n)
    //     {
    //         A.data[i][i + xn] = 1.0;
    //     }
    //     // влево (i - 1) — не на левом краю строки
    //     if((i % xn) != 0)
    //     {
    //         A.data[i][i - 1] = 1.0;
    //     }
    //     // вправо (i + 1) — не на правом краю строки
    //     if((i % xn) != xn - 1)
    //     {
    //         A.data[i][i + 1] = 1.0;
    //     }
    // }
    for(size_t i = start_row; i < end_row; ++i)
    {
        // центр
        A.data[i][i] = 4.0;

        // вверх (i - xn)
        if(i >= xn)
        {
            A.data[i][i - xn] = -1.0;
        }
        // вниз (i + xn)
        if(i + xn < n)
        {
            A.data[i][i + xn] = -1.0;
        }
        // влево (i - 1) — не на левом краю строки
        if((i % xn) != 0)
        {
            A.data[i][i - 1] = -1.0;
        }
        // вправо (i + 1) — не на правом краю строки
        if((i % xn) != xn - 1)
        {
            A.data[i][i + 1] = -1.0;
        }
    }

    // Synchronize to ensure all processes have finished writing
    MPI_Win_fence(0, A.win_buf);

    // printf("process with rank: %i ended matrix creation", rank);
    return A;
}

Matrix_MPI generate_general_spd_mpi(size_t n, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Matrix_MPI A = create_shared_matrix(n, n, comm);

    MPI_Win_fence(0, A.win_buf);

    size_t rows_per_proc = n / size;
    size_t remainder = n % size;

    size_t start_row =
        rank * rows_per_proc + (rank < remainder ? rank : remainder);
    size_t end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

    for(size_t i = start_row; i < end_row; ++i)
    {
        double diag_val = 6.0 + sin((double)i) + cos((double)i * (double)i);
        A.data[i][i] = diag_val;

        if(i > 0)
        {
            double off1 = -1.0 + 0.5 * sin((double)i);
            A.data[i][i - 1] = off1;
            A.data[i - 1][i] = off1;
        }

        if(i > 1)
        {
            double off2 = -0.5 + 0.2 * cos((double)i);
            A.data[i][i - 2] = off2;
            A.data[i - 2][i] = off2;
        }
    }
    MPI_Win_fence(0, A.win_buf);

    return A;
}

void print_matrix_mpi(Matrix_MPI* A, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t n = A->rows;
    size_t m = A->cols;

    // Только rank 0 будет печатать матрицу целиком
    if(rank == 0)
    {
        printf("Matrix (%zu x %zu):\n", n, m);
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
            {
                printf("%8.4f ", A->data[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);
    }

    // Синхронизация: все процессы ждут завершения печати
    MPI_Barrier(comm);
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
    // 1) умножение и сохранение
    multiply_and_save(A, x, filename_b);

    // 2) чтение обратно
    Vector b_from_file = read_vector_from_file(filename_b);

    // 3) вычисление оригинного b
    Vector b_original = create_vector((int)A->rows);
    for (size_t i = 0; i < A->rows; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; ++j)
            sum += A->data[i][j] * x->data[j];
        b_original.data[i] = sum;
    }

    // 4) сравнение
    double max_diff = max_difference_between_vectors(&b_original, &b_from_file);

    // Освобождение
    free_vector(b_from_file);
    free_vector(b_original);

    return max_diff;
}

Matrix_MPI cholesky_mpi(Matrix_MPI* A, int n, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Verify matrix dimensions
    if(rank == 0)
    {
        assert(A->cols == A->rows);
        assert(A->cols == n);
    }

    // const char* filename = create_filename("mpi_matrix_a", rank);
    // double sum = 0;
    // Matrix A_d = {A->rows, A->cols, A->data, A->buf};
    // save_matrix_market(&A_d, filename, &sum);
    // printf("rank = %i, check sum = %f", rank, sum);

    // Broadcast assertion result to all processes
    int ok = 1;
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if(!ok)
    {
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Create shared matrix for L
    Matrix_MPI L = create_shared_matrix(n, n, comm);

    // Ensure A is also synchronized if it's shared
    if(A->win_buf != MPI_WIN_NULL)
    {
        MPI_Win_fence(0, A->win_buf);
    }

    for(int j = 0; j < n; j++)
    {
        // Synchronize before column computation
        MPI_Win_fence(0, L.win_buf);

        // Process 0 computes the diagonal element
        if(rank == 0)
        {
            double s = 0.0;
            for(int k = 0; k < j; k++)
            {
                s += L.data[j][k] * L.data[j][k];
            }
            L.data[j][j] = sqrt(A->data[j][j] - s);

            MPI_Win_sync(L.win_buf);
        }

        MPI_Barrier(comm);

        // CRITICAL: Force memory consistency across all processes
        // MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, L.win_buf);

        // Synchronize to ensure L[j][j] is visible to all
        MPI_Win_fence(0, L.win_buf);

        // All processes need the diagonal value for computations below
        // Broadcast L[j][j] to ensure all have it (though shared memory should
        // provide it) But we need to ensure memory consistency:
        double diag_val = L.data[j][j];

        // Parallelize the i-loop across processes
        int remaining_rows = n - j - 1;
        if(remaining_rows > 0)
        {
            int chunk_size = remaining_rows / size;
            int remainder = remaining_rows % size;

            int local_start = j + 1;
            int local_count = chunk_size + (rank < remainder ? 1 : 0);

            // Calculate global start for each process
            int offset = 0;
            for(int p = 0; p < rank; p++)
            {
                offset += chunk_size + (p < remainder ? 1 : 0);
            }
            local_start += offset;

            // Each process computes its assigned rows
            for(int i = local_start; i < local_start + local_count; i++)
            {
                double s = 0.0;
                for(int k = 0; k < j; k++)
                {
                    s += L.data[i][k] * L.data[j][k];
                }
                L.data[i][j] = (A->data[i][j] - s) / diag_val;
            }
        }

        // Synchronize after column completion
        MPI_Win_fence(0, L.win_buf);
    }

    // Final synchronization
    MPI_Win_fence(0, L.win_buf);

    // const char* filename2 = create_filename("mpi_matrix_l", rank);
    // double sum2 = 0;
    // Matrix L_d = {L.rows, L.cols, L.data, L.buf};
    // save_matrix_market(&L_d, filename2, &sum2);
    // printf("rank = %i, check sum = %f", rank, sum2);

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

void errByEpsPcgChol(double a, double b, double c, double d, double h,
                     MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Vector x = linspace(a, b, (b - a) / h + 1);
    Vector y = linspace(c, d, (d - c) / h + 1);
    Matrix_MPI A_MPI = generate_five_diag_mpi(x.size - 2, y.size - 2, comm);
    Matrix_MPI L_MPI = cholesky_mpi(&A_MPI, A_MPI.cols, comm);
    if(rank == 0)
    {
        Matrix L = {L_MPI.rows, L_MPI.cols, L_MPI.data, L_MPI.buf};
        Matrix A = {A_MPI.rows, A_MPI.cols, A_MPI.data, A_MPI.buf};
        Vector us = uForXY(&x, &y);
        Vector B = F(&x, &y);
        int n = (x.size - 2) * (y.size - 2);

        printf("%zu, %zu, %i, %i, %i\n", A.cols, A.rows, B.size, x.size,
               y.size);
        // scalar_mul_self(A, -1);

        scalar_vector_mult_self(&B, -1);

        Matrix Lt = transpose(L);
        Vector zeros = create_vector(n);
        int i = 3;

        double eps = pow(10, -i);
        double relres = 0;
        int count = 0;

        Vector Sol =
            pcgPreconditioned(&A, &B, &zeros, eps, &relres, &count, &L, &Lt);

        double max = vectors_max_diff(&us, &Sol);
        printf(", iter = %i, eps = %.15f, max_comp_diff =  %.15f\n", count, eps,
               max);

        free_vector(Sol);

        free_vector(us);
        free_vector(B);

        free_matrix(&Lt);
        free_vector(zeros);
        // free_matrix(&L);
    }
    free_vector(x);
    free_vector(y);
    free_shared_matrix(&A_MPI, comm);
    free_shared_matrix(&L_MPI, comm);
    if(rank == 0)
    {
        printf("---\n");
    }
}

void errByEpsPcgChol_general_mpi(size_t n, double eps, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Matrix_MPI A_MPI = generate_general_spd_mpi(n, comm);
    // print_matrix_mpi(&A_MPI, comm);

    Matrix_MPI L_MPI = cholesky_mpi(&A_MPI, (int)n, comm);

    Vector x_true_mpi = generate_true_x(n);
    // if (rank == 0) {
    //     print_vector(&x_true_mpi);

    // }

    size_t base = n / size;
    size_t rem  = n % size;

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    size_t offset = 0;
    for(int p=0; p<size; ++p)
    {
        counts[p] = base + (p < (int)rem ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }

    size_t start = displs[rank];
    size_t count = counts[rank];

    Vector b_local = create_vector((int)count);
    for(size_t ii = 0; ii < count; ++ii)
    {
        size_t i = start + ii;  // глобальный индекс строки

        double sum = 0.0;
        for(size_t j = 0; j < n; ++j)
        {
            sum += A_MPI.data[i][j] * x_true_mpi.data[j];
        }

        b_local.data[ii] = sum;
    }

    Vector b_full;
    if(rank == 0)
    {
        b_full = create_vector(n);
    }
    MPI_Gatherv(
        b_local.data,       // отправляемый буфер
        counts[rank],       // сколько элементов отправляет этот процесс
        MPI_DOUBLE,         // тип данных
        rank == 0 ? b_full.data : NULL,  // принимающий буфер на root
        counts,             // массив размеров для каждого процесса
        displs,             // массив смещений в итоговом векторе
        MPI_DOUBLE,
        0,                  // root
        comm
    );

    free_vector(b_local);
    free(counts);
    free(displs);

    if(rank == 0)
    {
        // printf("B vector");
        // print_vector(&b_full);
        Matrix A = {A_MPI.rows, A_MPI.cols, A_MPI.data, A_MPI.buf};
        Matrix L = {L_MPI.rows, L_MPI.cols, L_MPI.data, L_MPI.buf};

        Matrix Lt = transpose(L);

        Vector x0 = create_vector(n);

        double relres = 0.0;
        int iter = 0;

        Vector x_sol = pcgPreconditioned(&A, &b_full, &x0, eps, &relres,
                                         &iter, &L, &Lt);

        double max_err = vectors_max_diff(&x_true_mpi, &x_sol);

        printf("MPI PCG finished\n");
        printf("Iterations: %d, eps = %.3e, max_error = %.3e\n",
                iter, eps, max_err);

        free_vector(x_sol);
        free_vector(x0);
        free_matrix(&Lt);
        free_vector(b_full);
    }

    free_vector(x_true_mpi);
    free_shared_matrix(&A_MPI, comm);
    free_shared_matrix(&L_MPI, comm);

    MPI_Barrier(comm);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &shmcomm);

    int rank, size;
    MPI_Comm_rank(shmcomm, &rank);
    MPI_Comm_size(shmcomm, &size);

    if(rank == 0)
    {
        printf("Starting MPI program with %d processes (shared)\n", size);
    }

    size_t n = 4096;
    double eps = 1e-5;

    MPI_Barrier(shmcomm);

    errByEpsPcgChol_general_mpi(n, eps, shmcomm);

    MPI_Finalize();
    return 0;
}


// int main(int argc, char** argv)
// {
//     int rank, size;
//     MPI_Status status;

//     MPI_Init(&argc, &argv);
//     MPI_Comm shmcomm;
//     MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
//                         &shmcomm);
//     MPI_Comm_rank(shmcomm, &rank);
//     MPI_Comm_size(shmcomm, &size);

//     if(rank == 0)
//     {
//         printf("Starting MPI program with %d processes\n", size);
//     }

//     double a = 0;
//     double b = 1.5;
//     double c = 0;
//     double d = 1.5;
//     double h = 0.05;
//     // printf("----------------------------------------------------\n");
//     MPI_Barrier(shmcomm);

//     errByEpsPcgChol(a, b, c, d, h, shmcomm);

//     MPI_Finalize();

//     return 0;
// }