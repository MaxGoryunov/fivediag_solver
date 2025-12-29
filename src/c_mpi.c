#include "../algebra.h"
// #include "utils/utils.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdio.h>
#include <math.h>


typedef struct
{
    size_t rows;
    size_t cols;
    double** data;
    double* buf;
    MPI_Win win_buf; 
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

    MPI_Aint sz;
    int disp;
    double* base_ptr;


    MPI_Aint mysize = (rank == 0) ? buf_size : 0;
    MPI_Win_allocate_shared(mysize, sizeof(double), MPI_INFO_NULL, comm, &m.buf,
                            &m.win_buf);

    MPI_Win_shared_query(m.win_buf, 0, &sz, &disp, &base_ptr);

    double* common_buf = base_ptr;

    m.data = (double**)malloc(rows * sizeof(double*));

    for(size_t i = 0; i < rows; i++)
    {
        m.data[i] = common_buf + i * cols;
    }

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

Matrix_MPI create_matrix_local(size_t rows, size_t cols, MPI_Comm comm)
{
    Matrix_MPI m;
    m.rows = rows;
    m.cols = cols;
    m.win_buf = MPI_WIN_NULL;  

    // выделяем свой собственный буфер
    m.buf = malloc(rows * cols * sizeof(double));
    m.data = malloc(rows * sizeof(double*));
    for(size_t i = 0; i < rows; i++)
        m.data[i] = m.buf + i * cols;

    // можно заполнить нулями
    memset(m.buf, 0, rows * cols * sizeof(double));

    return m;
}


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

    if(rank == 0)
    {
        assert(A->cols == A->rows);
        assert(A->cols == n);
    }

    int ok = 1;
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if(!ok)
    {
        MPI_Abort(comm, EXIT_FAILURE);
    }

    Matrix_MPI L = create_shared_matrix(n, n, comm);

    if(A->win_buf != MPI_WIN_NULL)
    {
        MPI_Win_fence(0, A->win_buf);
    }

    for(int j = 0; j < n; j++)
    {
        MPI_Win_fence(0, L.win_buf);

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

        MPI_Win_fence(0, L.win_buf);

        double diag_val = L.data[j][j];

        int remaining_rows = n - j - 1;
        if(remaining_rows > 0)
        {
            int chunk_size = remaining_rows / size;
            int remainder = remaining_rows % size;

            int local_start = j + 1;
            int local_count = chunk_size + (rank < remainder ? 1 : 0);

            int offset = 0;
            for(int p = 0; p < rank; p++)
            {
                offset += chunk_size + (p < remainder ? 1 : 0);
            }
            local_start += offset;

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

        MPI_Win_fence(0, L.win_buf);
    }

    MPI_Win_fence(0, L.win_buf);



    return L;
}

int owner(int bi, int bj, int size) {
    return (bi * size + bj) % size;
}

// Заполняем блоки матрицы A распределённо
void distribute_A_blocks(double *A, int n, int b, MPI_Comm comm,
                         int rank, int size,
                         double **local_blocks, int *block_indices) {
    int nb = (n + b - 1) / b;
    if (rank == 0) {
        // Рассылаем блоки
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j <= i; j++) {
                int r = owner(i, j, size);
                int row = i * b;
                int col = j * b;
                int rows = (row + b > n) ? (n - row) : b;
                int cols = (col + b > n) ? (n - col) : b;

                if (r == 0) {
                    // Сохранить локально
                    local_blocks[i*nb + j] = malloc(rows * cols * sizeof(double));
                    for (int ii = 0; ii < rows; ii++)
                        for (int jj = 0; jj < cols; jj++)
                            local_blocks[i*nb + j][ii*cols + jj] =
                                A[(row + ii)*n + (col + jj)];
                } else {
                    // Отправляем блок
                    double *tmp = malloc(rows * cols * sizeof(double));
                    for (int ii = 0; ii < rows; ii++)
                        for (int jj = 0; jj < cols; jj++)
                            tmp[ii*cols + jj] = A[(row + ii)*n + (col + jj)];
                    MPI_Send(tmp, rows*cols, MPI_DOUBLE, r, i*nb + j, comm);
                    free(tmp);
                }
            }
        }
    } else {
        // ПРИНИМАЕМ только нужные блоки
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j <= i; j++) {
                int r = owner(i, j, size);
                int row = i * b;
                int col = j * b;
                int rows = (row + b > n) ? (n - row) : b;
                int cols = (col + b > n) ? (n - col) : b;
                if (r == rank) {
                    local_blocks[i*nb + j] = malloc(rows*cols*sizeof(double));
                    MPI_Recv(local_blocks[i*nb + j], rows*cols, MPI_DOUBLE,
                             0, i*nb + j, comm, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    MPI_Barrier(comm);
}

// Высвобождает только блок памяти, выделенный для матрицы
void free_block_mpi(double *block) {
    if (block) free(block);
}

// Факторизация диагонального блока Lkk = chol(Akk)
void factorize_block_mpi(double *Akk, double *Lkk, int rows, int cols) {
    // копируем блок A -> L
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j <= i; ++j) {
            Lkk[i*cols + j] = Akk[i*cols + j];
        }
        for (int j = i+1; j < cols; ++j) {
            Lkk[i*cols + j] = 0.0;  // обнуляем наддиагональные элементы
        }
    }
    // стандартный последовательный Cholesky
    for (int k = 0; k < rows; ++k) {
        double sum = 0.0;
        for (int p = 0; p < k; ++p)
            sum += Lkk[k*cols + p] * Lkk[k*cols + p];
        Lkk[k*cols + k] = sqrt(Lkk[k*cols + k] - sum);

        for (int i = k + 1; i < rows; ++i) {
            double sum2 = 0.0;
            for (int p = 0; p < k; ++p)
                sum2 += Lkk[i*cols + p] * Lkk[k*cols + p];
            Lkk[i*cols + k] = (Lkk[i*cols + k] - sum2) / Lkk[k*cols + k];
        }
    }
}

// Треугольное решение для Lik
void solve_triangular_block_mpi(
    double *Aik, double *Lkk, double *Lik, int rows, int cols)
{
    // Lik <- Aik * inv(Lkk^T)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int p = 0; p < j; p++)
                sum += Lik[i*cols + p] * Lkk[j*cols + p];
            Lik[i*cols + j] = (Aik[i*cols + j] - sum) / Lkk[j*cols + j];
        }
    }
}

// Обновление хвостового блока: Aij -= Lik * Ljk^T
void update_trailing_block_mpi(
    double *Aij, double *Lik, double *Ljk, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            double sum = 0.0;
            for (int p = 0; p < cols; p++)
                sum += Lik[i*cols + p] * Ljk[j*cols + p];
            Aij[i*rows + j] -= sum;
        }
    }
}


void block_cholesky_mpi(
    double *A,        // полная матрица A (только у rank 0, у остальных NULL)
    int n,             // размер матрицы
    int b,             // размер блока
    MPI_Comm comm,
    double *L_out      // результат L (только у rank 0, у остальных NULL)
)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int nb = (n + b - 1) / b;

    /* ---------------- Локальное хранилище блоков ---------------- */
    double **local_blocks = calloc(nb * nb, sizeof(double*));
    if (!local_blocks) MPI_Abort(comm, 1);

    /* ---------------- Распределение блоков ---------------- */
    if (rank == 0) {
        for (int bi = 0; bi < nb; bi++) {
            for (int bj = 0; bj <= bi; bj++) {

                int owner = (bi * nb + bj) % size;
                int rows = (bi*b + b <= n) ? b : (n - bi*b);
                int cols = (bj*b + b <= n) ? b : (n - bj*b);

                double *blk = malloc(rows * cols * sizeof(double));
                if (!blk) MPI_Abort(comm, 1);

                for (int ii = 0; ii < rows; ii++)
                    for (int jj = 0; jj < cols; jj++)
                        blk[ii*cols + jj] =
                            A[(bi*b + ii)*n + (bj*b + jj)];

                if (owner == 0) {
                    local_blocks[bi*nb + bj] = blk;
                } else {
                    MPI_Send(blk, rows*cols, MPI_DOUBLE,
                             owner, bi*nb + bj, comm);
                    free(blk);
                }
            }
        }
    } else {
        for (int bi = 0; bi < nb; bi++) {
            for (int bj = 0; bj <= bi; bj++) {

                int owner = (bi * nb + bj) % size;
                if (owner == rank) {

                    int rows = (bi*b + b <= n) ? b : (n - bi*b);
                    int cols = (bj*b + b <= n) ? b : (n - bj*b);

                    double *blk = malloc(rows * cols * sizeof(double));
                    if (!blk) MPI_Abort(comm, 1);

                    MPI_Recv(blk, rows*cols, MPI_DOUBLE,
                             0, bi*nb + bj, comm, MPI_STATUS_IGNORE);

                    local_blocks[bi*nb + bj] = blk;
                }
            }
        }
    }

    MPI_Barrier(comm);

    /* =================== ОСНОВНОЙ ЦИКЛ =================== */
    for (int k = 0; k < nb; k++) {

        int kk_rows = (k*b + b <= n) ? b : (n - k*b);

        /* ---------- 1. Диагональный блок Lkk ---------- */
        double *Lkk = malloc(kk_rows * kk_rows * sizeof(double));
        if (!Lkk) MPI_Abort(comm, 1);

        int owner_kk = (k*nb + k) % size;

        if (rank == owner_kk) {
            double *Akk = local_blocks[k*nb + k];
            factorize_block_mpi(Akk, Lkk, kk_rows, kk_rows);
            free(Akk);
            local_blocks[k*nb + k] = Lkk;
        }

        MPI_Bcast(Lkk, kk_rows*kk_rows, MPI_DOUBLE, owner_kk, comm);

        if (rank != owner_kk) {
            local_blocks[k*nb + k] = Lkk;
        }

        /* ---------- 2. Блоки под диагональю ---------- */
        for (int i = k + 1; i < nb; i++) {

            int ik_rows = (i*b + b <= n) ? b : (n - i*b);
            double *Lik = malloc(ik_rows * kk_rows * sizeof(double));
            if (!Lik) MPI_Abort(comm, 1);

            int owner_ik = (i*nb + k) % size;

            if (rank == owner_ik) {
                double *Aik = local_blocks[i*nb + k];
                solve_triangular_block_mpi(
                    Aik, Lkk, Lik, ik_rows, kk_rows);
                free(Aik);
                local_blocks[i*nb + k] = Lik;
            }

            MPI_Bcast(Lik, ik_rows*kk_rows, MPI_DOUBLE, owner_ik, comm);

            if (rank != owner_ik) {
                local_blocks[i*nb + k] = Lik;
            }
        }

        /* ---------- 3. Обновление хвоста ---------- */
        for (int i = k + 1; i < nb; i++) {
            for (int j = k + 1; j <= i; j++) {

                int owner_ij = (i*nb + j) % size;
                if (rank != owner_ij) continue;

                int rows = (i*b + b <= n) ? b : (n - i*b);
                int cols = (j*b + b <= n) ? b : (n - j*b);
                int kk = kk_rows;

                double *Aij = local_blocks[i*nb + j];
                double *Lik = local_blocks[i*nb + k];
                double *Ljk = local_blocks[j*nb + k];

                update_trailing_block_mpi(Aij, Lik, Ljk, rows, cols);
            }
        }

        MPI_Barrier(comm);
    }

    /* ---------------- Сборка L на rank 0 ---------------- */
    if (rank == 0) {
        for (int bi = 0; bi < nb; bi++) {
            for (int bj = 0; bj <= bi; bj++) {

                int rows = (bi*b + b <= n) ? b : (n - bi*b);
                int cols = (bj*b + b <= n) ? b : (n - bj*b);

                double *blk = local_blocks[bi*nb + bj];
                for (int ii = 0; ii < rows; ii++)
                    for (int jj = 0; jj < cols; jj++)
                        L_out[(bi*b + ii)*n + (bj*b + jj)] =
                            blk[ii*cols + jj];
            }
        }
    }

    /* ---------------- Освобождение памяти ---------------- */
    for (int i = 0; i < nb*nb; i++)
        if (local_blocks[i]) free(local_blocks[i]);

    free(local_blocks);
}



// void block_cholesky_mpi(
//     double *A,       // полный массив (у rank0) или NULL у других
//     int n,           // размер матрицы
//     int b,           // размер блока
//     MPI_Comm comm,
//     double *L_out)   // полный результат L (у rank0), NULL у других
// {
//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     int nb = (n + b - 1) / b; // количество блоков по строкам/столбцам

//     // Локальные блоки каждого процесса
//     // хранится как массив указателей на блоки размера b*b
//     double *local_blocks[nb*nb];
//     for (int i = 0; i < nb*nb; i++)
//         local_blocks[i] = NULL;

//     // Распределение блоков между процессами
//     if (rank == 0) {
//         for (int bi = 0; bi < nb; bi++) {
//             for (int bj = 0; bj <= bi; bj++) {
//                 int owner = (bi * nb + bj) % size;
//                 int rows = ((bi*b + b) > n ? n - bi*b : b);
//                 int cols = ((bj*b + b) > n ? n - bj*b : b);

//                 double *block = malloc(rows*cols*sizeof(double));
//                 for (int ii=0; ii<rows; ii++)
//                     for (int jj=0; jj<cols; jj++)
//                         block[ii*cols + jj] = A[(bi*b + ii)*n + (bj*b + jj)];

//                 if (owner == 0) {
//                     local_blocks[bi*nb + bj] = block;
//                 } else {
//                     MPI_Send(block, rows*cols, MPI_DOUBLE, owner,
//                              bi*nb + bj, comm);
//                     free(block);
//                 }
//             }
//         }
//     } else {
//         for (int bi = 0; bi < nb; bi++) {
//             for (int bj = 0; bj <= bi; bj++) {
//                 int owner = (bi * nb + bj) % size;
//                 if (owner == rank) {
//                     int rows = ((bi*b + b) > n ? n - bi*b : b);
//                     int cols = ((bj*b + b) > n ? n - bj*b : b);
//                     double *block = malloc(rows*cols*sizeof(double));
//                     MPI_Recv(block, rows*cols, MPI_DOUBLE,
//                              0, bi*nb + bj, comm, MPI_STATUS_IGNORE);
//                     local_blocks[bi*nb + bj] = block;
//                 }
//             }
//         }
//     }

//     MPI_Barrier(comm);

//     // Основной блочный алгоритм
//     for (int k = 0; k < nb; k++) {
//         // 1) Диагональный блок (k,k)
//         double *Lkk = NULL;
//         if (local_blocks[k*nb + k] != NULL) {
//             int rows = ((k*b + b) > n ? n - k*b : b);
//             double *tmp = malloc(rows*rows*sizeof(double));
//             factorize_block_mpi(local_blocks[k*nb + k], tmp, rows, rows);
//             free(local_blocks[k*nb + k]);
//             local_blocks[k*nb + k] = tmp;
//             Lkk = tmp;
//         }
//         // рассылка диагонального блока всем
//         MPI_Bcast(Lkk, b*b, MPI_DOUBLE, (k*nb + k) % size, comm);

//         // 2) Блоки столбца
//         for (int i = k+1; i < nb; i++) {
//             int idx = i*nb + k;
//             double *Lik = local_blocks[idx];
//             if (Lik != NULL) {
//                 int rows = ((i*b + b) > n ? n - i*b : b);
//                 double *tmp = malloc(rows*b*sizeof(double));
//                 solve_triangular_block_mpi(
//                     local_blocks[idx], Lkk, tmp, rows, b);
//                 free(local_blocks[idx]);
//                 local_blocks[idx] = tmp;
//                 Lik = tmp;
//             }
//             MPI_Bcast(Lik, b*b, MPI_DOUBLE, idx % size, comm);
//         }

//         // 3) Обновление хвоста
//         for (int i = k+1; i < nb; i++) {
//             for (int j = k+1; j <= i; j++) {
//                 int idx = i*nb + j;
//                 if (local_blocks[idx] != NULL) {
//                     double *Aij = local_blocks[idx];
//                     double *Lik = local_blocks[i*nb + k];
//                     double *Ljk = local_blocks[j*nb + k];
//                     update_trailing_block_mpi(Aij, Lik, Ljk, b, b);
//                 }
//             }
//         }
//         MPI_Barrier(comm);
//     }

//     // Собираем L обратно на rank 0
//     if (rank == 0) {
//         for (int bi = 0; bi < nb; bi++) {
//             for (int bj = 0; bj <= bi; bj++) {
//                 int rows = ((bi*b + b) > n ? n - bi*b : b);
//                 int cols = ((bj*b + b) > n ? n - bj*b : b);
//                 double *block = local_blocks[bi*nb + bj];
//                 for (int ii=0; ii<rows; ii++)
//                     for (int jj=0; jj<cols; jj++)
//                         L_out[(bi*b + ii)*n + (bj*b + jj)] =
//                         block[ii*cols + jj];
//             }
//         }
//     }

//     // Free local blocks
//     for (int bi = 0; bi < nb; bi++)
//         for (int bj = 0; bj <= bi; bj++)
//             if (local_blocks[bi*nb + bj] != NULL)
//                 free_block_mpi(local_blocks[bi*nb + bj]);
// }





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

double** make_matrix_ptrs(double *buf, size_t rows, size_t cols)
{
    double **data = malloc(rows * sizeof(double *));
    if (!data) return NULL;

    for (size_t i = 0; i < rows; ++i)
    {
        data[i] = buf + i * cols;
    }
    return data;
}

void fill_spd_matrix(double *A, size_t n)
{
    // Заполняем нулями
    for (size_t i = 0; i < n*n; ++i) {
        A[i] = 0.0;
    }

    // Диагональные и окрестные элементы
    for (size_t i = 0; i < n; ++i) {
        A[i*n + i] = 6.0 + sin((double)i) + cos((double)i * (double)i);
        if (i > 0) {
            double v = -1.0 + 0.5 * sin((double)i);
            A[i*n + (i-1)] = v;
            A[(i-1)*n + i] = v;
        }
        if (i > 1) {
            double v2 = -0.5 + 0.2 * cos((double)i);
            A[i*n + (i-2)] = v2;
            A[(i-2)*n + i] = v2;
        }
    }
}



void pcgCholMPI(size_t n, double eps, const char *filename_b, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // printf("Inside Chol: rank = %i, size = %i\n", rank, size);

    // --- Генерация SPD матрицы A и её распространение ---
    double *A_full = malloc(n * n * sizeof(double));
    if (!A_full) MPI_Abort(comm, 1);
    if (rank == 0) {
        fill_spd_matrix(A_full, n);
    }
    // printf("A was filled\n");
    
    MPI_Bcast(A_full, n*n, MPI_DOUBLE, 0, comm);
    // printf("A was broadcasted\n");

    // --- Блочное распределённое разложение Холецкого ---
    // double *L_full = NULL;
    double *L_full = malloc(n * n * sizeof(double));
    // if(rank == 0) {
    //     L_full = malloc(n * n * sizeof(double));
    // }
    int block_size = 64;
    // printf("Start Cholesky factorization...\n");
    block_cholesky_mpi(A_full, n, block_size, comm, L_full);
    // printf("Completed Cholesky factorization\n");

    // --- Чтение вектора b из файла только на rank 0 ---
    Vector b_true;
    if(rank == 0) {
        b_true = read_vector_from_file(filename_b);
        if(b_true.size != (int)n) {
            fprintf(stderr, "Vector size mismatch (%d vs %zu)\n",
                    b_true.size, n);
            MPI_Abort(comm, 1);
        }
    }

    // --- Распространяем вектор b всем процессам ---
    if(rank != 0) {
        b_true.data = malloc(n * sizeof(double));
    }
    MPI_Bcast(b_true.data, n, MPI_DOUBLE, 0, comm);


    // --- PCG только на rank 0 ---
    if(rank == 0) {
        double *Anew = malloc(n * n * sizeof(double));
        if (!Anew) MPI_Abort(comm, 1);
        fill_spd_matrix(Anew, n);
        Matrix A_mat = { n, n, make_matrix_ptrs(Anew, n, n) };
        Matrix L_mat = { n, n, make_matrix_ptrs(L_full, n, n) };
        // printf("Matrix A\n");
        // print_matrix(&A_mat);
        // Matrix Lfile = read_matrix_from_file("matrix_debug.txt");
        Matrix Lt_mat = transpose(L_mat);
        // printf("Max diff between L and Ltrue: %.15f\n", 
        //     max_difference_between_matrices(&L_mat, &Lfile));
        Vector x0 = create_vector(n);
        Vector x_true = generate_true_x(n);

        double relres = 0.0;
        int iter = 0;
        // printf("Vector :\n");
        // print_vector(&b_true);

        Vector x_sol = pcgPreconditioned(
            &A_mat, &b_true, &x0, eps,
            &relres, &iter,
            &L_mat, &Lt_mat
        );

        double max_err = vectors_max_diff(&x_true, &x_sol);

        printf("MPI PCG finished\n");
        printf("Iterations: %d, eps=%.3e, relres=%.3e max_error=%.3e\n",
               iter, eps, relres, max_err);

        free_vector(x_sol);
        free_vector(x0);
        free_matrix(&Lt_mat);
    }

    free_vector(b_true);
    free(A_full);
    if(rank == 0) free(L_full);

    MPI_Barrier(comm);
}




// void pcgCholMPI(size_t n, double eps, MPI_Comm comm)
// {
//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     Matrix_MPI A_MPI = generate_general_spd_mpi(n, comm);
    
//     // print_matrix_mpi(&A_MPI, comm);

//     Matrix_MPI L_MPI = cholesky_mpi(&A_MPI, (int)n, comm);

//     Vector x_true_mpi = generate_true_x(n);
//     // if (rank == 0) {
//     //     print_vector(&x_true_mpi);

//     // }

//     size_t base = n / size;
//     size_t rem  = n % size;

//     int *counts = malloc(size * sizeof(int));
//     int *displs = malloc(size * sizeof(int));
//     size_t offset = 0;
//     for(int p=0; p<size; ++p)
//     {
//         counts[p] = base + (p < (int)rem ? 1 : 0);
//         displs[p] = offset;
//         offset += counts[p];
//     }

//     size_t start = displs[rank];
//     size_t count = counts[rank];

//     Vector b_local = create_vector((int)count);
//     for(size_t ii = 0; ii < count; ++ii)
//     {
//         size_t i = start + ii;  // глобальный индекс строки

//         double sum = 0.0;
//         for(size_t j = 0; j < n; ++j)
//         {
//             sum += A_MPI.data[i][j] * x_true_mpi.data[j];
//         }

//         b_local.data[ii] = sum;
//     }

//     Vector b_full;
//     if(rank == 0)
//     {
//         b_full = create_vector(n);
//     }
//     MPI_Gatherv(
//         b_local.data,       
//         counts[rank],       
//         MPI_DOUBLE,         
//         rank == 0 ? b_full.data : NULL,  
//         counts,             
//         displs,             
//         MPI_DOUBLE,
//         0,                  
//         comm
//     );

//     free_vector(b_local);
//     free(counts);
//     free(displs);

//     if(rank == 0)
//     {
//         // printf("B vector");
//         // print_vector(&b_full);
//         Matrix A = {A_MPI.rows, A_MPI.cols, A_MPI.data, A_MPI.buf};
//         Matrix L = {L_MPI.rows, L_MPI.cols, L_MPI.data, L_MPI.buf};

//         Matrix Lt = transpose(L);

//         Vector x0 = create_vector(n);

//         double relres = 0.0;
//         int iter = 0;

//         Vector x_sol = pcgPreconditioned(&A, &b_full, &x0, eps, &relres,
//                                          &iter, &L, &Lt);

//         double max_err = vectors_max_diff(&x_true_mpi, &x_sol);

//         printf("MPI PCG finished\n");
//         printf("Iterations: %d, eps = %.3e, max_error = %.3e\n",
//                 iter, eps, max_err);

//         free_vector(x_sol);
//         free_vector(x0);
//         free_matrix(&Lt);
//         free_vector(b_full);
//     }

//     free_vector(x_true_mpi);
//     free_shared_matrix(&A_MPI, comm);
//     free_shared_matrix(&L_MPI, comm);

//     MPI_Barrier(comm);
// }



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int size;
    MPI_Comm_size(comm, &size);


    if(rank == 0) {
        printf("Starting MPI program with %d processes\n",
               (int)size);
    }

    size_t n       = 4096;
    double eps     = 1e-5;
    const char* file_b = "b_vector.txt";

    MPI_Barrier(comm);

    pcgCholMPI(n, eps, file_b, comm);

    MPI_Finalize();
    return 0;
}




// int main(int argc, char** argv)
// {
//     MPI_Init(&argc, &argv);

//     MPI_Comm shmcomm;
//     MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
//                         0, MPI_INFO_NULL, &shmcomm);

//     int rank, size;
//     MPI_Comm_rank(shmcomm, &rank);
//     MPI_Comm_size(shmcomm, &size);

//     if(rank == 0)
//     {
//         printf("Starting MPI program with %d processes (shared)\n", size);
//     }

//     size_t n = 4096;
//     double eps = 1e-5;

//     MPI_Barrier(shmcomm);

//     pcgCholMPI(n, eps, shmcomm);

//     MPI_Finalize();
//     return 0;
// }
