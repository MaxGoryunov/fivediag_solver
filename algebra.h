#ifndef MATRIX_HEADER
#define MATRIX_HEADER

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct
{
    size_t rows;
    size_t cols;
    double** data; 
    double* buf; 
} Matrix;

typedef struct
{
    int size;
    double* data;
} Vector;

// Выделяет двумерный блок размера b x b
double** allocate_block(int b)
{
    // Выделяем массив указателей на строки
    double** block = (double**)malloc(b * sizeof(double*));
    if (!block) {
        fprintf(stderr, "Ошибка выделения памяти для блока\n");
        exit(EXIT_FAILURE);
    }

    // Выделяем сами строки
    for (int i = 0; i < b; i++) {
        block[i] = (double*)malloc(b * sizeof(double));
        if (!block[i]) {
            fprintf(stderr, "Ошибка выделения памяти для строки блока\n");
            // Освобождаем уже выделенное
            for (int j = 0; j < i; j++)
                free(block[j]);
            free(block);
            exit(EXIT_FAILURE);
        }
    }

    return block;
}

// Освобождает блок, выделенный через allocate_block
void free_block(double** block, int b)
{
    if (!block) return;

    // Сначала освобождаем каждую строку
    for (int i = 0; i < b; i++) {
        free(block[i]);
    }

    // Потом освобождаем массив указателей
    free(block);
}


static void die(const char* msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

static int mul_overflow_size_t(size_t a, size_t b, size_t* res)
{
    // Простая проверка переполнения: если a > 0 и (*res)/a != b — переполнение
    // Возвращает 1 при переполнении, 0 иначе.
    if(a == 0 || b == 0)
    {
        *res = 0;
        return 0;
    }
    if(SIZE_MAX / a < b)
        return 1;
    *res = a * b;
    return 0;
}

// Вывод матрицы
void print_matrix(Matrix* mat)
{
    for(int i = 0; i < mat->rows; i++)
    {
        for(int j = 0; j < mat->cols; j++)
        {
            printf("%8.3f", mat->data[i][j]);
        }
        printf("\n");
    }
}

// Вывод вектора
void print_vector(Vector* vec)
{
    for(int i = 0; i < vec->size; i++)
    {
        printf("%8.2f", vec->data[i]);
    }
    printf("\n");
}

void save_matrix_to_file(Matrix* mat)
{
    FILE* fd = fopen("matrix.txt", "w");
    fprintf(fd, "%zu, %zu\n", mat->cols, mat->rows);
    for(int i = 0; i < mat->rows; i++)
    {
        for(int j = 0; j < mat->cols; j++)
        {
            fprintf(fd, "%8.2f", mat->data[i][j]);
        }
        fprintf(fd, "\n");
    }
}


static Matrix create_matrix(size_t rows, size_t cols)
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

    for(size_t i = 0; i < rows; ++i)
    {
        m.data[i] = m.buf + i * cols;
    }
    return m;
}

// Создание вектора
Vector create_vector(int size)
{
    Vector vec;
    vec.size = size;
    vec.data = (double*)malloc(size * sizeof(double));
    // Инициализация нулями
    for(int i = 0; i < size; i++)
    {
        vec.data[i] = 0.0;
    }
    return vec;
}

void write_matrix_to_file(const Matrix* M, const char* filename)
{
    FILE* f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Error: cannot open file %s for writing\n", filename);
        return;
    }

    // записываем размеры
    fprintf(f, "%zu %zu\n", M->rows, M->cols);

    // записываем элементы
    for (size_t i = 0; i < M->rows; i++)
    {
        for (size_t j = 0; j < M->cols; j++)
        {
            fprintf(f, "%.17g ", M->data[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

Matrix read_matrix_from_file(const char* filename)
{
    Matrix M;
    M.rows = 0;
    M.cols = 0;
    M.data = NULL;
    M.buf = NULL;

    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Error: cannot open file %s for reading\n", filename);
        return M;
    }

    // читаем размеры
    if (fscanf(f, "%zu %zu", &M.rows, &M.cols) != 2)
    {
        fprintf(stderr, "Error: failed to read matrix dimensions\n");
        fclose(f);
        return M;
    }

    // выделяем память под строки
    M.data = (double**)malloc(M.rows * sizeof(double*));
    if (!M.data)
    {
        fprintf(stderr, "Error: memory allocation failed for matrix rows\n");
        fclose(f);
        return M;
    }

    // выделяем буфер под все элементы
    M.buf = (double*)malloc(M.rows * M.cols * sizeof(double));
    if (!M.buf)
    {
        fprintf(stderr, "Error: memory allocation failed for matrix data\n");
        free(M.data);
        fclose(f);
        return M;
    }

    // распределяем буфер по строкам
    for (size_t i = 0; i < M.rows; i++)
    {
        M.data[i] = M.buf + i * M.cols;
    }

    // читаем элементы
    for (size_t i = 0; i < M.rows; i++)
    {
        for (size_t j = 0; j < M.cols; j++)
        {
            if (fscanf(f, "%lf", &M.data[i][j]) != 1)
            {
                fprintf(stderr, "Error reading element at (%zu,%zu)\n", i, j);
            }
        }
    }

    fclose(f);
    return M;
}

double max_difference_between_matrices(const Matrix* A, const Matrix* B)
{
    if (!A || !B || A->rows != B->rows || A->cols != B->cols)
    {
        fprintf(stderr, "Error: matrix dimensions must match for comparison\n");
        return -1.0;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < A->rows; i++)
    {
        for (size_t j = 0; j < A->cols; j++)
        {
            double diff = fabs(A->data[i][j] - B->data[i][j]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}


// Освобождение памяти матрицы
// void free_matrix(Matrix mat)
// {
//     // for(int i = 0; i < mat.rows; i++)
//     // {
//     //     free(mat.data[i]);
//     // }
//     // free(mat.data);
// }
static void free_matrix(Matrix* m)
{
    if(!m)
        return;
    free(m->buf);
    free(m->data);
    m->buf = NULL;
    m->data = NULL;
    m->rows = m->cols = 0;
}

// Освобождение памяти вектора
void free_vector(Vector vec)
{
    // free(vec.data);
}

// Ввод матрицы
void input_matrix(Matrix mat)
{
    printf("Введите элементы матрицы %zux%zu:\n", mat.rows, mat.cols);
    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            scanf("%lf", &mat.data[i][j]);
        }
    }
}

// Ввод вектора
void input_vector(Vector vec)
{
    printf("Введите элементы вектора размером %d:\n", vec.size);
    for(int i = 0; i < vec.size; i++)
    {
        scanf("%lf", &vec.data[i]);
    }
}

// Сложение матриц
Matrix add_matrix(Matrix a, Matrix b)
{
    if(a.rows != b.rows || a.cols != b.cols)
    {
        printf("Ошибка: размеры матриц не совпадают!\n");
        exit(1);
    }

    Matrix result = create_matrix(a.rows, a.cols);
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
}

// Вычитание матриц
Matrix sub_matrix(Matrix a, Matrix b)
{
    if(a.rows != b.rows || a.cols != b.cols)
    {
        printf("Ошибка: размеры матриц не совпадают!\n");
        exit(1);
    }

    Matrix result = create_matrix(a.rows, a.cols);
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

// Умножение матриц
Matrix mul_matrix(Matrix a, Matrix b)
{
    if(a.cols != b.rows)
    {
        printf("Ошибка: неподходящие размеры матриц для умножения!\n");
        exit(1);
    }

    Matrix result = create_matrix(a.rows, b.cols);
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < b.cols; j++)
        {
            result.data[i][j] = 0;
            for(int k = 0; k < a.cols; k++)
            {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

// Умножение матрицы на вектор
Vector matrix_vector_mult(Matrix* mat, Vector* vec)
{
    if(mat->cols != vec->size)
    {
        printf("Ошибка: количество столбцов матрицы должно совпадать с "
               "размером вектора!\n");
        exit(1);
    }

    Vector result = create_vector(mat->rows);
    for(int i = 0; i < mat->rows; i++)
    {
        result.data[i] = 0;
        for(int j = 0; j < mat->cols; j++)
        {
            result.data[i] += mat->data[i][j] * vec->data[j];
        }
    }
    return result;
}

// Умножение вектора на матрицу (вектор-строка на матрицу)
Vector vector_matrix_mult(Vector vec, Matrix mat)
{
    if(vec.size != mat.rows)
    {
        printf("Ошибка: размер вектора должен совпадать с количеством строк "
               "матрицы!\n");
        exit(1);
    }

    Vector result = create_vector(mat.cols);
    for(int j = 0; j < mat.cols; j++)
    {
        result.data[j] = 0;
        for(int i = 0; i < mat.rows; i++)
        {
            result.data[j] += vec.data[i] * mat.data[i][j];
        }
    }
    return result;
}

// Скалярное произведение векторов
double dot_product(Vector* a, Vector* b)
{
    if(a->size != b->size)
    {
        printf("Ошибка: размеры векторов не совпадают!\n");
        exit(1);
    }

    double result = 0.0;
    for(int i = 0; i < a->size; i++)
    {
        result += a->data[i] * b->data[i];
    }
    return result;
}

// Сложение векторов
Vector add_vector(Vector* a, Vector* b)
{
    if(a->size != b->size)
    {
        printf("Ошибка: размеры векторов не совпадают!\n");
        exit(1);
    }

    Vector result = create_vector(a->size);
    for(int i = 0; i < a->size; i++)
    {
        result.data[i] = a->data[i] + b->data[i];
    }
    return result;
}

void add_vector_self(Vector* a, Vector* b)
{
    if(a->size != b->size)
    {
        printf("Ошибка: размеры векторов не совпадают!\n");
        exit(1);
    }

    for(int i = 0; i < a->size; i++)
    {
        a->data[i] += b->data[i];
    }
}

// Вычитание векторов
Vector sub_vector(Vector* a, Vector* b)
{
    if(a->size != b->size)
    {
        printf("Ошибка: размеры векторов не совпадают!\n");
        exit(1);
    }

    Vector result = create_vector(a->size);
    for(int i = 0; i < a->size; i++)
    {
        result.data[i] = a->data[i] - b->data[i];
    }
    return result;
}

// Умножение вектора на скаляр
Vector scalar_vector_mult(Vector* vec, double scalar)
{
    Vector result = create_vector(vec->size);
    for(int i = 0; i < vec->size; i++)
    {
        result.data[i] = vec->data[i] * scalar;
    }
    return result;
}

void scalar_vector_mult_self(Vector* vec, double scalar)
{
    for(int i = 0; i < vec->size; i++)
    {
        vec->data[i] *= scalar;
    }
}

// Длина (норма) вектора
double vector_length(Vector vec)
{
    double sum = 0.0;
    for(int i = 0; i < vec.size; i++)
    {
        sum += vec.data[i] * vec.data[i];
    }
    return sqrt(sum);
}

// Нормализация вектора
Vector normalize_vector(Vector vec)
{
    double length = vector_length(vec);
    if(length == 0.0)
    {
        printf("Ошибка: нельзя нормализовать нулевой вектор!\n");
        return vec;
    }

    Vector result = create_vector(vec.size);
    for(int i = 0; i < vec.size; i++)
    {
        result.data[i] = vec.data[i] / length;
    }
    return result;
}

// Умножение на скаляр (матрица)
Matrix scalar_mul(Matrix mat, double scalar)
{
    Matrix result = create_matrix(mat.rows, mat.cols);
    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            result.data[i][j] = mat.data[i][j] * scalar;
        }
    }
    return result;
}

void scalar_mul_self(Matrix mat, double scalar)
{
    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            mat.data[i][j] *= scalar;
        }
    }
}

// Транспонирование матрицы
Matrix transpose(Matrix mat)
{
    Matrix result = create_matrix(mat.cols, mat.rows);
    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            result.data[j][i] = mat.data[i][j];
        }
    }
    return result;
}

// Создание единичной матрицы
Matrix identity_matrix(int size)
{
    Matrix result = create_matrix(size, size);
    for(int i = 0; i < size; i++)
    {
        result.data[i][i] = 1.0;
    }
    return result;
}

Vector copy_vector(Vector* vec)
{
    Vector v = create_vector(vec->size);
    for(int i = 0; i < vec->size; ++i)
    {
        v.data[i] = vec->data[i];
    }
    return v;
}

Vector residue(Vector* b, Matrix* A, Vector* x)
{
    printf("%i, %zu, %zu, %i", b->size, A->cols, A->rows, x->size);
    Vector v = matrix_vector_mult(A, x);
    return sub_vector(b, &v);
}

double second_norm(Vector* v)
{
    double sum = 0;
    for(int i = 0; i < v->size; ++i)
    {
        sum += v->data[i] * v->data[i];
    }
    return sqrt(sum);
}

double aKbyPQ(Vector* r, Vector* p, double pq)
{
    return dot_product(r, p) / pq;
}

double vectors_max_diff(Vector* left, Vector* right)
{
    if(left->size != right->size)
    {
        printf("sizes do not match\n");
        exit(1);
    }
    Vector diff = sub_vector(left, right);
    double max = 0;
    for(int i = 0; i < diff.size; ++i)
    {
        if(fabs(diff.data[i]) > max)
        {
            max = fabs(diff.data[i]);
        }
    }
    return max;
}

// Matrix genereate_five_diag(size_t xn, size_t yn)
// {
//     size_t n = xn * yn;
//     Matrix A = create_matrix(n, n);
//     for(int i = 0; i < n; ++i)
//     {
//         if(i == 553)
//             save_matrix_to_file(&A);
//         if(i - xn >= 0)
//         {
//             A.data[i][i - xn] = 1.0;
//         }

//         if(i - 1 >= 0 && (i % xn != 0))
//         {
//             A.data[i][i - 1] = 1.0;
//         }

//         A.data[i][i] = -4.0;
//         if(i + 1 <= n - 1 && ((i + 1) % xn != 0))
//         {
//             A.data[i][i + 1] = 1.0;
//         }

//         if(i + xn <= n - 1)
//         {
//             A.data[i][i + xn] = 1.0;
//         }
//     }
//     return A;
// }

char* create_filename(const char* prefix, int number)
{
    char* filename =
        malloc(strlen(prefix) + 15); 

    if(filename == NULL)
    {
        return NULL; 
    }

    sprintf(filename, "%s_%03d.txt", prefix, number);

    return filename;
}

void save_matrix_market(Matrix* A, const char* filename, double* check_sum)
{
    FILE* f = fopen(filename, "w");
    if(f == NULL)
    {
        perror("Failed to open file for writing");
        return;
    }

    size_t nnz = 0;
    for(size_t i = 0; i < A->rows; i++)
    {
        for(size_t j = 0; j < A->cols; j++)
        {
            if(A->data[i][j] != 0.0)
            {
                nnz++;
            }
        }
    }

    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%zu %zu %zu\n", A->rows, A->cols, nnz);

    double sum = 0;

    for(size_t i = 0; i < A->rows; i++)
    {
        for(size_t j = 0; j < A->cols; j++)
        {
            if(A->data[i][j] != 0.0)
            {
                sum += A->data[i][j];
                fprintf(f, "%zu %zu %.15g\n", i + 1, j + 1, A->data[i][j]);
            }
        }
    }

    fclose(f);

    if(check_sum != NULL)
    {
        *check_sum = sum;
    }
}

#endif