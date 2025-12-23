#include "utils/utils.h"
#include <assert.h>
#include <omp.h>
#include <stdio.h>

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

#pragma omp parallel for
    for(size_t i = 0; i < rows; ++i)
    {
        m.data[i] = m.buf + i * cols;
    }
    return m;
}

Matrix generate_five_diag(size_t xn, size_t yn)
{

    size_t n, nn;
    if(mul_overflow_size_t(xn, yn, &n) || mul_overflow_size_t(n, n, &nn))
    {
        fprintf(stderr, "Слишком большая сетка (переполнение).\n");
        exit(EXIT_FAILURE);
    }

    Matrix A = create_matrix_openmp(n, n);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        // центр
        A.data[i][i] = -4.0;

        // вверх (i - xn)
        if(i >= xn)
        {
            A.data[i][i - xn] = 1.0;
        }
        // вниз (i + xn)
        if(i + xn < n)
        {
            A.data[i][i + xn] = 1.0;
        }
        // влево (i - 1) — не на левом краю строки
        if((i % xn) != 0)
        {
            A.data[i][i - 1] = 1.0;
        }
        // вправо (i + 1) — не на правом краю строки
        if((i % xn) != xn - 1)
        {
            A.data[i][i + 1] = 1.0;
        }
    }
    return A;
}

Matrix generate_general_spd_openmp(size_t n)
{
    // Создаем плотную матрицу n×n
    Matrix A = create_matrix_openmp(n, n);

    // Параллельная инициализация всех элементов
    // Нам нужно заполнить только диагонали:
    // главную, ±1 и ±2.
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        // Обнуляем на всякий случай (если calloc не гарантирует)
        // Не обязателен, но безопасен при многопоточной инициализации
        for (size_t j = 0; j < n; ++j)
        {
            A.data[i][j] = 0.0;
        }

        // Главная диагональ 6 вместо 4
        double diag = 6.0 + sin((double)i) + cos((double)i * (double)i);
        A.data[i][i] = diag;

        // Под/над диагональ ±1
        if (i > 0)
        {
            double val1 = -1.0 + 0.5 * sin((double)i);
            A.data[i][i - 1] = val1;
            A.data[i - 1][i] = val1;
        }

        // Под/над диагональ ±2
        if (i > 1)
        {
            double val2 = -0.5 + 0.2 * cos((double)i);
            A.data[i][i - 2] = val2;
            A.data[i - 2][i] = val2;
        }
    }

    return A;
}

Vector generate_true_x(size_t n)
{
    // Создать вектор
    Vector x_star = create_vector(n);

    // Параллельно заполняем элементы
    #pragma omp parallel for schedule(static)
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


Matrix cholesky(Matrix* A, int n)
{
    // double sum = 0;
    // Matrix A_d = {A->rows, A->cols, A->data, A->buf};
    // save_matrix_market(&A_d, "openmp_matrix_a", &sum);

    Matrix L = create_matrix_openmp(n, n);
    assert(A->cols == A->rows);
    assert(A->cols == n);

    for(int j = 0; j < n; j++)
    {
        double s = 0;
        for(int k = 0; k < j; k++)
        {
            s += L.data[j][k] * L.data[j][k];
        }
        L.data[j][j] = sqrt(A->data[j][j] - s);
#pragma omp parallel for
        for(int i = j + 1; i < n; i++)
        {
            double s = 0;
            for(int k = 0; k < j; k++)
            {
                s += L.data[i][k] * L.data[j][k];
            }
            L.data[i][j] = (1.0 / L.data[j][j] * (A->data[i][j] - s));
        }
    }
    // double sum2 = 0;
    // Matrix L_d = {L.rows, L.cols, L.data, L.buf};
    // save_matrix_market(&L_d, "openmp_matrix_l", &sum2);
    // printf(" check sum = %f", sum2);

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

void errByEpsPcgChol(double a, double b, double c, double d, double h)
{
    Vector x = linspace(a, b, (b - a) / h + 1);
    Vector y = linspace(c, d, (d - c) / h + 1);
    Vector us = uForXY(&x, &y);
    Vector B = F(&x, &y);
    int n = (x.size - 2) * (y.size - 2);
    Matrix A = generate_five_diag(x.size - 2, y.size - 2);
    printf("%zu, %zu, %i, %i, %i\n", A.cols, A.rows, B.size, x.size, y.size);
    scalar_mul_self(A, -1);

    scalar_vector_mult_self(&B, -1);

    // printf("----\n");

    Matrix L = cholesky(&A, A.cols);
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

    free_vector(x);
    free_vector(y);
    free_vector(us);
    free_vector(B);
    free_matrix(&A);
    free_matrix(&L);
    free_matrix(&Lt);
    free_vector(zeros);
}

void errByEpsPcgChol_general_openmp(size_t n, const char *b_filename, double eps)
{
    printf("=== OpenMP: General SPD + PCG + Cholesky ===\n");
    printf("Matrix size: %zu x %zu, eps = %.3e\n", n, n, eps);

    // 1) Read right-hand side vector from file
    Vector B = read_vector_from_file(b_filename);
    if ((size_t)B.size != n)
    {
        fprintf(stderr, "Error: vector B has size %d but matrix size is %zu\n", B.size, n);
        exit(EXIT_FAILURE);
    }

    // 2) Generate general SPD matrix using OpenMP
    Matrix A = generate_general_spd_openmp(n);
    Vector x_true = generate_true_x(n);

    // 3) Compute Cholesky factorization in parallel
    printf("Running OpenMP Cholesky factorization...\n");
    Matrix L = cholesky(&A, (int)n);
    Matrix Lt = transpose(L);

    // 4) Initial guess x0 = zero vector
    Vector x0 = create_vector(n);

    double relres = 0.0;
    int iter = 0;

    // 5) Run Preconditioned CG with Cholesky preconditioner
    Vector Xsol = pcgPreconditioned(
        &A,
        &B,
        &x0,
        eps,
        &relres,
        &iter,
        &L,
        &Lt
    );

    // 6) Generate true solution vector x_true

    // 7) Compute max difference between PCG solution and true x*
    double max_err = max_difference_between_vectors(&x_true, &Xsol);

    printf("PCG finished\n");
    printf("Iterations: %d\n", iter);
    printf("Relative residual: %.3e\n", relres);
    printf("Max |x - x*|: %.3e\n", max_err);

    // 8) Free memory
    free_vector(Xsol);
    free_vector(x0);
    free_vector(B);
    free_vector(x_true);
    free_matrix(&A);
    free_matrix(&L);
    free_matrix(&Lt);
}



int main()
{
    size_t n = 4096;
    const char *bfile = "b_vector.txt"; 
    double eps = 1e-8;

    errByEpsPcgChol_general_openmp(n, bfile, eps);

    return 0;
}



// int main()
// {
//     double a = 0;
//     double b = 1.5;
//     double c = 0;
//     double d = 1.5;
//     double h = 0.05;

//     errByEpsPcgChol(a, b, c, d, h);

//     return 0;
// }