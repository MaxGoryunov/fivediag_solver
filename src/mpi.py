import numpy as np
from mpi4py import MPI
import math

class SharedMatrix:
    def __init__(self, rows, cols, comm=None):
        self.rows = rows
        self.cols = cols
        self.data = None
        self.comm = comm if comm else MPI.COMM_WORLD
        self.win = None
        
        size = rows * cols
        if comm.Get_rank() == 0:
            self.win = MPI.Win.Allocate_shared(size * 8, 8, comm=self.comm)
        else:
            self.win = MPI.Win.Allocate_shared(0, 8, comm=self.comm)
        
        buf, itemsize = self.win.Shared_query(0)
        self.data = np.ndarray(buffer=buf, shape=(rows, cols), dtype=np.float64)
        
        if self.comm.Get_rank() == 0:
            self.data.fill(0.0)
        
        self.win.Fence()
    
    def free(self):
        if self.win:
            self.win.Fence()
            self.win.Free()
            self.win = None
        self.data = None

def create_shared_matrix(rows, cols, comm):
    return SharedMatrix(rows, cols, comm)

def free_shared_matrix(mat):
    if mat:
        mat.free()

def generate_general_spd_mpi(xn, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = xn
    A = create_shared_matrix(n, n, comm)
    rows_per_proc = n // size
    remainder = n % size

    start_row = rank * rows_per_proc + min(rank, remainder)
    end_row = start_row + rows_per_proc + (1 if rank < remainder else 0)

    for i in range(start_row, end_row):
        A.data[i, i] = 6.0 + math.sin(i) + math.cos(i*i)

        if i > 0:
            val1 = -1.0 + 0.5 * math.sin(i)
            A.data[i, i - 1] = val1
            A.data[i - 1, i] = val1

        if i > 1:
            val2 = -0.5 + 0.2 * math.cos(i)
            A.data[i, i - 2] = val2
            A.data[i - 2, i] = val2
    A.win.Fence()

    return A

def generate_true_x_mpi(n, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_count = n // size
    remainder = n % size

    counts = [base_count + (1 if i < remainder else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    local_n = counts[rank]
    start_index = displs[rank]

    local_x = np.zeros(local_n, dtype=np.float64)
    for idx in range(local_n):
        i = start_index + idx
        local_x[idx] = (math.sin(i) +
                        math.cos(math.sqrt(i)) +
                        0.3 * i)

    if rank == 0:
        x_star = np.empty(n, dtype=np.float64)
    else:
        x_star = None

    comm.Gatherv(sendbuf=local_x,
                 recvbuf=(x_star, counts, displs, MPI.DOUBLE),
                 root=0)

    return x_star

def read_vector_from_file_mpi(filename, comm):
    rank = comm.Get_rank()
    
    b = None

    if rank == 0:
        with open(filename, "r") as f:
            size = int(f.readline().strip())
            data = []
            for _ in range(size):
                line = f.readline().strip()
                if line:
                    data.append(float(line))
            b = np.array(data, dtype=np.float64)
            if len(b) != size:
                raise ValueError("File size does not match the number of values")
    b = comm.bcast(b, root=0)
    return b


def cholesky_mpi(A, n, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        assert A.cols == A.rows == n
    
    L = create_shared_matrix(n, n, comm)
    
    if A.win:
        A.win.Fence()
    
    for j in range(n):
        L.win.Fence()
        
        if rank == 0:
            s = 0.0
            for k in range(j):
                s += L.data[j, k] * L.data[j, k]
            L.data[j, j] = math.sqrt(A.data[j, j] - s)
        
        L.win.Fence()
        
        diag_val = L.data[j, j]
        
        remaining_rows = n - j - 1
        if remaining_rows > 0:
            chunk_size = remaining_rows // size
            remainder = remaining_rows % size
            
            local_count = chunk_size + (1 if rank < remainder else 0)
            
            offset = 0
            for p in range(rank):
                offset += chunk_size + (1 if p < remainder else 0)
            local_start = j + 1 + offset
            
            for i in range(local_start, local_start + local_count):
                s = 0.0
                for k in range(j):
                    s += L.data[i, k] * L.data[j, k]
                L.data[i, j] = (A.data[i, j] - s) / diag_val
        
        L.win.Fence()
    
    L.win.Fence()
    return L

def solve_gauss_forward(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        
        if abs(L[i, i]) < 1e-12:
            print(f"Error: zero diagonal element in row {i}!")
            return None
        
        y[i] = s / L[i, i]
    
    return y

def solve_gauss_reverse(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= U[i, j] * x[j]
        
        if abs(U[i, i]) < 1e-12:
            print(f"Error: zero diagonal element in row {i}!")
            return None
        
        x[i] = s / U[i, i]
    
    return x

def pcg_preconditioned(A, b, x0, tol, max_iter=1000, L=None, Lt=None):
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    
    if L is not None and Lt is not None:
        z = solve_gauss_forward(L, r)
        z = solve_gauss_reverse(Lt, z)
    else:
        z = r.copy()
    
    p = z.copy()
    
    r0_norm = np.linalg.norm(r)
    
    for k in range(max_iter):
        q = A @ p
        pq = np.dot(p, q)
        
        if abs(pq) < 1e-15:
            break
            
        alpha = np.dot(z, r) / pq
        
        x += alpha * p
        r_new = r - alpha * q
        
        rel_res = np.linalg.norm(r_new) / r0_norm
        if rel_res < tol:
            r = r_new
            break
        
        if L is not None and Lt is not None:
            z_new = solve_gauss_forward(L, r_new)
            z_new = solve_gauss_reverse(Lt, z_new)
        else:
            z_new = r_new.copy()
        
        beta = np.dot(z_new, r_new) / np.dot(z, r)
        
        p = z_new + beta * p
        r = r_new
        z = z_new
    
    return x, k+1, np.linalg.norm(r) / r0_norm





def block_indices(n, b):
    nb = (n + b - 1) // b
    for i in range(nb):
        for j in range(i + 1):
            yield i, j

def owner(i, j, size):
    return (i * (i + 1) // 2 + j) % size

def cholesky_blocked_mpi(A, b, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = A.shape[0]
    nb = (n + b - 1) // b

    local_blocks = {}

    if rank == 0:
        blocks = {}
        for i, j in block_indices(n, b):
            Ii, Ji = i*b, j*b
            I_len = min(b, n - Ii)
            J_len = min(b, n - Ji)
            blocks[(i,j)] = A[Ii:Ii+I_len, Ji:Ji+J_len].copy()
        for (i,j), blk in blocks.items():
            tgt = owner(i,j, size)
            if tgt == 0:
                local_blocks[(i,j)] = blk
            else:
                comm.send(blk, dest=tgt, tag=1000+i*nb+j)
    else:
        for i, j in block_indices(n,b):
            if owner(i,j,size) == rank:
                local_blocks[(i,j)] = comm.recv(source=0, tag=1000+i*nb+j)

    comm.Barrier()

    recv_buf = {}

    for k in range(nb):
        if (k,k) in local_blocks:
            Lkk = np.linalg.cholesky(local_blocks[(k,k)])
            local_blocks[(k,k)] = Lkk
        else:
            Lkk = None
        Lkk = comm.bcast(Lkk, root=owner(k,k,size))

        for i in range(k+1, nb):
            if (i,k) in local_blocks:
                Aik = local_blocks[(i,k)]
                Lik = np.linalg.solve(Lkk, Aik.T).T
                local_blocks[(i,k)] = Lik
            else:
                Lik = None
            Lik = comm.bcast(Lik, root=owner(i,k,size))
            recv_buf[(i,k)] = Lik

        comm.Barrier()

        for i in range(k+1, nb):
            for j in range(k+1, i+1):
                if (i,j) in local_blocks:
                    Aij = local_blocks[(i,j)]
                else:
                    Aij = np.zeros((0,0))

                Lik = recv_buf[(i,k)]
                Ljk = recv_buf[(j,k)]

                if Aij.size != 0:
                    Aij -= Lik @ Ljk.T
                    if i == j:
                        Aij = np.tril(Aij)
                    local_blocks[(i,j)] = Aij

        comm.Barrier()

    L = None
    if rank == 0:
        L = np.zeros_like(A)
        for (i,j), blk in local_blocks.items():
            Ii, Ji = i*b, j*b
            L[Ii: Ii+blk.shape[0], Ji: Ji+blk.shape[1]] = blk
        for r in range(1, size):
            for i,j in block_indices(n,b):
                if owner(i,j,size) == r:
                    blk = comm.recv(source=r, tag=2000+i*nb+j)
                    Ii, Ji = i*b, j*b
                    L[Ii: Ii+blk.shape[0], Ji: Ji+blk.shape[1]] = blk
    else:
        for (i,j), blk in local_blocks.items():
            comm.send(blk, dest=0, tag=2000+i*nb+j)

    return L



def pcg_chol_MPI(filename_b, n, eps, comm):
    rank = comm.Get_rank()
    b_vec = read_vector_from_file_mpi(filename_b, comm)
    # print(f"Read b_vec, {len(b_vec)}")
    A_mpi = generate_general_spd_mpi(n, comm)
    if rank == 0:
        print(f"{A_mpi.rows = }")
    # L_mpi = cholesky_mpi(A_mpi, n, comm)
    L_mpi = cholesky_blocked_mpi(A_mpi.data, b=64, comm=MPI.COMM_WORLD)
    if rank == 0:
        print(f"Finished Cholesky, {L_mpi.shape = }")

    if rank == 0:
        A = A_mpi.data
        # print('A:')
        # print(A)
        # print('b (right side):')
        # print(b_vec)
        
        # if pure numpy matrix
        # L = L_mpi.data
        L = L_mpi
        x0 = np.zeros(n, dtype=np.float64)

        sol, iterations, relres = pcg_preconditioned(
            A, b_vec, x0, eps, L=L, Lt=L.T
        )

        print(f"eps={eps:.1e}, iterations={iterations}, relres={relres:.3e}")

    free_shared_matrix(A_mpi)
    # free_shared_matrix(L_mpi)

    comm.Barrier()



def main():
    comm = MPI.COMM_WORLD
    shmcomm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    rank = shmcomm.Get_rank()

    if rank == 0:
        print("Starting MPI Python program")
    # print(f"Starting MPI Python program on {rank = }")

    filename_b = "b_vector.txt"
    n = 4096
    eps = 1e-8

    pcg_chol_MPI(filename_b, n, eps, shmcomm)


if __name__ == "__main__":
    main()