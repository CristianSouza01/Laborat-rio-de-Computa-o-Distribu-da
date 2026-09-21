"""
Exercicio 1 - Multiplicacao de Matrizes Distribuida com MPI
Lab 3 - Computacao Distribuida - FCI Mackenzie
Aluno: Cristian de Souza Araujo - RA 10436050 - Turma 06N

Estrategia:
  1. O rank 0 gera as matrizes A e B (N x N) com valores aleatorios.
  2. A e B sao replicadas em todos os processos via comm.bcast.
  3. Cada rank r calcula apenas a sua faixa de linhas de C (particionamento 1-D por linhas).
  4. As faixas parciais retornam ao rank 0 via comm.gather.
  5. O rank 0 remonta C e imprime o tempo total em milissegundos.
"""

import random
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Dimensao da matriz (default 300, pode vir por argumento de linha de comando)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def faixa_de_linhas(rank, size, n):
    """Divide n linhas entre size processos distribuindo o resto entre os primeiros ranks."""
    base = n // size
    resto = n % size
    inicio = rank * base + min(rank, resto)
    fim = inicio + base + (1 if rank < resto else 0)
    return inicio, fim


# ---------------------------------------------------------------- 1) Geracao
if rank == 0:
    random.seed(42)
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    print(f"[master] Matrizes A e B geradas com dimensao {N}x{N}")
    print(f"[master] Cluster com {size} processos MPI")
else:
    A = None
    B = None

comm.Barrier()
inicio = MPI.Wtime()

# ------------------------------------------------- 2) Distribuicao (bcast)
A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

# ------------------------------------------- 3) Calculo distribuido local
ini, fim = faixa_de_linhas(rank, size, N)
C_local = [[0.0] * N for _ in range(fim - ini)]

for i in range(ini, fim):
    linha_A = A[i]
    destino = C_local[i - ini]
    for k in range(N):
        a_ik = linha_A[k]
        if a_ik == 0.0:
            continue
        linha_B = B[k]
        for j in range(N):
            destino[j] += a_ik * linha_B[j]

print(f"[rank {rank}] processou as linhas {ini}..{fim - 1} "
      f"({fim - ini} linhas de C) no host {MPI.Get_processor_name()}")

# ------------------------------------------------ 4) Coleta (gather)
partes = comm.gather(C_local, root=0)

# ---------------------------------- 5) Consolidacao e tempo total no rank 0
if rank == 0:
    C = []
    for parte in partes:
        C.extend(parte)
    fim_t = MPI.Wtime()
    soma = sum(sum(linha) for linha in C)
    print("-" * 62)
    print(f"Dimensao N ............: {N}")
    print(f"Processos MPI .........: {size}")
    print(f"Linhas de C montadas ..: {len(C)}")
    print(f"C[0][0] ...............: {C[0][0]:.6f}")
    print(f"Soma acumulada de C ...: {soma:.6f}")
    print(f"Tempo distribuido MPI .: {(fim_t - inicio) * 1000:.2f} ms")
    print("-" * 62)
