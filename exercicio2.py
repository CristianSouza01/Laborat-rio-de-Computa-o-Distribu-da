"""
Exercicio 2 - Estimativa de PI pelo Metodo de Monte Carlo com MPI
Lab 3 - Computacao Distribuida - FCI Mackenzie
Aluno: Cristian de Souza Araujo - RA 10436050 - Turma 06N

Estrategia:
  1. O total de N pontos e dividido igualmente entre os size processos.
  2. Cada rank sorteia N/size pares (x, y) em [0,1] e conta quantos caem
     dentro do quarto de circulo de raio 1 (x^2 + y^2 <= 1).
  3. Os contadores locais sao somados com comm.reduce(op=MPI.SUM, root=0).
  4. O rank 0 calcula pi = 4 * dentro / N e imprime o tempo total.
"""

import random
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

# ------------------------------------------------------ 1) Particionamento
base = N // size
resto = N % size
N_local = base + (1 if rank < resto else 0)

if rank == 0:
    print(f"[master] Estimativa de PI por Monte Carlo com N = {N:,} pontos")
    print(f"[master] {size} processos MPI, ~{base:,} pontos por processo")

comm.Barrier()
inicio = MPI.Wtime()

# --------------------------------------------- 2) Geracao e contagem local
random.seed(1234 + rank)  # semente distinta por rank -> fluxos independentes
dentro_local = 0
for _ in range(N_local):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1.0:
        dentro_local += 1

print(f"[rank {rank}] {N_local:,} pontos sorteados, {dentro_local:,} dentro do circulo "
      f"({100.0 * dentro_local / N_local:.4f}%) no host {MPI.Get_processor_name()}")

# ------------------------------------------------ 3) Agregacao (reduce SUM)
dentro_total = comm.reduce(dentro_local, op=MPI.SUM, root=0)

# ------------------------------------------------------- 4) Calculo final
if rank == 0:
    fim = MPI.Wtime()
    pi_estimado = 4.0 * dentro_total / N
    erro = abs(pi_estimado - 3.141592653589793)
    print("-" * 62)
    print(f"Pontos totais .........: {N:,}")
    print(f"Pontos dentro .........: {dentro_total:,}")
    print(f"PI aproximado .........: {pi_estimado:.6f}")
    print(f"PI de referencia ......: 3.141593")
    print(f"Erro absoluto .........: {erro:.6f}")
    print(f"Tempo distribuido MPI .: {(fim - inicio) * 1000:.2f} ms")
    print("-" * 62)
