import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def lu_decomposition(A, b):
    n_size = get_2d_array_num_rows(A)
    P = torch.zeros(int(n_size), int(n_size))
    for i in range(int(0), int(n_size)):
        P[int(i), int(i)] = 1.0
    L = torch.zeros(int(n_size), int(n_size))
    U = torch.zeros(int(n_size), int(n_size))
    for j in range(int(0), int(n_size)):
        max_row = j
        for i in range(int((j + 1)), int(n_size)):
            if torch.abs(A[int(i), int(j)] if isinstance(A[int(i), int(j)], torch.Tensor) else torch.tensor(float(A[int(i), int(j)]))) > torch.abs(A[int(max_row), int(j)] if isinstance(A[int(max_row), int(j)], torch.Tensor) else torch.tensor(float(A[int(max_row), int(j)]))):
                max_row = i
        A_next = torch.zeros(int(n_size), int(n_size))
        L_next = torch.zeros(int(n_size), int(n_size))
        P_next = torch.zeros(int(n_size), int(n_size))
        for r in range(int(0), int(n_size)):
            source = r
            if r == j:
                source = max_row
            else:
                if r == max_row:
                    source = j
            for c in range(int(0), int(n_size)):
                A_next[int(r), int(c)] = A[int(source), int(c)]
                L_next[int(r), int(c)] = L[int(source), int(c)]
                P_next[int(r), int(c)] = P[int(source), int(c)]
        A = A_next
        L = L_next
        P = P_next
        u_col = torch.zeros(int(n_size))
        for i in range(int(0), int((j + 1))):
            partial = 0.0
            for k in range(int(0), int(i)):
                partial = (partial + (u_col[int(k)] * L[int(i), int(k)]))
            new_val = (A[int(i), int(j)] - partial)
            u_col_next = torch.zeros(int(n_size))
            for c in range(int(0), int(n_size)):
                if c == i:
                    u_col_next[int(c)] = new_val
                else:
                    u_col_next[int(c)] = u_col[int(c)]
            u_col = u_col_next
        l_col = torch.zeros(int(n_size))
        for i in range(int(j), int(n_size)):
            partial = 0.0
            for k in range(int(0), int(j)):
                partial = (partial + (u_col[int(k)] * L[int(i), int(k)]))
            new_val = ((A[int(i), int(j)] - partial) / u_col[int(j)])
            l_col_next = torch.zeros(int(n_size))
            for c in range(int(0), int(n_size)):
                if c == i:
                    l_col_next[int(c)] = new_val
                else:
                    l_col_next[int(c)] = l_col[int(c)]
            l_col = l_col_next
        U_next = torch.zeros(int(n_size), int(n_size))
        L_next2 = torch.zeros(int(n_size), int(n_size))
        for r in range(int(0), int(n_size)):
            for c in range(int(0), int(n_size)):
                if c == j:
                    U_next[int(r), int(c)] = u_col[int(r)]
                    L_next2[int(r), int(c)] = l_col[int(r)]
                else:
                    U_next[int(r), int(c)] = U[int(r), int(c)]
                    L_next2[int(r), int(c)] = L[int(r), int(c)]
        U = U_next
        L = L_next2
    Pb = torch.zeros(int(n_size))
    for i in range(int(0), int(n_size)):
        sum_val = 0.0
        for j in range(int(0), int(n_size)):
            sum_val = (sum_val + (P[int(i), int(j)] * b[int(j)]))
        Pb[int(i)] = sum_val
    y = torch.zeros(int(n_size))
    for i in range(int(0), int(n_size)):
        total = Pb[int(i)]
        for j in range(int(0), int(i)):
            total = (total - (L[int(i), int(j)] * y[int(j)]))
        solved_val = (total / L[int(i), int(i)])
        y_next = torch.zeros(int(n_size))
        for c in range(int(0), int(n_size)):
            if c == i:
                y_next[int(c)] = solved_val
            else:
                y_next[int(c)] = y[int(c)]
        y = y_next
    results = torch.zeros(int(n_size))
    for i in range(int(0), int(n_size)):
        idx = ((n_size - 1) - i)
        total = y[int(idx)]
        for j in range(int((idx + 1)), int(n_size)):
            total = (total - (U[int(idx), int(j)] * results[int(j)]))
        solved_val = (total / U[int(idx), int(idx)])
        results_next = torch.zeros(int(n_size))
        for c in range(int(0), int(n_size)):
            if c == idx:
                results_next[int(c)] = solved_val
            else:
                results_next[int(c)] = results[int(c)]
        results = results_next
    return results

# === Program ===
A = torch.tensor([[(-1), 0, 3], [2, 1, 3], [1, 1, 2]], device=DEVICE)
b = torch.tensor([1, 2, 3], device=DEVICE)
print(lu_decomposition(A, b))