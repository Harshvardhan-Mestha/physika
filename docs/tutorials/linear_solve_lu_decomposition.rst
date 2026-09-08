Linear solve using LU Decomposition
=====================================

In this tutorial we will implement the LU decomposition method in Physika. It is also recommended to read the :doc:`Gaussian elimination tutorial <linear_solve_gaussian_elimination>`,
as the concepts of pivoting and row operations explained in that tutorial will help make the LU decomposition method easier to understand.


What is LU decomposition?
---------------------------


LU decomposition, also known as LU factorization method which factors a square matrix :math:`A` into the product of two simpler matrices,
a lower triangular matrix :math:`L` and  and an upper triangular matrix :math:`U`. [WikipediaLU]_
For this tutorial, the Doolittle algorithm will be used to perform LU decomposition. This method provides an alternative way to factor :math:`A` without
going through the cumbersome steps of Gaussian elimination, which will be explained in the sections below.
The matrix :math:`A` is decomposed into a lower triangular matrix :math:`L` and an upper triangular matrix :math:`U`. Together, these matrices give the following equation: [GraphoeLU]_

.. math::

    LU = PA \label{lu_equation}


where,

- ``L`` is lower triangular matrix where diagonal entries are 1.
- ``U`` is upper triangular matrix contains the pivot rows after elimination.
- ``P`` is permutation matrix (initialized as an identity matrix,which will record row swaps performed for partial pivoting)
- ``A`` is the square matrix which we are going to solve to find ``L`` and ``U``.


This tutorial further gets divided into two main sections. In first section we will solve example matrix :math:`A` numerically to learn how 
LU decomposition method works and in second section we will implement the method in Physika.

The Equation
------------

Following is the linear equations which we are going to solve:

.. math::
 
   \begin{aligned}
   -x + 3z &= 1 \\
   2x + y + 3z &= 2 \\
   x + y + 2z &= 3
   \end{aligned}
   \label{linear_equation_form}

First we will write this equations in form of ``Ax = B`` which is as follow:

.. math::

    \underset{A}{
    \begin{bmatrix}
    -1 & 0 & 3 \\
    2 & 1 & 3 \\
    1 & 1 & 2
    \end{bmatrix}}
    \underset{x}{
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}}
    =
    \underset{B}{
    \begin{bmatrix}
    1 \\
    2 \\
    3
    \end{bmatrix}}
    \label{ax_equation}

In physika we define this matrices such as:

.. code-block:: text

    A: ℝ[3, 3] = [
        [-1, 0, 3],
        [2, 1, 3],
        [1, 1, 2]
    ]

    b: ℝ[3] = [1, 2, 3]


Section 1: Solve numerically
-----------------------------


In this section we will solve numerically to understand the concept of LU decomposition,
now lets put :math:`A` in equation :math:`\eqref{lu_equation}` which gives us:

.. math::

    L U = P \begin{bmatrix}
    -1.0 & 0.0 & 3.0 \\
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}

Since we already know that :math:`P` is an Identity matrix, lets substitute matrix :math:`P`, which will gives us:

.. math::

    L U = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    -1.0 & 0.0 & 3.0 \\
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}

Now the goal is to find :math:`L` and :math:`U` matrices.

So lets start by finding values of matrix :math:`U` through elimination method, we did same thing in gaussian elimination tutorial also, but 
for this case we are going to save the multipliers which we use for row operations.

Calculate :math:`U` matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Since the goals is to use :math:`A` and convert it into Upper triangular matrix :math:`U`, We will make zeros under the pivot row which is diagonal row of matrix :math:`A` (highlighted by red boxes)


.. math::

    A = \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{-1.0} & 0.0 & 3.0 \\
    2.0 & \bbox[2pt, border: 1.5pt solid red]{1.0} & 3.0 \\
    1.0 & 1.0 & \bbox[2pt, border: 1.5pt solid red]{2.0}
    \end{bmatrix}

We start with column 0, particularly from values under pivot value of that column, so for first column pivot value is `-1.0` (highlighted by red box).

.. math::

    \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{-1.0} \\
    2.0 \\
    1.0
    \end{bmatrix}

Here you can see that absolute value of ``A[0, 1]`` which is `2` is higher than absolute value of ``A[0, 0]`` which is `-1.0` (pivot value), we have to swap f  irst row with second row which
will give us:

.. math::

   A = \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
   -1.0 & \bbox[2pt, border: 1.5pt solid red]{0.0} & 3.0 \\
   1.0 & 1.0 & \bbox[2pt, border: 1.5pt solid red]{2.0}
   \end{bmatrix}

Also remember to do the same row swapping in matrix :math:`P`, which will gives us:


.. math::

    L U = \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
    -1.0 & 0.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}


Now, lets make values under pivot value (marked as red box) in column 1 zeros, for that we have to perform row operations on Row-2 and Row-3

.. math::

    R_2 \leftarrow R_2 + 0.5 R_1 \label{first_row_operation}


.. math::

    R_3 \leftarrow R_3 - 0.5 R_1 \label{second_row_operation}

After this matrix :math:`U` becomes:

.. math::

   U = \begin{bmatrix}
   \bbox[2pt, border: 1.5pt solid red]{2.0} & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & \bbox[2pt, border: 1.5pt solid red]{0.5} & 0.5 \\
   \color{green}{\mathbf{0.0}} & 0.5 & \bbox[2pt, border: 1.5pt solid red]{4.5}
   \end{bmatrix}


Now lets make zero under second pivot value which is at ``A[1, 1]`` which is in second row as 0.5, for that we will perform following row operation:

.. math::

    R_3 \leftarrow R_3 - 1 R_2 \label{third_row_operation}

After this, we finally get our matrix :math:`U` which is:

.. math::

   U = \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & 0.5 & 0.5 \\
   \color{green}{\mathbf{0.0}} & \color{green}{\mathbf{0.0}} & -4.0
   \end{bmatrix}

Lets put the matrix :math:`U`, :math:`P` and :math:`A` into equation :math:`\eqref{lu_equation}`

.. math::

   L
   \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   \color{green}{\mathbf{0.0}} & 0.5 & 0.5 \\
   \color{green}{\mathbf{0.0}} & \color{green}{\mathbf{0.0}} & -4.0
   \end{bmatrix}
   =
   \begin{bmatrix}
   0 & 1 & 0 \\
   1 & 0 & 0 \\
   0 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   2.0 & 1.0 & 3.0 \\
   -1.0 & 0.0 & 3.0 \\
   1.0 & 1.0 & 2.0
   \end{bmatrix}


Calculate :math:`L` matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At beginning the :math:`L` matrix is initialized as Lower Triangular matrix with all one's in diagonal row, which gives us:

.. math::

   L = \begin{bmatrix}
   1.0 & 0.0 & 0.0 \\
   l_{21} & 1.0 & 0.0 \\
   l_{31} & l_{32} & 1.0
   \end{bmatrix}

Now, the unknown entries :math:`l_{21}`, :math:`l_{31}`, and :math:`l_{32}` are simply the exact multipliers we used during the row operations in equations :math:`\eqref{first_row_operation}`
, :math:`\eqref{second_row_operation}` and :math:`\eqref{third_row_operation}` in elimination step which are:

.. math::

   \begin{aligned}
   R_2 &\leftarrow R_2 + \bbox[2pt, border: 1.5pt solid red]{0.5} R_1 \\
   R_3 &\leftarrow R_3 - \bbox[2pt, border: 1.5pt solid red]{0.5} R_1 \\
   R_3 &\leftarrow R_3 - \bbox[2pt, border: 1.5pt solid red]{1.0} R_2
   \end{aligned}

In above equation, multipliers are denoted by red boxes which are values of our :math:`L` matrix:

.. math::

   L = \begin{bmatrix}
   1.0 & 0.0 & 0.0 \\
   \color{green}{\mathbf{-0.5}} & 1.0 & 0.0 \\
   \color{green}{\mathbf{0.5}} & \color{green}{\mathbf{1.0}} & 1.0
   \end{bmatrix}

Check correctness of LU = PA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now lets put :math:`L` in equation :math:`\eqref{lu_equation}` which gives us:

.. math::

    \begin{bmatrix}
    1.0 & 0.0 & 0.0 \\
    0.5 & 1.0 & 0.0 \\
    -0.5 & 1.0 & 1.0
    \end{bmatrix}
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    0.0 & 0.5 & 0.5 \\
    0.0 & 0.0 & -4.0
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    -1.0 & 0.0 & 3.0 \\
    1.0 & 1.0 & 2.0
    \end{bmatrix}


Now we can also check correctness of our answer by multiplying matrix :math:`L` with 


.. math::

    L \cdot U = P \cdot A


.. math::

    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0 \\
    -1.0 & 0.0 & 3.0
    \end{bmatrix}
    =
    \begin{bmatrix}
    2.0 & 1.0 & 3.0 \\
    1.0 & 1.0 & 2.0 \\
    -1.0 & 0.0 & 3.0
    \end{bmatrix}

Solving Linear System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have verified that

.. math::

   LU = PA

we can use this decomposition to solve our linear system of the form

.. math::

   Ax = b

Since the decomposition contains the permutation matrix ``P``, we multiply
both sides by ``P``:

.. math::

   PAx = Pb

and since, :math:`PA = LU`, we can have:

.. math::

   LUx = Pb


Now lets calculate :math:`Pb` which is just matmul operation between :math:`P` and :math:`b`
We have


.. math::

    P =
    \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}

.. math::

    b =
    \begin{bmatrix}
    1.0 \\
    2.0 \\
    3.0
    \end{bmatrix}

Therefore,

.. math::

    Pb =
    \begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    1.0 \\
    2.0 \\
    3.0
    \end{bmatrix}

Multiplying the matrices gives

.. math::

    Pb =
    \begin{bmatrix}
    2.0 \\
    1.0 \\
    3.0
    \end{bmatrix}

Therefore, our system becomes

.. math::

    LUx =
    \begin{bmatrix}
    2.0 \\
    1.0 \\
    3.0
    \end{bmatrix}

At this point, instead of trying to solve :math:`LUx`` directly, we can
use the fact that :math:`L` and :math:`U` are triangular matrices.

We first solve a system involving :math:`L`, and then use its result to
solve a system involving :math:`U`.



Forward substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create an intermediate vector :math:`y` such that 

.. math:: 
    
    y = Ux

Therefore,

.. math:: 

    LUx = Ly

and our equation:

.. math::

    Lux = Pb

becomes:

.. math::

    Ly = Pb


We already calculated 

.. math:: 
    Pb = 
    \begin{bmatrix} 
    2.0 \\
    1.0 \\
    3.0
    \end{bmatrix}

Therefore, we need to solve

.. math:: 
    
    \begin{bmatrix}
    1.0 & 0.0 & 0.0 \\
    -0.5 & 1.0 & 0.0 \\
    0.5 & 1.0 & 1.0 
    \end{bmatrix}
    \begin{bmatrix} 
    y_1 \\
    y_2 \\
    y_3 
    \end{bmatrix}
    = 
    \begin{bmatrix}
    2.0 \\
    1.0 \\
    3.0 
    \end{bmatrix}

Since :math:`L` is a lower triangular matrix, we can solve the equations from top to bottom.
From the first row:

.. math::

    y_1 = 2.0


For the second row:

.. math::

    -0.5y_1 + y_2 = 1.0

Substituting :math:`y_1 = 2.0`:

.. math::

    -0.5(2.0) + y_2 = 1.0 \\
    y_2 = 2.0

Finally, from the third row:

.. math::

    0.5y_1 + y_2 + y_3 = 3.0 \\
    0.5(2.0) + 2.0 + y_3 = 3.0 \\
    y_3 = 0.0


Therefore, 

.. math::

    y =
    \begin{bmatrix}
        2.0 \\
        2.0 \\
        0.0
    \end{bmatrix}

We have now completed the first triangular solve.




Backward substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We still need to find :math:`x`. Recall that we introduced :math:`y` using:

.. math::

    y = Ux

Substituting our matrices gives 

.. math::

    \underbrace{
        \begin{bmatrix}
            2.0 & 1.0 & 3.0 \\
            0.0 & 0.5 & 4.5 \\
            0.0 & 0.0 & -4.0 
        \end{bmatrix}
    }_{U}
    \underbrace{
        \begin{bmatrix} 
            x_1 \\
            x_2 \\
            x_3
        \end{bmatrix}
    }_{x}
    =
    \underbrace{
        \begin{bmatrix}
            2.0 \\
            2.0 \\
            0.0
        \end{bmatrix}
    }_{y}

Since :math:`U` is an upper triangular matrix, we solve the equations from bottom to top.
Starting with the third row:

.. math::

    -4.0x_3 = 0.0 \\
    x_3 = 0.0

Now consider the second row:

.. math::

    0.5x_2 + 4.5x_3 = 2.0 \\
    0.5x_2 + 4.5(0.0) = 2.0 \\
    x_2 = 4.0

Finally, consider the first row:

.. math::

    2.0x_1 + x_2 + 3.0x_3 = 2.0 \\
    2.0x_1 + 4.0 + 3.0(0.0) = 2.0 \\
    x_1 = -1.0


Hence, the final solution is 

.. math::

    \begin{bmatrix}
        x \\
        y \\
        z
    \end{bmatrix}
    =
    \begin{bmatrix}
        -1.0 \\
        4.0 \\
        0.0
    \end{bmatrix}




Section 2: Solve through Physika code
--------------------------------------

Before starting lets create a function as:

.. code-block:: text

    def lu_decomposition(A: ℝ[n, n], b: ℝ[n]): ℝ[n]:
    ...

- ``A: ℝ[n, n]`` - represents ``A`` matrix.
- ``b: ℝ[n]`` - represents ``b`` matrix.


.. code-block:: text
    
    n_size: ℕ = get_2d_array_num_rows(A)
    
    # Create P, L and U matrices
    P: ℝ[n_size, n_size] = zeros(n_size, n_size)
    for i:ℕ(n_size):
        P[i, i] = 1.0
    L: ℝ[n_size, n_size] = zeros(n_size, n_size)
    U: ℝ[n_size, n_size] = zeros(n_size, n_size)


- ``n_size`` - is total number of rows to loop.
- ``L``, ``U``, ``P`` - represent the Lower triangular matrix, Upper triangular matrix, and Permutation matrix, respectively.

.. code-block:: text

    for j:ℕ(n_size):
        # ------------------------------------
        # Find pivot value
        # ------------------------------------
        max_row = j
        for i:ℕ(j+1, n_size):
            if abs(A[i, j]) > abs(A[max_row, j]):
                max_row = i
        # ------------------------------------
        # swap row if max_row got changed
        # ------------------------------------
        A_next = zeros(n_size, n_size)
        L_next = zeros(n_size, n_size)
        P_next = zeros(n_size, n_size)
        for r: ℕ(n_size):
            source = r
            if r == j:
                source = max_row
            else:
                if r == max_row:
                    source = j  
            for c: ℕ(n_size):
                A_next[r, c] = A[source, c]
                L_next[r, c] = L[source, c]
                P_next[r, c] = P[source, c]
        A = A_next
        L = L_next
        P = P_next


This is similar to what we did in Gaussian elimination tutorial, so In this outer loop we iterate column :math:`j` ``(j = 0, 1, ..., n_size-1)`` to perform
partial pivoting, we first scan down column :math:`j` to find the row with maximum absolute value, if we find a larger pivot value we swap the rows in matrix :math:`P` and :math:`A` .


Next, in the same outer loop code after partial pivoting we start to build our Upper triangular matrix :math:`U`, same approach we did in gaussian elimination,
and also the Lower triangular matrix :math:`L`:

.. math::

    U_{ij} = A_{ij} - \sum_{k=0}^{i-1} L_{ik}\,U_{kj}, \qquad 0 \le i \le j

.. math::

    L_{ij} = \frac{A_{ij} - \displaystyle\sum_{k=0}^{j-1} L_{ik}\,U_{kj}}{U_{jj}}, \qquad j \le i \le n-1

.. code-block:: text

    # ------------------------------------
    # Update U and L matrices
    # ------------------------------------
    u_col = zeros(n_size)
    for i: ℕ(j + 1):
        partial = 0.0
        for k: ℕ(i):
            partial = partial + u_col[k] * L[i, k]
        new_val = A[i, j] - partial
        u_col_next = zeros(n_size)
        for c: ℕ(n_size):
            if c == i:
                u_col_next[c] = new_val
            else:
                u_col_next[c] = u_col[c]
        u_col = u_col_next
    # Compute column `j` of L in same way
    l_col = zeros(n_size)
    for i: ℕ(j, n_size):
        partial = 0.0
        for k: ℕ(j):
            partial = partial + u_col[k] * L[i, k]
        new_val = (A[i, j] - partial) / u_col[j]
        l_col_next = zeros(n_size)
        for c: ℕ(n_size):
            if c == i:
                l_col_next[c] = new_val
            else:
                l_col_next[c] = l_col[c]
        l_col = l_col_next
    # Merge finished columns of U and L
    U_next = zeros(n_size, n_size)
    L_next2 = zeros(n_size, n_size)
    for r: ℕ(n_size):
        for c: ℕ(n_size):
            if c == j:
                U_next[r, c] = u_col[r]
                L_next2[r, c] = l_col[r]
            else:
                U_next[r, c] = U[r, c]
                L_next2[r, c] = L[r, c]
    U = U_next
    L = L_next2


Therefore, once we get our :math:`L`, :math:`U` and :math:`P` matrices we can solve the linear
equation in form of :math:`Ax = b`

We start with matrix multiplication of :math:`P` and :math:`b`

.. code-block:: text

    # --------------------------------
    # P @ b
    # --------------------------------
    Pb: ℝ[n_size] = zeros(n_size)
    for i:ℕ(n_size):
        sum_val = 0.0
        for j:ℕ(n_size):
            sum_val = sum_val + P[i, j] * b[j]
        Pb[i] = sum_val


after that we can perform Forward and Backward substitution as:

.. code-block:: text

    # --------------------------------
    # Forward substitution
    # --------------------------------
    y: ℝ[n_size] = zeros(n_size)
    for i: ℕ(n_size):
        total = Pb[i]
        for j: ℕ(i):
            total = total - L[i, j] * y[j]
        solved_val = total / L[i, i]
        y_next = zeros(n_size)
        for c: ℕ(n_size):
            if c == i:
                y_next[c] = solved_val
            else:
                y_next[c] = y[c]
        y = y_next
    # --------------------------------
    # Back substitution
    # --------------------------------
    results: ℝ[n_size] = zeros(n_size)
    for i: ℕ(n_size):
        idx = n_size - 1 - i
        total = y[idx]
        for j: ℕ(idx + 1, n_size):
            total = total - U[idx, j] * results[j]
        solved_val = total / U[idx, idx]
        results_next = zeros(n_size)
        for c: ℕ(n_size):
            if c == idx:
                results_next[c] = solved_val
            else:
                results_next[c] = results[c]
        results = results_next


Here is the full ``lu_decomposition`` function:

.. code-block:: text

    def lu_decomposition(A: ℝ[n, n], b: ℝ[n]): ℝ[n]:
        n_size: ℕ = get_2d_array_num_rows(A)
        
        # Create P, L and U matrices
        P: ℝ[n_size, n_size] = zeros(n_size, n_size)
        for i:ℕ(n_size):
            P[i, i] = 1.0
        L: ℝ[n_size, n_size] = zeros(n_size, n_size)
        U: ℝ[n_size, n_size] = zeros(n_size, n_size)
        
        for j:ℕ(n_size):
            # ------------------------------------
            # Find pivot value
            # ------------------------------------
            max_row = j
            for i:ℕ(j+1, n_size):
                if abs(A[i, j]) > abs(A[max_row, j]):
                    max_row = i
            # ------------------------------------
            # swap row if max_row got changed
            # ------------------------------------
            A_next = zeros(n_size, n_size)
            L_next = zeros(n_size, n_size)
            P_next = zeros(n_size, n_size)
            for r: ℕ(n_size):
                source = r
                if r == j:
                    source = max_row
                else:
                    if r == max_row:
                        source = j  
                for c: ℕ(n_size):
                    A_next[r, c] = A[source, c]
                    L_next[r, c] = L[source, c]
                    P_next[r, c] = P[source, c]
            A = A_next
            L = L_next
            P = P_next
            # ------------------------------------
            # Update U and L matrices
            # ------------------------------------
            u_col = zeros(n_size)
            for i: ℕ(j + 1):
                partial = 0.0
                for k: ℕ(i):
                    partial = partial + u_col[k] * L[i, k]
                new_val = A[i, j] - partial
                u_col_next = zeros(n_size)
                for c: ℕ(n_size):
                    if c == i:
                        u_col_next[c] = new_val
                    else:
                        u_col_next[c] = u_col[c]
                u_col = u_col_next
            # Compute column `j` of L in same way
            l_col = zeros(n_size)
            for i: ℕ(j, n_size):
                partial = 0.0
                for k: ℕ(j):
                    partial = partial + u_col[k] * L[i, k]
                new_val = (A[i, j] - partial) / u_col[j]
                l_col_next = zeros(n_size)
                for c: ℕ(n_size):
                    if c == i:
                        l_col_next[c] = new_val
                    else:
                        l_col_next[c] = l_col[c]
                l_col = l_col_next
            # Merge finished columns of U and L
            U_next = zeros(n_size, n_size)
            L_next2 = zeros(n_size, n_size)
            for r: ℕ(n_size):
                for c: ℕ(n_size):
                    if c == j:
                        U_next[r, c] = u_col[r]
                        L_next2[r, c] = l_col[r]
                    else:
                        U_next[r, c] = U[r, c]
                        L_next2[r, c] = L[r, c]
            U = U_next
            L = L_next2
        # --------------------------------
        # P @ b
        # --------------------------------
        Pb: ℝ[n_size] = zeros(n_size)
        for i:ℕ(n_size):
            sum_val = 0.0
            for j:ℕ(n_size):
                sum_val = sum_val + P[i, j] * b[j]
            Pb[i] = sum_val
        # --------------------------------
        # Forward substitution
        # --------------------------------
        y: ℝ[n_size] = zeros(n_size)
        for i: ℕ(n_size):
            total = Pb[i]
            for j: ℕ(i):
                total = total - L[i, j] * y[j]
            solved_val = total / L[i, i]
            y_next = zeros(n_size)
            for c: ℕ(n_size):
                if c == i:
                    y_next[c] = solved_val
                else:
                    y_next[c] = y[c]
            y = y_next
        # --------------------------------
        # Back substitution
        # --------------------------------
        results: ℝ[n_size] = zeros(n_size)
        for i: ℕ(n_size):
            idx = n_size - 1 - i
            total = y[idx]
            for j: ℕ(idx + 1, n_size):
                total = total - U[idx, j] * results[j]
            solved_val = total / U[idx, idx]
            results_next = zeros(n_size)
            for c: ℕ(n_size):
                if c == idx:
                    results_next[c] = solved_val
                else:
                    results_next[c] = results[c]
            results = results_next
        return results



We can test this function with:

.. code-block:: text

    A: ℝ[3, 3] = [
        [-1, 0, 3],
        [2, 1, 3],
        [1, 1, 2]
    ]
    b: ℝ[3] = [1, 2, 3]

    lu_decomposition(A, b)



Full code
-------------

.. code-block:: text

    # Helper function
    def get_2d_array_num_rows(x: ℝ[m, n]): ℕ:
        total: ℕ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total


    def lu_decomposition(A: ℝ[n, n], b: ℝ[n]): ℝ[n]:
        n_size: ℕ = get_2d_array_num_rows(A)
        
        # Create P, L and U matrices
        P: ℝ[n_size, n_size] = zeros(n_size, n_size)
        for i:ℕ(n_size):
            P[i, i] = 1.0
        L: ℝ[n_size, n_size] = zeros(n_size, n_size)
        U: ℝ[n_size, n_size] = zeros(n_size, n_size)
        
        for j:ℕ(n_size):
            # ------------------------------------
            # Find pivot value
            # ------------------------------------
            max_row = j
            for i:ℕ(j+1, n_size):
                if abs(A[i, j]) > abs(A[max_row, j]):
                    max_row = i
            # ------------------------------------
            # swap row if max_row got changed
            # ------------------------------------
            A_next = zeros(n_size, n_size)
            L_next = zeros(n_size, n_size)
            P_next = zeros(n_size, n_size)
            for r: ℕ(n_size):
                source = r
                if r == j:
                    source = max_row
                else:
                    if r == max_row:
                        source = j  
                for c: ℕ(n_size):
                    A_next[r, c] = A[source, c]
                    L_next[r, c] = L[source, c]
                    P_next[r, c] = P[source, c]
            A = A_next
            L = L_next
            P = P_next
            # ------------------------------------
            # Update U and L matrices
            # ------------------------------------
            u_col = zeros(n_size)
            for i: ℕ(j + 1):
                partial = 0.0
                for k: ℕ(i):
                    partial = partial + u_col[k] * L[i, k]
                new_val = A[i, j] - partial
                u_col_next = zeros(n_size)
                for c: ℕ(n_size):
                    if c == i:
                        u_col_next[c] = new_val
                    else:
                        u_col_next[c] = u_col[c]
                u_col = u_col_next
            # Compute column `j` of L in same way
            l_col = zeros(n_size)
            for i: ℕ(j, n_size):
                partial = 0.0
                for k: ℕ(j):
                    partial = partial + u_col[k] * L[i, k]
                new_val = (A[i, j] - partial) / u_col[j]
                l_col_next = zeros(n_size)
                for c: ℕ(n_size):
                    if c == i:
                        l_col_next[c] = new_val
                    else:
                        l_col_next[c] = l_col[c]
                l_col = l_col_next
            # Merge finished columns of U and L
            U_next = zeros(n_size, n_size)
            L_next2 = zeros(n_size, n_size)
            for r: ℕ(n_size):
                for c: ℕ(n_size):
                    if c == j:
                        U_next[r, c] = u_col[r]
                        L_next2[r, c] = l_col[r]
                    else:
                        U_next[r, c] = U[r, c]
                        L_next2[r, c] = L[r, c]
            U = U_next
            L = L_next2
        # --------------------------------
        # P @ b
        # --------------------------------
        Pb: ℝ[n_size] = zeros(n_size)
        for i:ℕ(n_size):
            sum_val = 0.0
            for j:ℕ(n_size):
                sum_val = sum_val + P[i, j] * b[j]
            Pb[i] = sum_val
        # --------------------------------
        # Forward substitution
        # --------------------------------
        y: ℝ[n_size] = zeros(n_size)
        for i: ℕ(n_size):
            total = Pb[i]
            for j: ℕ(i):
                total = total - L[i, j] * y[j]
            solved_val = total / L[i, i]
            y_next = zeros(n_size)
            for c: ℕ(n_size):
                if c == i:
                    y_next[c] = solved_val
                else:
                    y_next[c] = y[c]
            y = y_next
        # --------------------------------
        # Back substitution
        # --------------------------------
        results: ℝ[n_size] = zeros(n_size)
        for i: ℕ(n_size):
            idx = n_size - 1 - i
            total = y[idx]
            for j: ℕ(idx + 1, n_size):
                total = total - U[idx, j] * results[j]
            solved_val = total / U[idx, idx]
            results_next = zeros(n_size)
            for c: ℕ(n_size):
                if c == idx:
                    results_next[c] = solved_val
                else:
                    results_next[c] = results[c]
            results = results_next
        return results



    A: ℝ[3, 3] = [
        [-1, 0, 3],
        [2, 1, 3],
        [1, 1, 2]
    ]

    b: ℝ[3] = [1, 2, 3]


    lu_decomposition(A, b)



References
----------

.. [WikipediaLU] Wikipedia contributors, *LU decomposition*, Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/wiki/LU_decomposition
.. [GraphoeLU] Graphoe, *LU Factorization - Numerical Methods for Linear Systems*, Graphoe Resources. https://graphoe.com/resources/numerical-methods/linear-system/lu-factorization