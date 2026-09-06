Deep Equilibrium Models
=======================

This tutorial introduces Deep Equilibrium Models (DEQs) and shows how to implement one in Physika.
Suppose we want the representational power of a very deep network, but without paying to store and backpropagate through every one of its layers.
A Deep Equilibrium Model gives us exactly this: instead of stacking many distinct layers, it applies **one** layer over and over until its output stops changing, and treats that settled value as the network's answer.
That settled value is called an *equilibrium* (or *fixed point*), and finding it turns the forward pass of a neural network into a **root-finding problem**.

By the end of this tutorial you will understand what a fixed point is, why a weight-tied "infinite depth" network can be summarized by one, how to solve for that fixed point with a quasi-Newton root finder, and how it is differentiated.
You will then train a small DEQ that reconstructs handwritten digits from the MNIST dataset.
This tutorial is based on Bai, Kolter, and Koltun's *Deep Equilibrium Models* [BaiDEQ2019]_ and the Deep Implicit Layers tutorial [ImplicitLayers]_.


What are Deep Equilibrium Models?
---------------------------------

A conventional deep network computes a sequence of hidden states, one per layer:

.. math::
    h_1 = f_1(h_0, x), \quad h_2 = f_2(h_1, x), \quad \ldots, \quad h_T = f_T(h_{T-1}, x)

Each layer :math:`f_t` usually has its own parameters, and the memory needed for training grows with the number of layers :math:`T`, because every intermediate :math:`h_t` must be kept for the backward pass.

A Deep Equilibrium Model makes two changes.
First, it **ties the weights**: every layer is the *same* function :math:`f(\cdot, x, \theta)`.
Second, it asks what happens as the depth goes to infinity.
If repeatedly applying :math:`f` drives the hidden state toward a value that no longer changes, then that limiting value :math:`h^\star` satisfies

.. math::
    h^\star = f(h^\star, x, \theta).

A point that maps to itself under :math:`f` is called a **fixed point** (or **equilibrium point**).
Rather than run :math:`f` a fixed number of times, a DEQ directly *solves* for this fixed point.
The infinite stack of identical layers is replaced by a single object — the equilibrium — and the entire forward pass becomes "find the :math:`h^\star` that :math:`f` leaves unchanged" [BaiDEQ2019]_.

.. figure:: /_static/tutorial_files/deq/deq_fixed_point.png
   :alt: An infinitely deep weight-tied network whose hidden state converges to a single fixed point h-star, drawn as a spiral settling onto a point.
   :align: center
   :width: 500px

   **Figure 1.** *(placeholder)* A weight-tied network applies the same layer :math:`f(\cdot, x, \theta)` repeatedly. As depth grows, the hidden state settles onto an equilibrium :math:`h^\star` that satisfies :math:`h^\star = f(h^\star, x, \theta)`.

Setup and Notation
^^^^^^^^^^^^^^^^^^

We work with three objects throughout.

The **input** :math:`x \in \mathbb{R}^{d}` is the data the network is given (for us, a flattened :math:`28 \times 28 = 784`-dimensional MNIST image, a vector of shape :math:`(d,)`).

The **hidden state** :math:`h \in \mathbb{R}^{n}` is the internal representation the network refines, a vector of shape :math:`(n,)`.

The **parameters** :math:`\theta` collect every learnable weight and bias in the layer.
In our model :math:`\theta = \{W, U, b, W_o, b_o\}`.

The layer itself is a function :math:`f: \mathbb{R}^{n} \times \mathbb{R}^{d} \to \mathbb{R}^{n}`, which takes the current hidden state and the (fixed) input and returns the next hidden state.
The concrete choice used in this tutorial is a single fully connected layer with a :math:`\tanh` nonlinearity:

.. math::
    f(h, x, \theta) = \tanh\!\left(h\,W + x\,U + b\right).

Here :math:`W \in \mathbb{R}^{n \times n}` mixes the hidden state with itself, :math:`U \in \mathbb{R}^{d \times n}` injects the input, and :math:`b \in \mathbb{R}^{1 \times n}` is a bias.
Note that :math:`x` enters :math:`f` but never changes while we iterate — it is a constant *drive* term that anchors the equilibrium.

.. note::

   This tutorial and the code name the hidden state :math:`h` and its equilibrium :math:`h^\star`. Some references (including the Deep Implicit Layers tutorial [ImplicitLayers]_) write the same quantities as :math:`z` and :math:`z^\star`. They mean the same thing; we use :math:`h` consistently.

Fixed Points and the Banach Fixed-Point Theorem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **fixed point** of a function :math:`g` is any input that the function returns unchanged: a point :math:`h^\star` with :math:`g(h^\star) = h^\star`.
For a DEQ, the function is :math:`f(\cdot, x, \theta)` with :math:`x` and :math:`\theta` held fixed, and the equilibrium is a fixed point of that map.

Two questions immediately arise: does such a point exist, and is it unique?
The **Banach fixed-point theorem** (also called the contraction mapping theorem) answers both under one condition [Wikipedia_Banach]_.

We first need the notion of a **contraction**.
A map :math:`f` is a contraction if it always brings pairs of points *closer together* by at least a constant factor: there exists a **Lipschitz constant** :math:`L < 1` such that

.. math::
    \left\| f(a, x, \theta) - f(b, x, \theta) \right\| \le L \, \left\| a - b \right\| \qquad \text{for all } a, b,

where :math:`\|\cdot\|` denotes the Euclidean distance between two vectors.
Intuitively, applying a contraction shrinks distances, so it cannot spread points apart.

The Banach fixed-point theorem states that a contraction on a complete space has **exactly one** fixed point :math:`h^\star`.
It also says the plain iteration :math:`h_{k+1} = f(h_k, x, \theta)` — called **Picard iteration** — converges to :math:`h^\star` from any start, with error shrinking geometrically as :math:`L^k`.
So a contractive layer guarantees the "infinitely deep weight-tied network" is a meaningful object: the equilibrium exists and is unique.

Our layer makes this easy to arrange. Because :math:`\tanh` is *1-Lipschitz* (its slope never exceeds :math:`1`), the layer satisfies :math:`\|f(a,x,\theta) - f(b,x,\theta)\| \le \|W\|\,\|a - b\|`, so keeping :math:`\|W\|` below :math:`1` makes :math:`f` a contraction.
This is why the weights are initialized small: it keeps the equilibrium unique and the solver well behaved.

Solving for the Equilibrium (the Forward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finding :math:`h^\star` is a **root-finding problem**.
Define the *residual*

.. math::
    g_\theta(h) = f(h, x, \theta) - h.

The equilibrium is exactly the value that makes the residual vanish, :math:`g_\theta(h^\star) = 0`.

The solver used below is a **quasi-Newton** method.
Full **Newton's method** would use the residual Jacobian :math:`\mathcal{J}_g = \partial f/\partial h - I` to take curvature-aware steps,

.. math::
    h_{k+1} = h_k - \mathcal{J}_g(h_k)^{-1}\, g_\theta(h_k),

converging in very few iterations but re-forming and re-inverting the :math:`n \times n` Jacobian every step.
The implementation uses the simplest quasi-Newton variant, **modified Newton**: it forms and inverts the residual Jacobian *once* at the starting iterate and reuses that inverse for every step,

.. math::
    h_{k+1} = h_k - \mathcal{J}_g(h_0)^{-1}\, g_\theta(h_k).

Two features of Physika make this read almost exactly like the math.
First, the residual Jacobian is not derived by hand — it is read straight off automatic differentiation, ``grad(g, h)``, which returns the full matrix :math:`\partial f/\partial h - I`.
Second, the inverse is applied with a small ``inv`` helper (built in the next section), so the whole step is ``h - g @ inv(grad(g, h))``.
For comparison, the simplest solver of all is Picard iteration :math:`h_{k+1} = f(h_k, x, \theta)`, which needs no Jacobian; production DEQs use stronger quasi-Newton solvers such as Broyden's method or Anderson acceleration that approximate :math:`\mathcal{J}_g^{-1}` from past iterates and scale to large :math:`n` [BaiDEQ2019]_.
Whichever solver we pick, the forward pass returns a single vector :math:`h^\star`, and the model's "depth" is however many steps the solver happened to take.

Differentiating Through the Equilibrium (the Backward Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model we need the gradient of the loss with respect to the parameters, which requires the gradient of the equilibrium :math:`h^\star` with respect to :math:`\theta`.
Differentiating the equilibrium condition itself gives a closed form, called **implicit differentiation** [BaiDEQ2019]_.
Starting from :math:`h^\star = f(h^\star, x, \theta)` and differentiating both sides with respect to :math:`\theta` (the right-hand side needs the chain rule because :math:`h^\star` depends on :math:`\theta`):

.. math::
    \frac{\partial h^\star}{\partial \theta}
    = \frac{\partial f}{\partial h^\star}\,\frac{\partial h^\star}{\partial \theta}
    + \frac{\partial f}{\partial \theta}.

Collecting the :math:`\partial h^\star/\partial \theta` terms produces the residual Jacobian :math:`\partial f/\partial h^\star - I`:

.. math::
    \left(\frac{\partial f}{\partial h^\star} - I\right)\frac{\partial h^\star}{\partial \theta}
    = -\frac{\partial f}{\partial \theta}
    \quad\Longrightarrow\quad
    \frac{\partial h^\star}{\partial \theta}
    = -\left(\frac{\partial f}{\partial h^\star} - I\right)^{-1}\frac{\partial f}{\partial \theta}.

Chaining once more with the loss gives the gradient we want:

.. math::
    \boxed{\;\frac{\partial \mathcal{L}}{\partial \theta}
    = -\,\frac{\partial \mathcal{L}}{\partial h^\star}
      \left(\frac{\partial f}{\partial h^\star} - I\right)^{-1}
      \frac{\partial f}{\partial \theta}\;}

This references :math:`h^\star` only — the solver's trajectory has dropped out entirely. Here :math:`\partial f/\partial h^\star \in \mathbb{R}^{n \times n}` is the Jacobian of the layer at the equilibrium and :math:`I` is the identity.

**Worked scalar example.**
Take a one-dimensional equilibrium with :math:`f(h, x) = \tanh(w h + u x + b)` — exactly the scalar version of the layer used in the code.
Using :math:`\tanh' = 1 - \tanh^2` and :math:`h^\star = \tanh(\cdot)`, the layer derivative at the fixed point is :math:`\partial f/\partial h^\star = w\,(1 - {h^\star}^2)`, so

.. math::
    \frac{\partial h^\star}{\partial b}
    = -\bigl(w(1 - {h^\star}^2) - 1\bigr)^{-1}\,(1 - {h^\star}^2).

For instance, with :math:`w = 0.5` and :math:`h^\star = 0.4`, the layer derivative is :math:`0.42`, so :math:`-(0.42 - 1)^{-1} = 1.724` multiplies the one-layer gradient :math:`0.84`, giving :math:`\partial h^\star/\partial b \approx 1.448`.
The :math:`-(\partial f/\partial h^\star - I)^{-1}` term is what an *infinitely deep* network contributes: in the scalar case it is the geometric series :math:`1 + f' + f'^2 + \cdots = (1 - f')^{-1}`, the summed influence of the layer applied over and over.

Differentiability in Physika
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Physika class compiles to a differentiable module, and ``grad()`` backpropagates through its methods with automatic differentiation.
We do not write a custom backward pass: because the forward solver is an ordinary loop, ``grad(L, this.params)`` simply differentiates through the unrolled quasi-Newton iteration and into every parameter.

At first glance this looks like it would give the wrong answer — the solver inverts a Jacobian, so surely that inverse must be differentiated too?
It does not, and the reason is exactly the boxed formula above.
The frozen inverse :math:`\mathcal{J}_g(h_0)^{-1}` acts only as a *preconditioner*: it changes how fast the iteration converges, not where it converges to.
When the iteration has settled on :math:`h^\star`, the gradient of the unrolled loop satisfies the same linear relation as the implicit-differentiation result, and the preconditioner cancels out.
So autograd through the unrolled quasi-Newton solve recovers the exact implicit gradient :math:`-\frac{\partial \mathcal{L}}{\partial h^\star}(\partial f/\partial h^\star - I)^{-1}\frac{\partial f}{\partial \theta}`, without our ever coding that formula.
The only cost is memory: the unrolled version stores its iterates, whereas a hand-written backward using the closed form would not.


Methods for Solving the Fixed Point
-----------------------------------

The forward pass of a DEQ is only as good as the solver that produces :math:`h^\star`.
This is not an exhaustive list, but below are the solvers most commonly used, from simplest to most powerful.

1. Picard (Fixed-Point) Iteration
    The most direct solver simply iterates the layer, :math:`h_{k+1} = f(h_k, x, \theta)`.
    It requires nothing beyond evaluating :math:`f`, and by the Banach theorem it converges whenever :math:`f` is a contraction, with error shrinking by a factor :math:`L` per step.
    Its weaknesses are the flip side of its simplicity: convergence is only linear, and it diverges if :math:`f` is not a contraction.

2. Newton's Method
    Newton's method uses the residual Jacobian :math:`\mathcal{J}_g = \partial f/\partial h - I` to take much larger steps, :math:`h_{k+1} = h_k - \mathcal{J}_g(h_k)^{-1} g_\theta(h_k)`.
    Near the solution it converges *quadratically* — the number of correct digits roughly doubles each step — so it needs very few iterations, at the cost of forming and inverting the :math:`n \times n` Jacobian every step.

3. Quasi-Newton (used here; Broyden, Anderson Acceleration)
    Quasi-Newton methods keep Newton's fast convergence while avoiding the exact per-step Jacobian inverse.
    The implementation below uses the simplest such scheme, **modified Newton**: it inverts the residual Jacobian once at the initial iterate and reuses that inverse for every step.
    Production DEQs use stronger variants — **Broyden's method** maintains a low-rank running approximation of :math:`\mathcal{J}_g^{-1}`, and **Anderson acceleration** forms each iterate as a least-squares-optimal mix of the last few — both of which scale to large :math:`n` while still converging in a handful of steps [BaiDEQ2019]_.

Regardless of which solver is chosen, the backward pass is unchanged: the implicit-differentiation formula depends only on the converged :math:`h^\star`, so improving the solver never changes how gradients are computed.


Inverting a Matrix in Physika
-----------------------------

The quasi-Newton step needs to invert the :math:`n \times n` residual Jacobian, and Physika has no built-in matrix inverse, so we write a small ``inv`` of our own using **Gauss-Jordan elimination**.

The idea is to place the matrix :math:`A` next to the identity, forming the augmented block :math:`[\,A \mid I\,]`, and then apply row operations that turn the left block into the identity.
Whatever sequence of row operations reduces :math:`A` to :math:`I` also transforms the :math:`I` on the right into :math:`A^{-1}`, because those operations together *are* multiplication by :math:`A^{-1}`:

.. math::
    [\,A \mid I\,] \;\xrightarrow{\ \text{row ops}\ }\; [\,I \mid A^{-1}\,].

Concretely, we sweep the columns one at a time. For column :math:`i` we take the diagonal entry :math:`A_{ii}` as the **pivot**, divide that whole row by the pivot so the pivot becomes :math:`1`, and then subtract the right multiple of the pivot row from every *other* row so that column :math:`i` becomes zero elsewhere.
After all :math:`n` columns are processed the left block is the identity and the right block is the inverse.

Two small points make this fit the language cleanly.
First, ``eye`` builds the identity by filling a zero matrix and setting the diagonal entries to :math:`1` in a loop (no equality operator is needed).
Second, the elimination step should skip the pivot row, but a conditional would be awkward; instead we let the loop run over *every* row (which zeroes the pivot row) and then restore that row from a saved normalized copy afterward.

.. code-block:: text

    def eye(n: ℝ): ℝ[n, n]:
        I: ℝ[n, n] = for i:ℕ(n) → for j:ℕ(n) → j * 0.0
        for i:ℕ(n):
            I[i, i] = 1.0
        return I

    def inv(A: ℝ[n, n]): ℝ[n, n]:
        M: ℝ[n, n] = A * 1.0            # working copy of A, becomes the identity
        B: ℝ[n, n] = eye(n)            # starts as I, becomes A^(-1)
        for i:ℕ(n):
            p: ℝ = M[i, i]             # pivot
            mrow: ℝ[n] = M[i, :] / p   # normalized pivot row (fresh copies)
            brow: ℝ[n] = B[i, :] / p
            for j:ℕ(n):
                c: ℝ = M[j, i]         # entry to eliminate in column i
                M[j, :] = M[j, :] - c * mrow
                B[j, :] = B[j, :] - c * brow
            M[i, :] = mrow             # restore pivot row (the loop zeroed it)
            B[i, :] = brow
        return B

This version does no pivoting, so it assumes the pivots :math:`M_{ii}` are non-zero.
That holds for the matrix we invert here: because the layer is a contraction, :math:`\partial f/\partial h - I` is well conditioned.
As a quick check, ``inv([[4.0, 7.0], [2.0, 6.0]])`` returns ``[[0.6, -0.7], [-0.2, 0.4]]``, and multiplying it by the original matrix gives the identity.


Implementing a DEQ in Physika
-----------------------------

The class below contains the core of a DEQ in Physika.
It is an autoencoding DEQ: an MNIST image :math:`x` drives the layer to an equilibrium hidden state :math:`h^\star`, and a linear decoder maps :math:`h^\star` back to a :math:`784`-dimensional reconstruction :math:`\hat{x}`.
The model is trained to make :math:`\hat{x}` match :math:`x`.
It uses the ``inv`` helper from the previous section, plus the :math:`\tanh` activation written out from ``exp``:

.. code-block:: text

    def tanh(a: ℝ[p,q]): ℝ[p,q]:
        num: ℝ[p,q] = exp(a) - exp(-a)
        denom: ℝ[p,q] = exp(a) + exp(-a)
        return num / denom

The **layer** :math:`f(h, x) = \tanh(hW + xU + b)` is the ``f`` method — the one map a DEQ applies at "every depth". The code reads exactly like the equation:

.. code-block:: text

    def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
        return tanh(h @ W + x @ U + b)

The **forward pass** ``equilibrium`` is the quasi-Newton solve. It forms the residual :math:`g = f(h,x) - h`, reads the residual Jacobian :math:`\partial f/\partial h - I` straight from ``grad(g, h)``, inverts it once, and reuses that inverse. Each line matches the math one-to-one — the residual, the inverted Jacobian, the Newton step:

.. code-block:: text

    def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
        h: ℝ[1,n] = this.b * 0.0
        g: ℝ[1,n] = this.f(h, x) - h        # residual g(h)
        Jinv: ℝ[n,n] = inv(grad(g, h))      # (∂f/∂h − I)⁻¹, formed once
        for k:ℕ(10):
            g = this.f(h, x) - h
            h = h - g @ Jinv                # Newton step
        return h

The **call operator** ``λ`` runs the solver and decodes the equilibrium into data space, :math:`\hat{x} = h^\star W_o + b_o`:

.. code-block:: text

    def λ(x: ℝ[1,d]) → ℝ[1,d]:
        h_star: ℝ[1,n] = this.equilibrium(x)
        return h_star @ Wo + bo

The **loss** is the squared reconstruction error :math:`\|\,x - \hat{x}\,\|^2`; because the model is an autoencoder, the target is the input image itself.
Training is ordinary gradient descent: ``train`` computes the loss and calls ``grad(L, this.params)``, which differentiates through the unrolled solver as explained above.

.. note::

    In Physika the DEQ is differentiable end to end with no hand-written backward: gradients flow through the decoder, through the unrolled quasi-Newton solve, and into :math:`W, U, b, W_o, b_o` automatically. Because the frozen Jacobian cancels at the fixed point, this recovers the exact implicit gradient.


Training a DEQ on the MNIST Dataset
-----------------------------------

This is the complete program for training the DEQ on MNIST.

.. note::
   ``load_mnist`` is used here as a stand-in data loader that returns the first
   ``images`` MNIST digits as a ``ℝ[1000, 784]`` array of flattened, normalized
   images. If it is not available in your Physika runtime, add a small helper to
   ``physika/runtime.py`` that loads MNIST with ``torchvision`` and flattens each
   ``28×28`` image to a length-``784`` vector.


Full Code
---------

.. code-block:: text

    physika.seed(0)

    def tanh(a: ℝ[p,q]): ℝ[p,q]:
        num: ℝ[p,q] = exp(a) - exp(-a)
        denom: ℝ[p,q] = exp(a) + exp(-a)
        return num / denom

    def rand_array(n: ℝ, m: ℝ, μ: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.0, 1.0)

    def zero_array(n: ℝ, m: ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → j * 0.0

    def eye(n: ℝ): ℝ[n, n]:
        I: ℝ[n, n] = for i:ℕ(n) → for j:ℕ(n) → j * 0.0
        for i:ℕ(n):
            I[i, i] = 1.0
        return I

    def inv(A: ℝ[n, n]): ℝ[n, n]:
        M: ℝ[n, n] = A * 1.0
        B: ℝ[n, n] = eye(n)
        for i:ℕ(n):
            p: ℝ = M[i, i]
            mrow: ℝ[n] = M[i, :] / p
            brow: ℝ[n] = B[i, :] / p
            for j:ℕ(n):
                c: ℝ = M[j, i]
                M[j, :] = M[j, :] - c * mrow
                B[j, :] = B[j, :] - c * brow
            M[i, :] = mrow
            B[i, :] = brow
        return B

    class DEQ(W: ℝ[n,n], U: ℝ[d,n], b: ℝ[1,n], Wo: ℝ[n,d], bo: ℝ[1,d]):
        def f(h: ℝ[1,n], x: ℝ[1,d]): ℝ[1,n]:
            return tanh(h @ W + x @ U + b)
        def equilibrium(x: ℝ[1,d]): ℝ[1,n]:
            h: ℝ[1,n] = this.b * 0.0
            g: ℝ[1,n] = this.f(h, x) - h        # residual g(h)
            Jinv: ℝ[n,n] = inv(grad(g, h))      # (∂f/∂h − I)⁻¹, formed once
            for k:ℕ(10):
                g = this.f(h, x) - h
                h = h - g @ Jinv                # Newton step
            return h
        def λ(x: ℝ[1,d]) → ℝ[1,d]:
            h_star: ℝ[1,n] = this.equilibrium(x)
            return h_star @ Wo + bo
        def loss(target: ℝ[1,784], x_hat: ℝ[1,784]): ℝ:
            diff: ℝ[1,784] = target - x_hat
            return sum(diff * diff)
        def train(X: ℝ[1000,784], epochs: ℕ, lr: ℝ, images: ℝ):
            for epoch:ℕ(epochs):
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    preds = this(x)
                    L = this.loss(x, preds)
                    grads = grad(L, this.params)
                    this.update_params(lr, grads)
                total = 0
                for i:ℕ(images):
                    x: ℝ[1,d] = [X[i]]
                    pred = this(x)
                    total += this.loss(x, pred)
                print(total / images)
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W = this.W - lr * learnable_grads[0]
            this.U = this.U - lr * learnable_grads[1]
            this.b = this.b - lr * learnable_grads[2]
            this.Wo = this.Wo - lr * learnable_grads[3]
            this.bo = this.bo - lr * learnable_grads[4]

    print(DEVICE)
    W: ℝ[32,32] = rand_array(32, 32, 0.01)
    U: ℝ[784,32] = rand_array(784, 32, 0.02)
    b: ℝ[1,32] = zero_array(1, 32)
    Wo: ℝ[32,784] = rand_array(32, 784, 0.05)
    bo: ℝ[1,784] = zero_array(1, 784)
    deq = DEQ(W, U, b, Wo, bo)

    images: ℝ = 1000
    X: ℝ[1000, 784] = load_mnist(images)

    x0: ℝ[1,784] = [X[0]]
    recon_before: ℝ[1,784] = deq(x0)
    loss_before: ℝ = deq.loss(x0, recon_before)
    print(loss_before)

    epochs: ℕ = 20
    lr: ℝ = 0.0001
    deq.train(X, epochs, lr, images)
    recon_after: ℝ[1,784] = deq(x0)
    loss_after: ℝ = deq.loss(x0, recon_after)
    print(loss_after)


Training plots
--------------

After running the code above, you should see the average reconstruction loss decrease over epochs as the model learns to encode and decode the digits through its equilibrium state.

.. figure:: /_static/tutorial_files/deq/deq_train_curve.png
   :alt:
   :align: center
   :width: 750px


References
----------

.. [BaiDEQ2019] S. Bai, J. Z. Kolter, and V. Koltun,
    *Deep Equilibrium Models*.
    https://arxiv.org/abs/1909.01377
    *(placeholder — verify citation and page/section numbers before publishing)*

.. [ImplicitLayers] J. Z. Kolter, D. Duvenaud, and M. Johnson,
    *Deep Implicit Layers: Neural ODEs, Deep Equilibrium Models, and Beyond*.
    https://implicit-layers-tutorial.org/
    *(placeholder — verify chapter links for the DEQ and implicit-function sections)*

.. [Wikipedia_Banach] Wikipedia,
    *Banach fixed-point theorem*.
    https://en.wikipedia.org/wiki/Banach_fixed-point_theorem

.. [Wikipedia_IFT] Wikipedia,
    *Implicit function theorem*.
    https://en.wikipedia.org/wiki/Implicit_function_theorem
    *(placeholder — used for the backward-pass derivation)*
