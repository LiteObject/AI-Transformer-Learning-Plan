# Phase 2 — Neural Network Fundamentals

> **Goal:** Understand a neural network's forward pass and training step by step. Tracing a tiny network gives you a foundation for the more complex architectures that follow.

---

## 1. The Big Idea

A neural network is a **function** (remember Phase 1: input → transformation → output) that learns from examples.

```
input → hidden layers → output
```

During **training**, the network:
1. Makes a prediction.
2. Compares it to the correct answer.
3. Measures how wrong it is (the **loss**).
4. Adjusts its internal numbers so the prediction gets closer next time.

That's the whole loop. Everything else is detail.

---

## 2. Neurons, Weights, and Bias

A **neuron** is a tiny calculator. It takes several numbers in, combines them, and produces one number out.

Each input has a **weight** — a number that says "how important is this input?" The neuron:

1. Multiplies each input by its weight.
2. Adds everything together.
3. Adds a **bias** (an extra number that lets the neuron shift its output).
4. Passes the result through an **activation function**.

**Example — one neuron with two inputs:**

```
inputs:      x1 = 2,   x2 = 3
weights:     w1 = 0.5, w2 = -1
bias:        b = 0.1

step 1:  multiply   →  2 × 0.5 = 1.0   and   3 × (-1) = -3
step 2:  add        →  1.0 + (-3) = -2
step 3:  add bias   →  -2 + 0.1 = -1.9
step 4:  activate   →  depends on the activation function
```

### Why the analogy?

- **Weights** are like **volume knobs**: how much each input matters.
- **Bias** is like a **baseline**: how likely the neuron is to "fire" even without input.
- **Training** is just finding the best values for all the weights and biases.

In math, one neuron computes: `output = activation(w1·x1 + w2·x2 + ... + b)`.

---

## 3. Linear Transformations

A bunch of neurons together form a **layer**. Each layer takes a vector in and produces a vector out using **matrix multiplication** (from Phase 1):

```
output = W × input + b
```

Where `W` is a matrix of weights and `b` is a vector of biases.

A "linear layer" (also called a **dense** or **fully connected** layer) means **every neuron connects to every input**.

With a bias, the transformation is technically **affine**, although libraries still call it a linear layer. For one vector, `W` has shape `(outputs, inputs)`; for a batch of row vectors, the equivalent calculation is `X @ W.T + b` (Phase 1).

```
input (3 numbers)          hidden layer (4 neurons)
     x1 ────────────────→  n1
     x2 ────────────────→  n2
     x3 ────────────────→  n3
                          n4
```

Each connection has its own weight. The layer's job: **mix the inputs into new combinations.**

---

## 4. Activation Functions

If we only used linear layers, stacking many of them would be the same as one big linear layer — the network couldn't learn anything interesting. **Activation functions add "curves"** (non-linearity) so the network can learn complex patterns.

Two common ones:

### ReLU (Rectified Linear Unit)

```
ReLU(x) = x  if x > 0
ReLU(x) = 0  if x ≤ 0
```

In plain words: **"if positive, keep it; if negative, make it zero."**

```
ReLU(-3) = 0
ReLU(2)  = 2
```

ReLU is simple and fast, and it remains common in many neural networks. Transformer-based language models often use alternatives such as GELU or gated activations such as SwiGLU.

### Sigmoid

The sigmoid squashes any number into a value **between 0 and 1**.

```
sigmoid(-10) ≈ 0
sigmoid(0)   = 0.5
sigmoid(10)  ≈ 1
```

It's used when we want an output that behaves like a probability (a value from 0 to 1).

---

## 5. Softmax: Turning Numbers into Probabilities

**Softmax** takes a list of numbers and turns them into **probabilities** (all between 0 and 1, adding up to exactly 1).

**Example:**

```
raw scores (logits):  [2.0, 1.0, 0.1]
after softmax:        [0.66, 0.24, 0.10]   (adds up to 1.0)
```

- The **biggest** raw number becomes the **most likely** outcome.
- The ordering is preserved, but the probabilities are not proportional to the raw scores. Softmax depends exponentially on the differences between logits.
- The formula makes sure everything is positive and sums to 1.

### Why do we care?

At the end of a language model, the network outputs a raw score for every possible next word. **Softmax converts those scores into "what fraction of the model's confidence is on each word?"** That's how the model decides which word to say.

---

## 6. Loss Functions: Measuring How Wrong We Are

The **loss** is a single number: **how far the prediction is from the correct answer.**

- Loss = 0 → perfect prediction.
- Loss > 0 → the bigger, the worse.

Common loss functions:

### Mean Squared Error (MSE)

For numeric predictions — average of the squared differences.

```
predicted = 6,  correct = 4
error = (6 - 4)² = 4
```

### Cross-Entropy Loss

For probability predictions (like word prediction). It's the standard loss for language models.

It heavily punishes the model when it's **very confident about the wrong answer**. If the correct word is "cat" but the model gave "cat" only 2% probability, the loss is large; if the model gave it 98%, the loss is tiny.

For one correct class or token `y`, the formula is:

```
loss = -log(p_y)       (natural logarithm)
-log(0.02) ≈ 3.912
-log(0.98) ≈ 0.020
```

In PyTorch, pass **raw logits**, not already-softmaxed probabilities, to `CrossEntropyLoss` or `F.cross_entropy`. These functions combine log-softmax and the loss in a numerically stable calculation. Use softmax separately when you want probabilities for display or sampling.

### Why do we care?

The loss is the **scoreboard**. Training is nothing more than: "keep adjusting weights until the loss is as small as possible."

---

## 7. Gradient Descent: Learning by Going Downhill

Imagine the loss as the **height of a mountain**, and the weights as your position on the map. Training is like walking downhill in the fog — you can't see the whole mountain, but you can feel which way is downhill and take a step.

**Gradient descent algorithm:**

1. Compute the loss.
2. Compute the **gradient** — a vector showing which direction increases the loss fastest (Phase 1). The **opposite** direction is downhill.
3. Nudge the weights a tiny step in the downhill direction.
4. Repeat thousands or millions of times.

```
new weight = old weight − (learning rate × gradient)
```

The **learning rate** is the step size:
- **Too big** → you overshoot and bounce around, never settling.
- **Too small** → you learn, but glacially slowly.

### Why "descent"? 

Because we're trying to **decrease** (descend) the loss, step by step, until the model is as accurate as possible.

### One weight update by hand

Take `prediction = w × x` and `loss = 0.5 × (prediction - target)²`, with `x = 2`, `target = 1`, and `w = 0.25`.

```
prediction = 0.25 × 2 = 0.5
loss = 0.5 × (0.5 - 1)² = 0.125

gradient = (prediction - target) × x
         = (0.5 - 1) × 2 = -1

new w = 0.25 - (0.1 × -1) = 0.35
new prediction = 0.35 × 2 = 0.7
new loss = 0.5 × (0.7 - 1)² = 0.045
```

The chain rule multiplies the effect of prediction on loss by the effect of the weight on prediction. Subtracting a **negative** gradient increases this weight and reduces the loss.

---

## 8. Backpropagation: How the Error Travels Backward

Gradient descent needs a gradient for **every single weight** — and a network can have billions. Computing each one separately would be impossibly slow.

**Backpropagation** (short for "backward propagation of errors") is the clever trick that computes all gradients at once:

1. **Forward pass:** data flows through the network, layer by layer, to make a prediction. We record every intermediate value.
2. **Compute the loss.**
3. **Backward pass:** starting from the loss, we work **backwards** through the network, applying the **chain rule** of calculus to figure out how much each weight contributed to the error.

The **chain rule** just says: to find how much a change in the *first* layer affects the final loss, multiply the effects through each intermediate step.

> 💡 **Analogy:** A team passes a bucket of water down a line. To find out who spilled it, you trace the spill **backward** from the last person to the first, checking each person's hands along the way.

---

## Putting It All Together: A Tiny Network

Here is a complete miniature network with one hidden layer:

```
Input (2 numbers)
   ↓
Linear layer (W1, b1)  →  3 neurons
   ↓
ReLU
   ↓
Linear layer (W2, b2)  →  2 neurons
   ↓
Softmax
   ↓
Output: 2 probabilities
```

**Training loop (in words):**

```
repeat many times:
    1. forward pass:    input → layers → softmax → prediction
    2. compute loss:    how wrong is the prediction?
    3. backward pass:   backpropagation computes all gradients
    4. update weights:  nudge every weight downhill (gradient descent)
```

After enough repetitions, the network "learns" the pattern hidden in the training data.

---

## Suggested Mini-Project: Tiny Neural Network

Build and train this with Python and NumPy. First trace the forward pass, then inspect the gradients and updates:

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max())          # subtract max for stability
    return e / e.sum()

def linear(x, W, b):
    return W @ x + b

# Tiny network with one hidden layer
np.random.seed(42)
W1 = np.random.randn(3, 2)           # 3 neurons, 2 inputs
b1 = np.random.randn(3)
W2 = np.random.randn(2, 3)           # 2 outputs, 3 hidden neurons
b2 = np.random.randn(2)

x = np.array([1.0, 2.0])

h = relu(linear(x, W1, b1))          # hidden layer + ReLU
y = softmax(linear(h, W2, b2))       # output layer + softmax

print("hidden:", h)
print("output probabilities:", y, "sum:", y.sum())

def loss_and_gradients(x, target):
    z1 = linear(x, W1, b1)
    h = relu(z1)
    logits = linear(h, W2, b2)
    probabilities = softmax(logits)
    shifted = logits - logits.max()
    loss = np.log(np.exp(shifted).sum()) - shifted[target]

    dz2 = probabilities.copy()
    dz2[target] -= 1
    dW2 = np.outer(dz2, h)
    db2 = dz2
    dh = W2.T @ dz2
    dz1 = dh * (z1 > 0)
    dW1 = np.outer(dz1, x)
    db1 = dz1
    return loss, (dW1, db1, dW2, db2)

inputs = np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]])
targets = np.array([0, 0, 1, 1])
parameters = (W1, b1, W2, b2)
initial_loss = np.mean([loss_and_gradients(x, t)[0] for x, t in zip(inputs, targets)])

for epoch in range(200):
    for x, target in zip(inputs, targets):
        loss, gradients = loss_and_gradients(x, target)
        for parameter, gradient in zip(parameters, gradients):
            parameter -= 0.05 * gradient

final_loss = np.mean([loss_and_gradients(x, t)[0] for x, t in zip(inputs, targets)])
predictions = np.array([
    softmax(linear(relu(linear(x, W1, b1)), W2, b2)).argmax()
    for x in inputs
])
print("mean loss:", initial_loss, "->", final_loss)
print("predicted classes:", predictions)  # [0, 0, 1, 1]
```

For softmax followed by cross-entropy, the gradient with respect to the output logits is `probabilities - one_hot(target)`. The code then follows the chain backward: output weights, hidden activations, ReLU, and input weights. `np.outer` produces a gradient with the same shape as its weight matrix.

This tiny dataset checks that training works; it does **not** measure generalization. Phase 6 introduces separate validation text.

## Before Moving On

| Check | Completion criterion |
|---|---|
| By hand | Reproduce the single-weight update from `0.25` to `0.35` and the loss reduction from `0.125` to `0.045`. |
| In code | Train the tiny classifier, confirm the mean loss decreases, and obtain classes `[0, 0, 1, 1]`. Compare one weight's analytical gradient with a central finite difference from Phase 1, away from ReLU's kink at zero. |
| In your own words | Distinguish backpropagation (computing gradients) from the optimizer (updating parameters), and explain why PyTorch cross-entropy receives logits. |


---

**Next step →** [Phase 3: Embeddings](Phase3_Embeddings.md)
