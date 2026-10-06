# Phase 2 — Neural Network Fundamentals

> **Goal:** Understand how a neural network works step by step — not as a magic black box, but as a simple machine you can follow by hand. If you can trace the math through a tiny network, you can understand any neural network.

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

Build this with Python and NumPy (no fancy library needed):

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
```

Trace each number by hand. Once you can follow this tiny network, no larger network should feel like a mystery — it's just more of the same, repeated many times.

---

**Next step →** [Phase 3: Embeddings](Phase3_Embeddings.md)
