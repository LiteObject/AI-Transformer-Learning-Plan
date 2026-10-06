# Phase 1 — Essential Math

> **Goal:** Learn just enough math to understand the AI concepts coming next. You do **not** need a math degree. We learn one small idea at a time, right before we need it.

---

## 1. What Is a Vector?

A **vector** is just an **ordered list of numbers**. That's it.

- A single number (like `5`) is called a **scalar**.
- A list of numbers is a **vector**.
- A grid of numbers is a **matrix**.

Example of a vector with 3 numbers (we say it has **3 dimensions**):

```
v = [3, 1, 4]
```

Think of a vector as an **arrow pointing in space**. In 2D, `[3, 4]` means "go 3 steps right and 4 steps up." The arrow has:

- a **direction** (which way it points), and
- a **length** (also called **magnitude**).

### Why do we care?

In AI, we turn **words into vectors** (lists of numbers). A computer can't understand the word "cat", but it can do math with the numbers `[0.21, -0.73, 0.14]`. Vectors are how AI "thinks" about words.

---

## 2. What Is a Matrix?

A **matrix** is a grid (a table) of numbers. It has **rows** and **columns**.

```
     column 1   column 2
row 1    1         2
row 2    3         4
row 3    5         6
```

This is a "3 by 2" matrix (3 rows, 2 columns). We write it as:

```
M = | 1  2 |
    | 3  4 |
    | 5  6 |
```

### Why do we care?

A matrix is like a **machine with many knobs** that takes a vector in and produces a new vector out. In a neural network, these knobs are called **weights**, and the network learns what numbers to put in the matrix.

---

## 3. Dot Product (The Most Important Operation)

The **dot product** takes two vectors of the same length and returns a single number.

**How to calculate it:** multiply matching positions, then add everything up.

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1 × 4) + (2 × 5) + (3 × 6)
      = 4 + 10 + 18
      = 32
```

### What does the dot product mean?

It measures **how much two vectors point in the same direction**:

- **Big positive number** → the vectors have a strong alignment, though the value also depends on their lengths.
- **Zero** → the vectors are orthogonal in this mathematical space. That does not prove that the concepts have no relationship.
- **Negative number** → the vectors have some opposing direction. This does not necessarily mean the concepts have opposite meanings.

### Why do we care?

This is the heart of AI! When a model wants to know "how similar are these two words?", it often computes a dot product between their vectors. Bigger dot product = more related.

> 💡 **Tiny example:** "cat" and "kitten" have vectors that point similarly → large dot product. "cat" and "bicycle" have vectors that point differently → small dot product.

---

## 4. Matrix Multiplication

Matrix multiplication is just **many dot products done at once**.

To multiply a matrix by a vector:

- Take each **row** of the matrix.
- Compute its **dot product** with the vector.
- The results become the new vector.

```
| 1  2 |   | 3 |   | (1×3)+(2×4) |   | 11 |
| 3  4 | × | 4 | = | (3×3)+(4×4) | = | 25 |
| 5  6 |   |   |   | (5×3)+(6×4) |   | 39 |
```

The matrix "transforms" the input vector `[3, 4]` into a new vector `[11, 25, 39]`.

### Why do we care?

Every layer of a neural network does exactly this: **multiply an input vector by a matrix of weights, add a bias, and apply a function.** If you understand matrix multiplication, you understand the skeleton of almost every AI model.

---

## 5. Functions: Input → Transformation → Output

A **function** is a rule that takes an input and produces an output.

```
input → [function] → output
```

Examples:

- `f(x) = x + 1`  →  if input is 5, output is 6.
- `f(x) = x²`     →  if input is 3, output is 9.

### Why do we care?

**A neural network is just a function** — a very big, complicated function made by stacking many smaller functions together. When we "train" a model, we're really just finding the best numbers for that function so that it produces the right outputs.

---

## 6. Probability Basics

**Probability** is a number between 0 and 1 that says how likely something is.

- Probability `0` → it will never happen.
- Probability `1` → it will always happen.
- Probability `0.7` → it happens 70% of the time.

A **probability distribution** is a list of probabilities for all possible outcomes. The probabilities always add up to 1.

```
Outcome:      "cat"   "dog"   "fish"
Probability:   0.7     0.2     0.1     (adds up to 1.0)
```

### Conditional probability

A **conditional probability** asks: "What's the probability of A, **given that** B already happened?"

```
P(A | B) = probability of A given B
```

**Example:** What's the probability the next word is "cute", given that the previous word is "kitten"? The model uses conditional probability to predict the next word based on the words before it.

### Why do we care?

Large language models (LLMs) **don't "know" answers** — they compute the probability of every possible next word and pick one. Probability is the language of prediction.

---

## 7. Basic Calculus: Derivatives and Gradients

A **derivative** tells you **how fast something changes**.

**Example:** If you're driving and your speed changes over time, the derivative of your distance tells you your speed. It answers: "If I change the input a tiny bit, how much does the output change?"

A **gradient** is the derivative extended to multiple dimensions — it's a vector that points in the **direction of steepest increase**.

### Why do we care?

When training a model:

1. We measure how wrong it is (the **loss**).
2. We compute the **gradient** — "if I nudge each weight a little, does the loss go up or down?"
3. We nudge the weights **opposite to the gradient** (downhill), so the loss gets smaller.

This is called **gradient descent**, and it's how nearly all AI models learn. The word "gradient" will appear constantly in AI, so get comfortable with this one idea: **the gradient tells you which way is downhill, and training walks downhill.**

---

## 8. Statistics Basics

These are just ways to summarize a list of numbers.

### Mean (average)

Add everything up, then divide by the count.

```
numbers = [2, 4, 6]
mean = (2 + 4 + 6) / 3 = 4
```

### Variance

A measure of **how spread out** the numbers are. If all numbers are the same, variance is 0. If they're wildly different, variance is big.

### Standard deviation

The square root of variance. It tells you, on average, how far numbers are from the mean. Small standard deviation = numbers are close together; big = spread out.

### Normalization

**Normalization** rescales numbers so they have a standard range (for example, a mean of 0 and a standard deviation of 1).

```
normalized = (value - mean) / standard deviation
```

### Why do we care?

In Transformers, there's a step called **Layer Normalization** that keeps the numbers flowing through the network stable. Without it, the numbers can grow wildly and the model becomes hard to train. Understanding mean, variance, and standard deviation makes normalization feel obvious instead of mysterious.

---

## Summary: What to Remember

| Concept | One-line meaning |
|---|---|
| Vector | An ordered list of numbers; an arrow in space |
| Matrix | A grid of numbers; a machine that transforms vectors |
| Dot product | A single number measuring how much two vectors agree |
| Matrix multiplication | Many dot products at once; how layers transform data |
| Function | Input → transformation → output |
| Probability | A number from 0 to 1 saying how likely something is |
| Conditional probability | Likelihood of A given that B happened |
| Derivative / Gradient | How fast something changes; which way is "uphill" |
| Mean / Variance / Std dev | Average, spread, typical distance from average |
| Normalization | Rescaling numbers to a stable range |

---

## Suggested Mini-Project: Vector Playground

Open Python and try these with the `NumPy` library:

```python
import numpy as np

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
print(a @ b)          # 32

# Matrix multiplication
M = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([3, 4])
print(M @ x)          # [11, 25, 39]

# Mean and standard deviation
data = np.array([2, 4, 6, 8])
print(data.mean())            # 5.0
print(data.std())             # ~2.24

# Normalize (subtract mean, divide by std)
normalized = (data - data.mean()) / data.std()
print(normalized)
```

Run each line and make sure the numbers match your hand calculations. Comfort with these operations will make every later phase easier.

---

**Next step →** [Phase 2: Neural Network Fundamentals](Phase2_Neural_Network_Fundamentals.md)
