# Target Exact Cosine Family

We use the exact separable cosine target

```math
g(\xi)=c+\sum_{i=1}^d a_i \cos(2\pi k_i x_i+\phi_i),
\qquad \xi=(x_1,\dots,x_d)\in[0,1]^d,
```

with the boundedness condition

```math
\sum_{i=1}^d |a_i| \le \min(c, 1-c),
```

so that `g(xi) in [0, 1]`.

## Tilted Distribution

For policy `pi=(p_s, p_u)` with `p_s + p_u = 1`, define

```math
\lambda := \beta p_u.
```

Then the tilted target is

```math
\nu_\beta(\xi \mid \pi)
=
\frac{\exp(\lambda g(\xi))}{Z(\lambda)} \mathbf{1}_{[0,1]^d}(\xi),
```

with normalizer

```math
Z(\lambda)=\int_{[0,1]^d} e^{\lambda g(\xi)} d\xi.
```

## Exact Normalization

Because the target is separable,

```math
Z(\lambda)
=
e^{\lambda c}
\prod_{i=1}^d
\int_0^1 e^{\lambda a_i \cos(2\pi k_i x + \phi_i)} dx.
```

Phase and integer frequency do not change the period integral, so

```math
\int_0^1 e^{\lambda a_i \cos(2\pi k_i x + \phi_i)} dx = I_0(\lambda a_i).
```

Hence

```math
Z(\lambda)=e^{\lambda c}\prod_{i=1}^d I_0(\lambda a_i).
```

The log-partition is

```math
A(\lambda)=\log Z(\lambda)=\lambda c+\sum_{i=1}^d \log I_0(\lambda a_i).
```

## Unsafe Action Value

The unsafe action value is the expectation of `g` under the tilted target:

```math
Q_\beta(a_u)=\mathbb{E}_{\nu_\beta}[g(\xi)]=A'(\lambda).
```

Differentiating gives the exact formula

```math
Q_\beta(a_u)
=
c+\sum_{i=1}^d a_i \frac{I_1(\lambda a_i)}{I_0(\lambda a_i)}.
```

The safe action value is constant:

```math
Q_\beta(a_s)=q.
```

## Variance

The curvature of the log-partition controls the tilted variance:

```math
\mathrm{Var}_{\nu_\beta}(g)=A''(\lambda).
```

So all cumulants are available from `A`.

## Fixed-Point Equation

With soft policy update temperature `tau > 0`,

```math
p_u = \sigma\left(\tau (Q_\beta(a_u)-q)\right),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

Substituting `Q_beta(a_u)=A'(\beta p_u)` gives the exact fixed-point equation

```math
p_u
=
\sigma\left(
\tau\left[
c+\sum_{i=1}^d a_i \frac{I_1(\beta p_u a_i)}{I_0(\beta p_u a_i)} - q
\right]
\right).
```

## Extreme-Beta Limits

Since `I_1(z)/I_0(z) -> 1` as `z -> +infty` and `I_1(z)/I_0(z) -> -1` as `z -> -infty`,

```math
Q_\beta(a_u)\to c+\sum_i a_i \qquad (\beta p_u \to +\infty),
```

```math
Q_\beta(a_u)\to c-\sum_i a_i \qquad (\beta p_u \to -\infty).
```

Under the boundedness constraint, these are exactly `g_max` and `g_min`.

## CLI Parameters

The toy scripts now accept:

- `--target-c`
- `--target-a`
- `--target-k`
- `--target-phi`
- `--target-amplitude-budget`

Examples:

```bash
python learning/module/gbs/gbs_test_toy.py \
  --dim 3 \
  --target-c 0.5 \
  --target-a 0.10,0.15,0.20 \
  --target-k 1,2,3 \
  --target-phi 0.0,1.2,2.4
```

If `--target-a` is omitted, amplitudes are generated from the amplitude budget.
