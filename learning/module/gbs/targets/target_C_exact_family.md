# Target_3 Exact Policy-Shifted Cosine Family

We define **Target_3** as a separable cosine target whose **mode location depends on the policy**.
This extends the earlier `Target` family, where the policy only affected the tilt strength through
`lambda = beta p_u`, but not the mode location itself. In `Target_3`, the policy changes both:

- the **concentration** of the tilted density through `lambda = beta p_u`, and
- the **mode location** through a policy-dependent center `mu(p)`.

## Target Definition

Let

```math
\xi=(x_1,\dots,x_d)\in[0,1]^d,
\qquad
\pi=(p_s,p_u),
\qquad
p_s+p_u=1.
```

Define

```math
g(\xi;\pi)
=
c+\sum_{i=1}^d a_i\cos\bigl(2\pi k_i(x_i-\mu_i(\pi))\bigr),
```

with policy-dependent centers

```math
\mu_i(\pi)=\mu_{i,0}+b_i p_u
\pmod{1}.
```

Here:

- `c in (0,1)` is the base level,
- `a_i` are amplitudes,
- `k_i in N` are integer frequencies,
- `mu_{i,0}` are base centers,
- `b_i` controls how strongly policy shifts the mode in coordinate `i`.

The boundedness condition is the same as before:

```math
\sum_{i=1}^d |a_i| \le \min(c, 1-c),
```

which guarantees

```math
g(\xi;\pi)\in[0,1]
\qquad
\text{for all } \xi\in[0,1]^d \text{ and all } \pi.
```

## Tilted Distribution

As before, define

```math
\lambda := \beta p_u.
```

Then the policy-conditioned tilted target is

```math
\nu_\beta(\xi\mid\pi)
=
\frac{\exp\bigl(\lambda g(\xi;\pi)\bigr)}{Z(\lambda,\pi)}
\mathbf 1_{[0,1]^d}(\xi).
```

Equivalently,

```math
\xi_{t+1}\sim \nu_\beta(\cdot\mid\pi_t).
```

This is the policy-to-domain transition law for `Target_3`.

## Exact Normalization

Because the target remains separable,

```math
Z(\lambda,\pi)
=
\int_{[0,1]^d} e^{\lambda g(\xi;\pi)}d\xi
=
e^{\lambda c}
\prod_{i=1}^d
\int_0^1 e^{\lambda a_i\cos(2\pi k_i(x-\mu_i(\pi)))}dx.
```

For integer `k_i`, shifting the center does not change the period integral, so

```math
\int_0^1 e^{\lambda a_i\cos(2\pi k_i(x-\mu_i(\pi)))}dx = I_0(\lambda a_i).
```

Hence the exact normalizer is still

```math
Z(\lambda,\pi)=e^{\lambda c}\prod_{i=1}^d I_0(\lambda a_i).
```

Therefore the log-partition is

```math
A(\lambda)=\log Z(\lambda,\pi)=\lambda c+\sum_{i=1}^d \log I_0(\lambda a_i),
```

which is **independent of the shifting centers** `mu_i(pi)`.

This is the key advantage of `Target_3`: the mode moves with policy, while exact normalization is preserved.

## Mode Locations

A coordinate-wise maximum occurs when

```math
2\pi k_i(x_i-\mu_i(\pi)) = 2\pi m_i,
\qquad m_i\in\{0,\dots,k_i-1\}.
```

So the mode coordinates are

```math
x_i^\star(\pi)=\mu_i(\pi)+\frac{m_i}{k_i}
\pmod{1}.
```

Substituting `mu_i(pi)=mu_{i,0}+b_i p_u` gives

```math
x_i^\star(\pi)=\mu_{i,0}+b_i p_u+\frac{m_i}{k_i}
\pmod{1}.
```

Thus `p_u` changes the mode location linearly.

## Unsafe Action Value

Although the **location** of the peaks changes with policy, the expectation of `g` under the tilted target still depends only on `lambda`:

```math
Q_\beta(a_u)=\mathbb E_{\nu_\beta}[g(\xi;\pi)] = A'(\lambda).
```

Therefore

```math
Q_\beta(a_u)
=
c+\sum_{i=1}^d a_i\frac{I_1(\lambda a_i)}{I_0(\lambda a_i)}.
```

The safe action value remains

```math
Q_\beta(a_s)=q.
```

## Variance

The variance under the tilted target is still

```math
\mathrm{Var}_{\nu_\beta}(g)=A''(\lambda),
```

so the cumulant structure remains exactly available.

## Soft Policy Fixed Point

With temperature `tau > 0`, let

```math
p_u=\sigma\left(\tau(Q_\beta(a_u)-q)\right),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

Then the exact fixed-point equation is unchanged in form:

```math
p_u
=
\sigma\left(
\tau\left[
c+\sum_{i=1}^d a_i\frac{I_1(\beta p_u a_i)}{I_0(\beta p_u a_i)}-q
\right]
\right).
```

So `Target_3` changes the **geometry of the density** without changing the closed-form fixed-point relation for `p_u`.

## Transition Dynamics Equation

If you want the explicit domain-parameter transition law, it is

```math
\xi_{t+1}
\sim
\frac{
\exp\!\left(
\beta p_{u,t}
\left[
c+\sum_{i=1}^d a_i\cos\bigl(2\pi k_i(x_i-\mu_{i,t})\bigr)
\right]
\right)}
{
e^{\beta p_{u,t}c}\prod_{i=1}^d I_0(\beta p_{u,t}a_i)
}
\mathbf 1_{[0,1]^d}(\xi),
```

with

```math
\mu_{i,t}=\mu_{i,0}+b_i p_{u,t}\pmod{1}.
```

This makes both the density sharpness and the mode location depend on the policy.

## Notes

- `b_i = 0` recovers the original `Target` geometry.
- Larger `|b_i|` makes the mode move more strongly as `p_u` changes.
- Integer `k_i` is important for preserving the exact Bessel normalizer.
- The family remains analytically simple, but now has policy-dependent moving peaks.

## Suggested CLI Parameters

A natural extension of the toy scripts would be:

- `--target-3-c`
- `--target-3-a`
- `--target-3-k`
- `--target-3-mu0`
- `--target-3-b`
- `--target-3-amplitude-budget`

Example:

```bash
python learning/module/gbs/gbs_test_toy.py \
  --dim 2 \
  --target-3-c 0.50 \
  --target-3-a 0.20,0.15 \
  --target-3-k 3,5 \
  --target-3-mu0 0.20,0.65 \
  --target-3-b 0.35,-0.20
```

In this case, increasing `p_u` moves the first mode rightward and the second mode leftward.
