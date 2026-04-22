# Target_A Exact Truncated-Exponential Family

`Target_A` is the original non-sinusoidal target used in the toy experiments.
Unlike `target_B` and `target_C`, it is not multimodal. It is a separable
product of 1D truncated exponentials on the unit box.

## Target Definition

Let

```math
\xi=(x_1,\dots,x_d)\in[0,1]^d.
```

For a scalar tilt parameter `lambda`, define

```math
\nu_\lambda(\xi)
=
\frac{\exp\!\left(\lambda \sum_{i=1}^d x_i\right)}
{Z_d(\lambda)}
\mathbf 1_{[0,1]^d}(\xi).
```

Equivalently, the log-density is

```math
\log \nu_\lambda(\xi)
=
\lambda \sum_{i=1}^d x_i - \log Z_d(\lambda),
\qquad \xi\in[0,1]^d.
```

In code, this is exactly the product target implemented in
[target_A_notebook_utils.py](/home/sukchul/risk-sensitive-domain-randomization/learning/module/gbs/targets/target_A_notebook_utils.py).

## Exact Normalization

Because the target is separable,

```math
Z_d(\lambda)
=
\left(
\int_0^1 e^{\lambda x}dx
\right)^d.
```

For `lambda \neq 0`,

```math
\int_0^1 e^{\lambda x}dx
=
\frac{e^\lambda-1}{\lambda},
```

so

```math
Z_d(\lambda)
=
\left(\frac{e^\lambda-1}{\lambda}\right)^d.
```

At `lambda=0`, the density is uniform on `[0,1]^d`, so

```math
Z_d(0)=1.
```

The log-partition is therefore

```math
A_d(\lambda)
:=
\log Z_d(\lambda)
=
d\left[\log(e^\lambda-1)-\log \lambda\right],
\qquad \lambda\neq 0,
```

with the continuous extension `A_d(0)=0`.

## Coordinate Marginal

Each coordinate is an independent truncated exponential:

```math
\nu_\lambda(x_i)
=
\frac{\lambda e^{\lambda x_i}}{e^\lambda-1}\mathbf 1_{[0,1]}(x_i),
\qquad \lambda\neq 0.
```

At `lambda=0`, this reduces to the uniform density on `[0,1]`.

## Mean

For one coordinate,

```math
m(\lambda)
:=
\mathbb E_{\nu_\lambda}[x_i]
=
\frac{e^\lambda}{e^\lambda-1}-\frac{1}{\lambda},
\qquad \lambda\neq 0,
```

and `m(0)=1/2`.

Since the target factorizes,

```math
\mathbb E_{\nu_\lambda}\!\left[\sum_{i=1}^d x_i\right]
=
d\,m(\lambda).
```

If we use the per-coordinate statistic `g(\xi)=x_i` in the policy update, the
relevant target mean is simply `m(lambda)`.

## Mode Structure

`Target_1` is effectively single-mode.

- If `lambda > 0`, the density increases toward `x_i=1` in every coordinate, so
  the mode is at

```math
\xi^\star=(1,\dots,1).
```

- If `lambda < 0`, the density decreases toward `x_i=0` in every coordinate, so
  the mode is at

```math
\xi^\star=(0,\dots,0).
```

- If `lambda = 0`, the target is uniform and has no unique mode.

This is why `target_A` should be understood as the one-mode baseline family,
not the harmonic multimodal family.

## Policy Coupling

In the toy setup, policy enters through

```math
\lambda=\beta p_u,
```

where `p_u` is the unsafe-action probability.

The safe action value is constant:

```math
Q_\beta(a_s)=q.
```

The unsafe action value is the target mean used by the scripts:

```math
Q_\beta(a_u)=m(\beta p_u)
=
\frac{e^{\beta p_u}}{e^{\beta p_u}-1}-\frac{1}{\beta p_u},
```

with the continuous value `Q_beta(a_u)=1/2` when `beta p_u = 0`.

## Soft Fixed Point

With temperature `tau > 0`,

```math
p_u
=
\sigma\left(\tau(Q_\beta(a_u)-q)\right),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

Substituting the exact mean gives

```math
p_u
=
\sigma\left(
\tau\left[
\frac{e^{\beta p_u}}{e^{\beta p_u}-1}-\frac{1}{\beta p_u}-q
\right]
\right),
```

with the same continuous interpretation at `beta p_u = 0`.

## Sampling

Exact sampling is available by inverse CDF.

For `u ~ Uniform(0,1)`,

```math
x
=
\frac{1}{\lambda}\log\!\left(1+u(e^\lambda-1)\right),
\qquad \lambda\neq 0,
```

and `x=u` when `lambda=0`.

This is the formula used in the implementation for
`sample_truncated_exponential(...)`.
