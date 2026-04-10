import jax
import jax.numpy as jnp
import numpy as np
import distrax
import optax
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
from flax.training import train_state
from tqdm import trange

from learning.module.gbs.gbs_loss import (
    add_subtraj_aux_defaults,
    VP,
    lv_loss_from_rnd,
    lv_loss_from_values,
    lv_subtraj_loss,
    rnd_no_target,
    rnd_time_reversal_lv_no_target,
    rnd_time_reversal_lv_subtraj_no_target,
)
from learning.module.gbs.gbs_trainer import make_gbs_model
from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)
from learning.module.gbs.target4_family import (
    Target4HarmonicParams,
    Target4ShiftedHarmonicParams,
    get_target4_harmonic_params,
    sample_target4_product,
    target4_energy_values,
    target4_expected_energy,
    target4_log_normalizer,
)


def tanh_box_bijector(z, low, high):
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z, low, high):
    z = jnp.atleast_2d(z)
    half = 0.5 * (high - low)
    jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
    return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def target4_logprob(
    z,
    lam,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
    policy_p=None,
):
    z = jnp.atleast_2d(z)
    params = get_target4_harmonic_params(z.shape[-1], target_params, policy_p=policy_p)
    log_unnorm = jnp.asarray(lam, dtype=jnp.float32) * target4_energy_values(z, params, policy_p=policy_p)
    return (log_unnorm - target4_log_normalizer(lam, params)).squeeze()


def update_p_from_samples(
    x,
    tau,
    q,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
    policy_p=None,
):
    x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
    params = get_target4_harmonic_params(x.shape[-1], target_params, policy_p=policy_p)
    sample_mean_g = jnp.mean(target4_energy_values(x, params, policy_p=policy_p))
    p = jax.nn.sigmoid(tau * (sample_mean_g - q))
    return p, sample_mean_g


def update_p_with_ema_and_jump(prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key):
    base_p = float(jax.nn.sigmoid(tau * (sample_mean_g - q)))
    ema_p = float(np.clip(ema_alpha * prev_p + (1.0 - ema_alpha) * base_p, 0.0, 1.0))
    key_jump, key_uniform = jax.random.split(key)
    jumped = bool(jax.random.bernoulli(key_jump, p=jump_prob))
    if jumped:
        new_p = float(jax.random.uniform(key_uniform, minval=0.0, maxval=1.0))
    else:
        new_p = ema_p
    return new_p, base_p, ema_p, jumped


def optimal_p_from_target_mean(
    lam,
    tau,
    q,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
):
    params = get_target4_harmonic_params(len(target_params.a) if target_params is not None else 1, target_params)
    target_mean = float(target4_expected_energy(lam, params))
    optimal_p = 1.0 / (1.0 + np.exp(-tau * (target_mean - q)))
    return float(optimal_p), float(target_mean)


def sample_truncated_exponential(
    key,
    lam,
    shape,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
    policy_p=None,
):
    if len(shape) != 2:
        raise ValueError(f"shape must be rank-2, got {shape}")
    params = get_target4_harmonic_params(shape[1], target_params, policy_p=policy_p)
    return sample_target4_product(key, lam, shape, params, policy_p=policy_p)


def compute_target4_metrics(
    samples,
    lam,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
    num_bins=128,
    eps=1e-8,
    key=None,
    policy_p=None,
):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be rank-2, got {samples.shape}")
    dim = samples.shape[1]
    params = get_target4_harmonic_params(dim, target_params, policy_p=policy_p)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    grid = np.arange(num_bins, dtype=np.float64)
    mids = (grid + 0.5) / num_bins
    comp = np.asarray(
        target4_energy_values(jnp.asarray(mids[:, None].repeat(dim, axis=1)), params, policy_p=policy_p)
    )
    del comp
    forward_terms = []
    reverse_terms = []
    for i in range(dim):
        q_hist, _ = np.histogram(np.clip(samples[:, i], 0.0, 1.0), bins=edges, density=False)
        q_probs = q_hist.astype(np.float64)
        q_probs = q_probs / max(q_probs.sum(), 1.0)

        x_mid = jnp.asarray(mids[:, None], dtype=jnp.float32)
        single_params = Target4HarmonicParams(
            c=jnp.asarray(0.0, dtype=jnp.float32),
            a=params.a[i : i + 1],
            k=params.k[i : i + 1],
            phi=params.phi[i : i + 1],
        )
        logw = np.asarray(
            target4_logprob(
                x_mid, jnp.asarray(lam, dtype=jnp.float32), target_params=single_params, policy_p=policy_p
            )
        )
        p_probs = np.exp(logw - scipy_special_logsumexp_np(logw))
        p_probs = np.maximum(p_probs, eps)
        p_probs = p_probs / p_probs.sum()
        q_probs = np.maximum(q_probs, eps)
        q_probs = q_probs / q_probs.sum()

        forward_terms.append(np.sum(p_probs * (np.log(p_probs) - np.log(q_probs))))
        reverse_terms.append(np.sum(q_probs * (np.log(q_probs) - np.log(p_probs))))

    forward_kl = float(np.mean(forward_terms))
    reverse_kl = float(np.mean(reverse_terms))

    if key is None:
        ref = np.tile(
            (np.linspace(0.0, 1.0, samples.shape[0], endpoint=False) + 0.5 / max(samples.shape[0], 1))[:, None],
            (1, dim),
        )
    else:
        ref = np.asarray(sample_truncated_exponential(key, lam, samples.shape, params, policy_p=policy_p))
    wasserstein = float(
        np.mean(
            np.abs(
                np.sort(np.clip(samples, 0.0, 1.0), axis=0)
                - np.sort(np.clip(ref, 0.0, 1.0), axis=0)
            )
        )
    )
    return forward_kl, reverse_kl, wasserstein


def scipy_special_logsumexp_np(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def should_compute_interatomic_w2(n_particles: int, max_pairs: int = 4096) -> bool:
    if n_particles <= 1:
        return False
    return (n_particles * (n_particles - 1)) // 2 <= max_pairs


def build_eval_iters(total_iters: int, max_eval_points: int | None) -> set[int]:
    if max_eval_points is None or max_eval_points <= 0 or max_eval_points >= total_iters:
        return set(range(total_iters))
    eval_iters = {int(v) for v in np.linspace(0, total_iters - 1, max_eval_points)}
    eval_iters.add(total_iters - 1)
    return eval_iters


def run_gbs_toy_target4(
    low=jnp.array([0.0, 0.0]),
    high=jnp.array([1.0, 1.0]),
    dim=2,
    T=1000,
    batch_size=512,
    num_steps=25,
    lr=1e-3,
    init_std=0.5,
    gif_path=None,
    snap_iters=None,
    seed=0,
    beta=1.0,
    tau=0.1,
    q=1.0,
    initial_p=None,
    p_update_freq=1,
    p_ema_alpha=0.9,
    p_jump_prob=0.0,
    loss_mode="tr_lv",
    metric_num_bins=128,
    sinkhorn_num_samples=256,
    n_particles=None,
    n_spatial_dim=1,
    save_dir=".",
    use_tanh_bijection=True,
    model_type="pisgrad",
    model_num_layers=2,
    model_num_hid=64,
    final_sample_size=2**14,
    max_rnd=1e8,
    target_params: Target4HarmonicParams | Target4ShiftedHarmonicParams | None = None,
    return_snapshots: bool = False,
    snapshot_sample_size: int | None = None,
    max_metric_eval_points: int | None = None,
):
    if snap_iters is None:
        snap_iters = []
    if snapshot_sample_size is None:
        snapshot_sample_size = final_sample_size
    metric_eval_iters = build_eval_iters(T, max_metric_eval_points)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    proc = VP(
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        terminal_t=1.0,
        generative=False,
        sign=-1.0,
        include_base_drift=True,
    )

    key = jax.random.PRNGKey(seed)
    key, k_p0 = jax.random.split(key)
    if initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(initial_p, 0.0, 1.0))

    low = jnp.asarray(low)
    high = jnp.asarray(high)
    if low.shape[0] != dim or high.shape[0] != dim:
        raise ValueError(f"low/high must match dim={dim}, got {low.shape} and {high.shape}")
    if n_particles is None:
        if dim % n_spatial_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by n_spatial_dim={n_spatial_dim}")
        n_particles = dim // n_spatial_dim
    if n_particles * n_spatial_dim != dim:
        raise ValueError(
            f"n_particles * n_spatial_dim must equal dim, got {n_particles} * {n_spatial_dim} != {dim}"
        )
    # Optional tanh bijection: train in unconstrained latent space z and map to the
    # bounded box only when requested.
    if use_tanh_bijection:
        def to_box(z):
            return tanh_box_bijector(z, low=low, high=high)

        def target4_logprob_latent(z, lam, policy_p):
            x_box = to_box(z)
            return (
                target4_logprob(x_box, lam, target_params=target_params, policy_p=policy_p)
                + tanh_box_logabsdet(z, low=low, high=high)
            )

        latent_prior_loc = jnp.zeros(dim, dtype=jnp.float32)
        process_center = jnp.zeros(dim, dtype=jnp.float32)
    else:
        to_box = lambda z: z
        target4_logprob_latent = lambda z, lam, policy_p: target4_logprob(
            z, lam, target_params=target_params, policy_p=policy_p
        )
        latent_prior_loc = 0.5 * (low + high)
        process_center = latent_prior_loc

    prior = distrax.MultivariateNormalDiag(
        loc=latent_prior_loc,
        scale_diag=jnp.ones(dim) * init_std,
    )
    if use_tanh_bijection:
        prior_sampler = lambda k: jnp.squeeze(prior.sample(seed=k, sample_shape=(1,)))
    else:
        prior_sampler = lambda k: jnp.clip(
            jnp.squeeze(prior.sample(seed=k, sample_shape=(1,))), low, high
        )
    prior_log_prob = prior.log_prob

    model_cfg = dict(
        model_type=model_type,
        dim=dim,
        num_layers=model_num_layers,
        num_hid=model_num_hid,
    )
    fwd_model = make_gbs_model(**model_cfg)
    bwd_model = make_gbs_model(**model_cfg)

    key, k1, k2 = jax.random.split(key, 3)
    fwd_params = fwd_model.init(
        k1, jnp.ones([batch_size, dim]), jnp.ones([batch_size, 1]), jnp.ones([batch_size, dim])
    )
    bwd_params = bwd_model.init(
        k2, jnp.ones([batch_size, dim]), jnp.ones([batch_size, 1]), jnp.ones([batch_size, dim])
    )

    opt = optax.chain(optax.zero_nans(), optax.clip(1.0), optax.adam(lr))
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=opt)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=opt)

    if loss_mode == "tr_lv":
        rnd_jit = jax.jit(rnd_time_reversal_lv_no_target, static_argnums=(3, 4, 5, 6, 7))

        def loss_wrapped(key, model_state, fwd_params, bwd_params, lam, policy_p):
            x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
                key,
                model_state,
                fwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                proc,
                True,
                process_center=process_center,
            )
            target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam, policy_p)).reshape(-1)
            rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
            loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT, max_rnd=max_rnd)
            return loss, aux

        loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))
        hist = {
            k: []
            for k in ["train/rnd_mean", "train/rnd_var", "train/xT_mean_norm", "train/n_filtered"]
        }
    elif loss_mode == "tr_lv_subtraj":
        if model_type != "potential":
            raise ValueError("loss_mode='tr_lv_subtraj' requires model_type='potential'.")
        rnd_jit = jax.jit(rnd_time_reversal_lv_no_target, static_argnums=(3, 4, 5, 6, 7))

        def loss_wrapped(key, model_state, fwd_params, bwd_params, lam, policy_p):
            del bwd_params
            use_subtraj = jax.random.uniform(jax.random.fold_in(key, 23)) < 0.5

            def _subtraj_branch(_):
                x_start, xT, rnd_running, idx_init, idx_end = (
                    rnd_time_reversal_lv_subtraj_no_target(
                        key,
                        model_state,
                        fwd_params,
                        batch_size,
                        prior_sampler,
                        num_steps,
                        proc,
                        True,
                        False,
                        None,
                        low,
                        high,
                        True,
                        None,
                        None,
                        process_center,
                    )
                )
                target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam, policy_p)).reshape(-1)
                loss, aux, _ = lv_subtraj_loss(
                    fwd_state=model_state[0],
                    fwd_params=fwd_params,
                    x_start=x_start,
                    x_end=xT,
                    rnd_running=rnd_running,
                    idx_init=idx_init,
                    idx_end=idx_end,
                    num_steps=num_steps,
                    prior_log_prob=prior_log_prob,
                    target_lp_vals=target_lp_vals,
                    max_rnd=max_rnd,
                )
                return loss, aux

            def _full_branch(_):
                x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
                    key,
                    model_state,
                    fwd_params,
                    batch_size,
                    prior_sampler,
                    num_steps,
                    proc,
                    True,
                    process_center=process_center,
                )
                target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam, policy_p)).reshape(-1)
                rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
                loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT, max_rnd=max_rnd)
                aux = add_subtraj_aux_defaults(
                    aux,
                    idx_init=0.0,
                    idx_end=float(num_steps),
                    scale=1.0,
                )
                return loss, aux

            loss, aux = jax.lax.cond(use_subtraj, _subtraj_branch, _full_branch, operand=None)
            return loss, aux

        loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))
        hist = {
            k: []
            for k in [
                "train/rnd_mean",
                "train/rnd_var",
                "train/xT_mean_norm",
                "train/n_filtered",
                "train/subtraj_scale",
                "train/subtraj_idx_init",
                "train/subtraj_idx_end",
            ]
        }
    elif loss_mode == "dis":
        rnd_jit = jax.jit(rnd_no_target, static_argnums=(4, 5, 6, 7, 8))

        def loss_wrapped(key, model_state, fwd_params, bwd_params, lam, policy_p):
            x0, xT, log_ratio = rnd_no_target(
                key,
                model_state,
                fwd_params,
                bwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                proc,
                True,
                process_center=process_center,
            )
            target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam, policy_p)).reshape(-1)
            loss, aux = lv_loss_from_values(
                x0, xT, log_ratio, prior_log_prob, target_lp_vals, max_rnd=max_rnd
            )
            return loss, aux

        loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))
        hist = {k: [] for k in [
            "train/neg_elbo_mean",
            "train/neg_elbo_var",
            "train/running_mean",
            "train/terminal_mean",
            "train/xT_mean_norm",
            "train/n_filtered",
        ]}
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    hist.update(
        {
            "target4/p": [],
            "target4/lambda": [],
            "target4/sample_mean": [],
            "target4/forward_kl": [],
            "target4/reverse_kl": [],
            "target4/wasserstein": [],
            "target4/sinkhorn": [],
            "target4/ess": [],
            "target4/energy_w2": [],
            "target4/interatomic_w2": [],
            "target4/target_mean": [],
            "target4/optimal_p": [],
            "target4/p_updated": [],
            "target4/p_jumped": [],
            "target4/p_base": [],
            "target4/p_ema": [],
        }
    )

    frames = []
    snapshot_records = []

    for t in trange(T):
        current_lambda = beta * p
        current_policy_p = p
        key, k_step = jax.random.split(key)
        model_state = (fwd_state, bwd_state)

        if loss_mode == "tr_lv":
            x0, xT_latent, _rnd_running = rnd_jit(
                k_step, model_state, fwd_state.params,
                batch_size, prior_sampler, num_steps, proc, True,
                process_center=process_center,
            )
        elif loss_mode == "tr_lv_subtraj":
            x0, xT_latent, _rnd_running = rnd_jit(
                k_step, model_state, fwd_state.params,
                batch_size, prior_sampler, num_steps, proc, True,
                process_center=process_center,
            )
        else:
            x0, xT_latent, _log_ratio = rnd_jit(
                k_step, model_state, fwd_state.params, bwd_state.params,
                batch_size, prior_sampler, num_steps, proc, True,
                process_center=process_center,
            )
        xT = to_box(xT_latent)

        (fwd_grads, bwd_grads), aux = loss_grad(
            k_step,
            model_state,
            fwd_state.params,
            bwd_state.params,
            jnp.asarray(current_lambda),
            jnp.asarray(current_policy_p, dtype=jnp.float32),
        )
        fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
        bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

        for k in list(aux.keys()):
            if k in hist:
                hist[k].append(float(aux[k]))

        sample_mean_g = float(jnp.mean(target4_energy_values(xT, target_params, policy_p=current_policy_p)))
        key, k_metric, k_update = jax.random.split(key, 3)
        if t in metric_eval_iters:
            forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
                xT,
                current_lambda,
                target_params=target_params,
                num_bins=metric_num_bins,
                key=k_metric,
                policy_p=current_policy_p,
            )
            key, k_sink = jax.random.split(key)
            sinkhorn_target = sample_truncated_exponential(
                k_sink, current_lambda, xT.shape, target_params=target_params, policy_p=current_policy_p
            )
            n_sink = min(int(sinkhorn_num_samples), int(xT.shape[0]))
            sinkhorn = sinkhorn_distance(xT[:n_sink], sinkhorn_target[:n_sink])
            ess = effective_sample_size_from_log_weights(
                target4_logprob(xT, current_lambda, target_params=target_params, policy_p=current_policy_p)
            )
            energy_w2 = float(
                energy_wasserstein_1d(
                    xT[:n_sink],
                    sinkhorn_target[:n_sink],
                    current_lambda,
                    target_params=target_params,
                    policy_p=current_policy_p,
                )
            )
            if should_compute_interatomic_w2(n_particles):
                interatomic_w2 = float(
                    interatomic_wasserstein_1d(
                        xT[:n_sink],
                        sinkhorn_target[:n_sink],
                        n_particles=n_particles,
                        n_spatial_dim=n_spatial_dim,
                    )
                )
            else:
                interatomic_w2 = float("nan")
        else:
            forward_kl = float("nan")
            reverse_kl = float("nan")
            wasserstein = float("nan")
            sinkhorn = float("nan")
            ess = float("nan")
            energy_w2 = float("nan")
            interatomic_w2 = float("nan")
        optimal_p, target_mean = optimal_p_from_target_mean(
            current_lambda,
            tau,
            q,
            target_params=target_params,
        )

        hist["target4/p"].append(float(p))
        hist["target4/lambda"].append(float(current_lambda))
        hist["target4/sample_mean"].append(sample_mean_g)
        hist["target4/forward_kl"].append(forward_kl)
        hist["target4/reverse_kl"].append(reverse_kl)
        hist["target4/wasserstein"].append(wasserstein)
        hist["target4/sinkhorn"].append(sinkhorn)
        hist["target4/ess"].append(ess)
        hist["target4/energy_w2"].append(energy_w2)
        hist["target4/interatomic_w2"].append(interatomic_w2)
        hist["target4/target_mean"].append(target_mean)
        hist["target4/optimal_p"].append(optimal_p)
        should_update_p = p_update_freq > 0 and ((t + 1) % p_update_freq == 0)
        hist["target4/p_updated"].append(float(should_update_p))
        hist["target4/p_jumped"].append(0.0)
        hist["target4/p_base"].append(float(jax.nn.sigmoid(tau * (sample_mean_g - q))))
        hist["target4/p_ema"].append(float(p))
        if should_update_p:
            p, base_p, ema_p, jumped = update_p_with_ema_and_jump(
                prev_p=p,
                sample_mean_g=sample_mean_g,
                tau=tau,
                q=q,
                ema_alpha=p_ema_alpha,
                jump_prob=p_jump_prob,
                key=k_update,
            )
            hist["target4/p"][-1] = float(p)
            hist["target4/p_jumped"][-1] = float(jumped)
            hist["target4/p_base"][-1] = float(base_p)
            hist["target4/p_ema"][-1] = float(ema_p)

        if gif_path and (t in snap_iters):
            frames.append(np.asarray(xT))
        if return_snapshots and (t in snap_iters):
            key, k_snap = jax.random.split(key)
            if loss_mode in ("tr_lv", "tr_lv_subtraj"):
                _, xT_snap_latent, _ = rnd_time_reversal_lv_no_target(
                    k_snap,
                    (fwd_state, bwd_state),
                    fwd_state.params,
                    snapshot_sample_size,
                    prior_sampler,
                    num_steps,
                    proc,
                    True,
                    process_center=process_center,
                )
            else:
                _, xT_snap_latent, _ = rnd_no_target(
                    k_snap,
                    (fwd_state, bwd_state),
                    fwd_state.params,
                    bwd_state.params,
                    snapshot_sample_size,
                    prior_sampler,
                    num_steps,
                    proc,
                    True,
                    process_center=process_center,
                )
            xT_snap = to_box(xT_snap_latent)
            snapshot_records.append(
                {
                    "iter": int(t),
                    "p": float(p),
                    "samples": np.asarray(xT_snap),
                }
            )

    if gif_path and frames:
        rendered = []
        for idx, pts in enumerate(frames):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, c="r")
            ax.set_xlim(float(low[0]), float(high[0]))
            ax.set_ylim(float(low[1]), float(high[1]))
            ax.set_aspect("equal")
            ax.set_title(f"GBS target4 snapshot {idx}")
            fig.tight_layout()
            fig.canvas.draw()
            rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
        imageio.mimsave(gif_path, rendered, fps=4)

    key, k_final = jax.random.split(key)
    B = 2**14
    if loss_mode in ("tr_lv", "tr_lv_subtraj"):
        _, xT_final_latent, _ = rnd_time_reversal_lv_no_target(
            k_final, (fwd_state, bwd_state), fwd_state.params,
            final_sample_size, prior_sampler, num_steps, proc, True, process_center=process_center
        )
    else:
        _, xT_final_latent, _ = rnd_no_target(
            k_final, (fwd_state, bwd_state), fwd_state.params, bwd_state.params,
            final_sample_size, prior_sampler, num_steps, proc, True, process_center=process_center
        )
    xT_final = to_box(xT_final_latent)
    np.save((save_dir / "gbs_samples.npy").as_posix(), np.array(xT_final))

    result = (fwd_state, bwd_state, hist, np.asarray(xT_final))
    if return_snapshots:
        return result + (snapshot_records,)
    return result
