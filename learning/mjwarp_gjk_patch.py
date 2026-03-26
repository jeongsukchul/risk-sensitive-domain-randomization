"""Runtime patch for vendored MJWarp GJK convergence behavior.

This applies the upstream fix from mujoco_warp PR #1239 in-process by
redefining `collision_gjk.gjk` before Warp kernels are first compiled.
"""

from __future__ import annotations


def apply_gjk_stall_patch(verbose: bool = True) -> bool:
  """Patch vendored MJWarp GJK with the stalled-progress termination fix.

  Returns:
    True if the patch is active after this call, False otherwise.
  """
  try:
    import warp as wp
    from mujoco.mjx.third_party.mujoco_warp._src import collision_gjk as mod
    from mjwarp_gjk_override import gjk as patched_gjk
  except Exception as exc:  # pragma: no cover - defensive import guard.
    if verbose:
      print(f"[mjwarp_gjk_patch] skipped: could not import vendored MJWarp ({exc})")
    return False

  if getattr(mod, "_RSDR_GJK_PATCH_APPLIED", False):
    if verbose:
      print("[mjwarp_gjk_patch] already active")
    return True

  mod._RSDR_GJK_ORIGINAL = mod.gjk
  mod.gjk = patched_gjk
  mod._RSDR_GJK_PATCH_APPLIED = True
  mod._RSDR_GJK_PATCH_SOURCE = getattr(patched_gjk, "func", None)

  # Ensure Warp does not reuse kernels compiled against the old function body.
  wp.clear_kernel_cache()

  if verbose:
    print("[mjwarp_gjk_patch] applied vendored MJWarp GJK convergence patch")

  return True
