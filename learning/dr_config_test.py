from absl.testing import absltest

from learning.dr_config import build_dr_spec


class DomainRandomizationConfigTest(absltest.TestCase):

  def test_build_rotate_z_axis_spec(self):
    spec = build_dr_spec("LeapCubeRotateZAxis")
    self.assertIsNotNone(spec)
    self.assertEqual(spec.task, "LeapCubeRotateZAxis")
    self.assertEqual(spec.full_dim, 124)
    self.assertEqual(spec.learnable_dim, 27)

  def test_build_rotate_z_axis_spec_without_reset_params(self):
    spec = build_dr_spec(
        "LeapCubeRotateZAxis",
        include_reset_params=False,
    )
    self.assertIsNotNone(spec)
    self.assertEqual(spec.full_dim, 102)
    self.assertEqual(spec.learnable_dim, 5)

  def test_build_rotate_z_axis_spec_reset_only_learning(self):
    spec = build_dr_spec(
        "LeapCubeRotateZAxis",
        enable_dynamics_learning=False,
        enable_reset_learning=True,
    )
    self.assertIsNotNone(spec)
    self.assertEqual(spec.learnable_dim, 22)


if __name__ == "__main__":
  absltest.main()
