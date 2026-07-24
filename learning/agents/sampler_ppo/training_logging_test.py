import unittest

from learning.agents.sampler_ppo.training_logging import training_log_offsets


class TrainingLoggingTest(unittest.TestCase):

  def test_offsets_are_aligned_to_global_update_steps(self):
    self.assertEqual(
        training_log_offsets(0, 12, 5, False),
        (4, 9),
    )
    self.assertEqual(
        training_log_offsets(12, 12, 5, False),
        (2, 7),
    )

  def test_final_step_can_be_reserved_for_evaluation_log(self):
    self.assertEqual(
        training_log_offsets(0, 10, 5, True),
        (4,),
    )
    self.assertEqual(
        training_log_offsets(0, 10, 5, False),
        (4, 9),
    )

  def test_zero_frequency_disables_extra_logs(self):
    self.assertEqual(training_log_offsets(0, 10, 0, False), ())


if __name__ == "__main__":
  unittest.main()
