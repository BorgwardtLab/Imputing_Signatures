import unittest

import numpy as np
import torch

from src.utils.train_utils import build_regularly_spaced_grid


class TestGrid(unittest.TestCase):
    def test_regularly_sampled_grid(self):
        max_time = 5.
        n_tasks = 10
        generated_grid_spacing = 0.5
        query_grid_spacing = 1.0
        inputs = np.arange(
            0, max_time + generated_grid_spacing, generated_grid_spacing)[np.newaxis, :]
        indices = np.random.randint(0, n_tasks, size=inputs.shape[1])[np.newaxis, :]
        out_inputs, out_indices = build_regularly_spaced_grid(
            torch.tensor(inputs), torch.tensor(indices), 1, n_tasks, query_grid_spacing)
        test = np.array(out_inputs).reshape(-1, n_tasks)
        gt = np.tile(np.arange(0, max_time+1., 1.)[:, np.newaxis], [1, n_tasks])
        self.assertTrue((test == gt).all())


if __name__ == '__main__':
    unittest.main()
