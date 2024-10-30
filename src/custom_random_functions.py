import itertools
import numpy as np
import os
import src


class NumberSampler:
    def __init__(self):
        self.initialize_sources()

    def initialize_sources(self):
        self.uniform_data = self.load_uniform()
        self.normal_data = self.load_normal()

    def load_uniform(self):
        uniform_file = os.path.join(self.get_root_dir(), "uniforms.npy")
        uniform_values = np.load(uniform_file, allow_pickle=False)
        return iter(uniform_values)

    def load_normal(self):
        normal_file = os.path.join(self.get_root_dir(), "normals.npy")
        normal_values = np.load(normal_file, allow_pickle=False)
        return iter(normal_values)

    @staticmethod
    def get_root_dir():
        return os.path.abspath(os.path.join(os.path.dirname(src.__file__), "..", ".."))

    def sample_uniform(self):
        try:
            return next(self.uniform_data)
        except StopIteration:
            self.uniform_data = self.load_uniform()
            return next(self.uniform_data)

    def sample_normal(self):
        try:
            return next(self.normal_data)
        except StopIteration:
            self.normal_data = self.load_normal()
            return next(self.normal_data)


def int_in_range(start, end, sampler):
    return int(start + sampler.sample_uniform() * (end - start))


def float_in_range(start, end, sampler):
    return start + sampler.sample_uniform() * (end - start)


def normal_in_range(mean, stddev, sampler):
    return mean + sampler.sample_normal() * stddev


def randint(start, end=None, shape=None):
    end = start if end is None else end
    if shape is None:
        return int_in_range(start, end, sampler_instance)
    elif isinstance(shape, int):
        return np.array([int_in_range(start, end, sampler_instance) for _ in range(shape)], dtype=int)
    else:
        result = np.zeros(shape, dtype=int)
        for idx in itertools.product(*[range(dim) for dim in shape]):
            result[idx] = int_in_range(start, end, sampler_instance)
        return result


def uniform(start, end=None, shape=None):
    end = start if end is None else end
    if shape is None:
        return float_in_range(start, end, sampler_instance)
    elif isinstance(shape, int):
        return np.array([float_in_range(start, end, sampler_instance) for _ in range(shape)])
    else:
        result = np.zeros(shape)
        for idx in itertools.product(*[range(dim) for dim in shape]):
            result[idx] = float_in_range(start, end, sampler_instance)
        return result


def pick_from_array(arr, shape=None, replace=True, probs=None):
    if shape is not None or probs is not None or not replace:
        raise NotImplementedError("Custom 'pick_from_array' does not support these parameters.")
    
    if isinstance(arr, int):
        arr = np.arange(arr)
    elif isinstance(arr, (list, tuple)):
        assert arr, "Cannot pick from an empty list or tuple."
    elif isinstance(arr, np.ndarray):
        assert arr.size > 0, "Cannot pick from an empty array."
        assert arr.ndim == 1, "Array must be 1-dimensional."
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")

    index = int_in_range(0, len(arr), sampler_instance)
    return arr[index]


def random_value(shape=None):
    return uniform(start=0, end=1, shape=shape)


def normal_value(mean=0, stddev=1, shape=None):
    if shape is None:
        return normal_in_range(mean, stddev, sampler_instance)
    elif isinstance(shape, int):
        return np.array([normal_in_range(mean, stddev, sampler_instance) for _ in range(shape)])
    else:
        result = np.zeros(shape)
        for idx in itertools.product(*[range(dim) for dim in shape]):
            result[idx] = normal_in_range(mean, stddev, sampler_instance)
        return result


sampler_instance = NumberSampler()