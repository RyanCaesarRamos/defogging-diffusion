import argparse
import io
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple, Callable, Sequence, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from einops import rearrange
from ignite.metrics.gan.utils import InceptionModel, _BaseInceptionMetric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.engine import create_supervised_evaluator
from ignite.metrics import PSNR, SSIM, FID, InceptionScore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference batch npz file")
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()
    
    ref_batch = read_npz(args.sample_batch)
    sample_batch = read_npz(args.ref_batch)

    print("Computing evaluations...")
    metrics = calculate_metrics(sample_batch, ref_batch)
    
    print("PSNR:", metrics['psnr'])
    print("SSIM:", metrics['ssim'])
    print("FID:", metrics['fid'])
    print("Inception Score:", metrics['is'])
    print("Perceptual Distance:", metrics['pd'])
    
    
def calculate_metrics(sample_batch, ref_batch):
    evaluator = create_supervised_evaluator(DummyModel())
    
    psnr = PSNR(data_range=255)
    psnr.attach(evaluator, 'psnr')

    ssim = NewSSIM(data_range=255)
    ssim.attach(evaluator, 'ssim')

    fid = FID()
    fid.attach(evaluator, 'fid')

    in_sc = NewInceptionScore()
    in_sc.attach(evaluator, 'is')

    pd = PerceptualDistance()
    pd.attach(evaluator, 'pd')
    
    state = evaluator.run(list(zip(ref_batch, sample_batch)))
    return state.metrics
    
    
class DummyModel(nn.Module):
    def forward(self, x):
        return x


class NewInceptionScore(InceptionScore):
    # hack to let InceptionScore take in x, y when it only needs x
    def _extract_features(self, output):
        return super()._extract_features(output[0])    
    

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError



class NewSSIM(SSIM):
    # new SSIM that can process batches of different sizes

    @reinit__is_reduced
    def reset(self) -> None:
        # Not a tensor because batch size is not known in advance.
        self._sum_of_batchwise_ssim = 0.0  # type: Union[float, torch.Tensor]
        self._num_examples = 0
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)


    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        channel = y_pred.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1).to(device=y_pred.device)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([y_pred, y, y_pred * y_pred, y * y, y_pred * y])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * y_pred.size(0) : (x + 1) * y_pred.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        self._sum_of_batchwise_ssim += torch.sum(ssim_idx, dtype=torch.float64).to(self._device)
        self._num_examples += ssim_idx.numel()


    @sync_all_reduce("_sum_of_batchwise_ssim", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have at least one example before it can be computed.")
        return self._sum_of_batchwise_ssim / self._num_examples  # type: ignore[arg-type]

    
class PerceptualDistance(_BaseInceptionMetric):
    r"""Calculates Perceptual Distance.

    .. math::
       \text{PD} = \text{MSE}(x, y)

    where :math:`x` and :math:`y` refer to the representations of the generated data and real data
    in Inception-v3 feature space respectively.

    More details can be found in `Saharia et al. 2021`__

    __ https://arxiv.org/pdf/2111.05826.pdf

    Remark:

        This implementation uses Inception-v3 instead of Inception-v1 as originally proposed for
        simplicity.

    .. note::
        The default Inception model requires the `torchvision` module to be installed.

    Args:
        num_features: number of features predicted by the model or the reduced feature vector of the image.
            Default value is 2048.
        feature_extractor: a torch Module for extracting the features from the input data.
            It returns a tensor of shape (batch_size, num_features).
            If neither ``num_features`` nor ``feature_extractor`` are defined, by default we use an ImageNet
            pretrained Inception Model. If only ``num_features`` is defined but ``feature_extractor`` is not
            defined, ``feature_extractor`` is assigned Identity Function.
            Please note that the model will be implicitly converted to device mentioned in the ``device``
            argument.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

        .. code-block:: python

            y_pred, y = torch.rand(10, 3, 299, 299), torch.rand(10, 3, 299, 299)
            m = PerceptualDistance()
            m.update((y_pred, y))
            print(m.compute())
            
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)
            
        self.euclidean_distance = nn.PairwiseDistance(p=2)

        super(PerceptualDistance, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:

        self._num_examples: float = 0
        self._total_distance: int = 0

        super(PerceptualDistance, self).reset()


    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:

        train, test = output
        train_features = self._extract_features(train)
        test_features = self._extract_features(test)

        if train_features.shape[0] != test_features.shape[0] or train_features.shape[1] != test_features.shape[1]:
            raise ValueError(
                f"""
    Number of Training Features and Testing Features should be equal ({train_features.shape} != {test_features.shape})
                """
            )

        self._total_distance += self.euclidean_distance(train_features, test_features).sum().item()
        self._num_examples += train_features.shape[0]


    @sync_all_reduce("_num_examples", "_total_distance")
    def compute(self) -> float:

        pd = self._total_distance / self._num_examples

        return pd
            
            
def read_npz(npz_path: str, batch_size: int = 64) -> np.ndarray:
    with open_npz_array(npz_path, "arr_0") as reader:
        for batch in reader.read_batches(batch_size):
            batch = torch.from_numpy(batch.copy()).float()
            batch = rearrange(batch, 'b h w c -> b c h w')
            yield batch
    
    
class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()
    
    
class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)
    
    
@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)
            
            
def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data
            
            
@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f
    
    
if __name__ == "__main__":
    main()