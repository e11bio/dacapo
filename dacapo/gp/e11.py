import gunpowder as gp
import numpy as np
import random
import skimage
import logging
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt, binary_closing

logging.basicConfig(level=logging.INFO)


class CustomNormalize(gp.Normalize):
    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.spec.dtype = self.dtype

        factor = 1.0 / (np.max(array.data))
        array.data = array.data.astype(self.dtype) * factor


class CustomNoise(gp.NoiseAugment):
    def __init__(self, array, augment_every=0, **kwargs):
        super().__init__(array, **kwargs)
        self.array = array
        self.augment_every = augment_every

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        rand_int = random.randint(0, self.augment_every)

        if rand_int == 0:
            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
                "Noise augmentation requires float types for the raw array (not "
                + str(raw.data.dtype)
                + "). Consider using Normalize before."
            )

            if self.clip:
                assert (
                    raw.data.min() >= -1 and raw.data.max() <= 1
                ), "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."

            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, rng=self.seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)

        else:
            raw.data = raw.data


class ShuffleChannels(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def process(self, batch, request):
        array = batch.arrays[self.array]

        num_channels = array.data.shape[0]

        channel_perm = np.random.permutation(num_channels)

        array.data = array.data[channel_perm]


class Smooth(gp.BatchFilter):
    def __init__(self, array, sigma):
        self.array = array
        self.sigma = sigma

    def process(self, batch, request):
        array = batch[self.array]

        for c in range(array.data.shape[0]):
            array.data[c] = gaussian_filter(array.data[c], sigma=self.sigma)


class ChannelWiseIntensityAugment(gp.IntensityAugment):
    # assumes 4d raw data of c,z,y,x
    def __init__(self, array, augment_every=0, **kwargs):
        super().__init__(array, **kwargs)
        self.array = array
        self.augment_every = augment_every

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Intensity augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= 0 and raw.data.max() <= 1
            ), "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        rand_int = random.randint(0, self.augment_every)

        if rand_int == 0:
            for c in range(raw.data.shape[0]):
                raw.data[c] = self.__augment(
                    raw.data[c],
                    np.random.uniform(low=self.scale_min, high=self.scale_max),
                    np.random.uniform(low=self.shift_min, high=self.shift_max),
                )

            # clip values, we might have pushed them out of [0,1]
            if self.clip:
                raw.data[raw.data > 1] = 1
                raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):
        return a.mean() + (a - a.mean()) * scale + shift


class ExpandLabels(gp.BatchFilter):
    def __init__(self, labels, background=0, expansion_factor=1):
        self.labels = labels
        self.background = (background,)
        self.expansion_factor = expansion_factor

    def process(self, batch, request):
        labels = batch[self.labels].data
        expanded_labels = np.zeros_like(labels)

        z_slices = labels.shape[0]

        for z in range(z_slices):
            z_slice = labels[z]

            distances, indices = distance_transform_edt(
                z_slice == self.background, return_indices=True
            )

            dilate_mask = distances <= self.expansion_factor
            masked_indices = [
                dimension_indices[dilate_mask] for dimension_indices in indices
            ]
            nearest_labels = z_slice[tuple(masked_indices)]

            expanded_labels[z][dilate_mask] = nearest_labels

        batch[self.labels].data = expanded_labels


class FillHoles(gp.BatchFilter):
    def __init__(self, labels, iterations):
        self.labels = labels
        self.iterations = iterations

    def process(self, batch, request):
        labels = batch[self.labels].data

        filled_labels = np.zeros_like(labels)
        unique_labels = np.unique(labels)
        z_slices = labels.shape[0]

        for label in unique_labels:
            for z in range(z_slices):
                z_slice = labels[z]

                # pad to deal with borders
                padded_slice = np.pad(
                    z_slice == label, self.iterations, mode="constant"
                )

                filled_slice = binary_closing(
                    padded_slice,
                    structure=np.ones((3, 3)),
                    iterations=self.iterations,
                )

                # exclude padded regions
                filled_labels[z] = np.where(
                    filled_slice[
                        self.iterations : -self.iterations,
                        self.iterations : -self.iterations,
                    ],
                    label,
                    filled_labels[z],
                )

        batch[self.labels].data = filled_labels


class CreateMask(gp.BatchFilter):
    def __init__(self, labels, mask, mask_type="labels_mask"):
        self.labels = labels
        self.mask = mask
        self.mask_type = mask_type

    def setup(self):
        self.provides(self.mask, self.spec[self.labels].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.mask].copy()

        return deps

    def process(self, batch, request):
        labels = batch[self.labels].data

        valid_masks = ["labels_mask", "unlabelled"]

        assert (
            self.mask_type in valid_masks
        ), f"Invalid mask type: {self.mask_type}. Must be one of {valid_masks}"

        mask = (labels > 0) if self.mask_type == "unlabelled" else np.ones_like(labels)

        mask = mask.astype(np.uint8)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.mask].roi.copy()
        spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.mask] = gp.Array(mask, spec)

        return batch


def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):
    # todo: do this cleaner?

    if neighborhood is not None:
        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:
        method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()
