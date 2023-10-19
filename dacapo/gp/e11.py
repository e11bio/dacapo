from scipy.ndimage import distance_transform_edt, binary_closing
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid
from skimage.util import random_noise
import gunpowder as gp
import logging
import numpy as np
import random

logging.basicConfig(level=logging.INFO)


class ChannelWiseNoise(gp.NoiseAugment):
    def __init__(self, array, **kwargs):
        super().__init__(array, **kwargs)
        self.array = array

    def skip_node(self, request):
        return random.random() > self.p

    def _apply_noise(self, data):
        return random_noise(
            data, mode=self.mode, rng=self.seed, clip=self.clip, **self.kwargs
        ).astype(data.dtype)

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Noise augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )

        if self.clip:
            assert (
                raw.data.min() >= -1 and raw.data.max() <= 1
            ), "Noise augmentation expects raw values in [-1,1] or [0,1]. \
                    Consider using Normalize before."

        for c in range(raw.data.shape[0]):
            raw.data[c] = self._apply_noise(raw.data[c])


class ChannelWiseIntensityAugment(gp.IntensityAugment):
    # assumes 4d raw data of c,z,y,x
    def __init__(self, array, **kwargs):
        super().__init__(array, **kwargs)
        self.array = array

    def skip_node(self, request):
        return random.random() > self.p

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
            ), "Intensity augmentation expects raw values in [0,1]. Consider \
                    using Normalize before."

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


class ShuffleChannels(gp.BatchFilter):
    def __init__(self, array, p=1.0):
        self.array = array
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        array = batch.arrays[self.array]

        num_channels = array.data.shape[0]

        channel_perm = np.random.permutation(num_channels)

        array.data = array.data[channel_perm]


class Blur(gp.BatchFilter):
    def __init__(self, array, blur_range, p=1.0):
        self.array = array
        self.range = blur_range

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        array = batch[self.array].data

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(self.range[0], self.range[1])

        for z in range(array.shape[1]):
            array_sec = array[:, z]

            array[:, z] = np.array(
                [
                    gaussian_filter(array_sec[i], sigma=sigma)
                    for i in range(array_sec.shape[0])
                ]
            ).astype(array_sec.dtype)

        batch[self.array].data = array


class ZeroChannels(gp.BatchFilter):
    def __init__(self, array, num_channels=0, p=1.0):
        self.array = array
        self.num_channels = num_channels
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def get_bounds(self, size):
        start, end = np.random.randint(0, size // 3), np.random.randint(
            2 * size // 3, size
        )

        return start, end

    def draw_random_shape(self, z_size, y_size, x_size, shape_type="ellipsoid"):
        # Define random start and end points
        start_z, end_z = self.get_bounds(z_size)
        start_y, end_y = self.get_bounds(y_size)
        start_x, end_x = self.get_bounds(x_size)

        if shape_type == "ellipsoid":
            z_radius = (end_z - start_z) // 2
            y_radius = (end_y - start_y) // 2
            x_radius = (end_x - start_x) // 2

            ellipsoid_volume = ellipsoid(x_radius, y_radius, z_radius)
            zz, yy, xx = np.where(ellipsoid_volume)

            zz += start_z + z_radius
            yy += start_y + y_radius
            xx += start_x + x_radius

            valid_mask = (zz < z_size) & (yy < y_size) & (xx < x_size)

            return zz[valid_mask], yy[valid_mask], xx[valid_mask]

        elif shape_type == "points":
            max_radius = random.randint(3, 10)
            k = random.randint(10, 100)
            zz, yy, xx = [], [], []

            for _ in range(k):
                center_z = np.random.randint(max_radius, z_size - max_radius)
                center_y = np.random.randint(max_radius, y_size - max_radius)
                center_x = np.random.randint(max_radius, x_size - max_radius)

                # Randomly determine the size of the ellipsoid (spherical, so
                # all radii are the same)
                radius = np.random.randint(1, max_radius)

                ellipsoid_volume = ellipsoid(radius, radius, radius)
                zz, yy, xx = np.where(ellipsoid_volume)

                # Adjust coordinates based on the center
                zz += center_z - radius
                yy += center_y - radius
                xx += center_x - radius

                zz.extend(zz)
                yy.extend(yy)
                xx.extend(xx)

            zz = np.array(zz)
            yy = np.array(yy)
            xx = np.array(xx)

            valid_mask = (zz < z_size) & (yy < y_size) & (xx < x_size)

            return zz[valid_mask], yy[valid_mask], xx[valid_mask]

    def process(self, batch, request):
        data = batch.arrays[self.array].data

        assert len(data.shape) == 4, "Data must be 4-dimensional."

        # Choose a random number up to num_channels
        channels_to_zero_out = np.random.randint(1, self.num_channels + 1)

        # Randomly select the channels
        channels_to_zero = np.random.choice(
            data.shape[0], channels_to_zero_out, replace=False
        )

        use_shape = random.choice([True, False])  # 50/50 chance

        if use_shape:
            _, z, y, x = data.shape

            shape_type = np.random.choice(["ellipsoid", "points"])

            zz, yy, xx = self.draw_random_shape(z, y, x, shape_type=shape_type)

            # Ensure that the coordinates are within bounds
            valid_mask = (
                (zz >= 0) & (zz < z) & (yy >= 0) & (yy < y) & (xx >= 0) & (xx < x)
            )

            zz, yy, xx = zz[valid_mask], yy[valid_mask], xx[valid_mask]

            # Zero out selected channels in shape
            for channel in channels_to_zero:
                data[channel, zz, yy, xx] = 0

        else:
            # zero out whole selected channels
            data[channels_to_zero] = 0

        batch[self.array].data = data


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
