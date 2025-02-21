import numpy as np

from usplit.analysis.stitch_prediction import (_get_location, set_skip_boundary_pixels_mask,
                                               set_skip_central_pixels_mask, stitch_predictions,
                                               stitched_prediction_mask)
from usplit.data_loader.patch_index_manager import TilingMode, GridIndexManager


def test_skipping_boundaries():
    mask = np.full((10, 2, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_boundary_pixels_mask(mask, loc1, 1)
    set_skip_boundary_pixels_mask(mask, loc2, 1)
    correct_mask = np.full((10, 2, 8, 8), 1)
    # boundary for hwt1
    correct_mask[0, :, 0, [0, 1, 2, 3]] = False
    correct_mask[0, :, 3, [0, 1, 2, 3]] = False
    correct_mask[0, :, [0, 1, 2, 3], 0] = False
    correct_mask[0, :, [0, 1, 2, 3], 3] = False

    # boundary for hwt2
    correct_mask[2, :, 4, [4, 5, 6, 7]] = False
    correct_mask[2, :, 7, [4, 5, 6, 7]] = False
    correct_mask[2, :, [4, 5, 6, 7], 4] = False
    correct_mask[2, :, [4, 5, 6, 7], 7] = False
    assert (mask == correct_mask).all()


def test_picking_boundaries():
    mask = np.full((10, 2, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_central_pixels_mask(mask, loc1, 1)
    set_skip_central_pixels_mask(mask, loc2, 2)
    correct_mask = np.full((10, 2, 8, 8), 1)
    # boundary for hwt1
    correct_mask[0, :, 2, 2] = False
    # boundary for hwt2
    correct_mask[2, :, 5:7, 5:7] = False

    print(mask[hwt2[-1]])
    assert (mask == correct_mask).all()


class DummyDset:

    def __init__(self, grid_size, patch_size, data_shape) -> None:
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.data_shape = data_shape
        idx_manager = GridIndexManager(data_shape, grid_size, patch_size, TilingMode.ShiftBoundary)
        self.idx_manager = idx_manager

    def per_side_overlap_pixelcount(self):
        return (self.patch_size - self.grid_size) // 2

    def get_data_shape(self):
        return self.data_shape

    def get_grid_size(self):
        return self.grid_size


def test_stitch_predictions_square_frames():
    grid_size = 32
    patch_size = 64
    data_shape = (30, 1550, 1550, 2)
    N = data_shape[0] * (data_shape[1] // grid_size) * (data_shape[2] // grid_size)
    predictions = np.zeros((N, 2, patch_size, patch_size))
    dset = DummyDset(grid_size, patch_size, data_shape)
    output = stitch_predictions(predictions, dset)


def test_stitch_predictions_non_square_frames():
    grid_size = 32
    patch_size = 64
    data_shape = (30, 1550, 1920, 2)
    N = data_shape[0] * (data_shape[1] // grid_size) * (data_shape[2] // grid_size)
    predictions = np.zeros((N, 2, patch_size, patch_size))
    dset = DummyDset(grid_size, patch_size, data_shape)
    output = stitch_predictions(predictions, dset)

    # NOTE: masking is disabled. so are its tests
    # skip_boundary_pixel_count = 0
    # skip_central_pixel_count = 0
    # mask1 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)
    # assert (mask1 == 1).all()

    # skip_boundary_pixel_count = 2
    # skip_central_pixel_count = 0
    # mask2 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    # skip_boundary_pixel_count = 0
    # skip_central_pixel_count = 4
    # mask3 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    # assert ((mask2 + mask3) == 1).all()

    # skip_boundary_pixel_count = 1
    # skip_central_pixel_count = 2
    # mask4 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    # import matplotlib.pyplot as plt;
    # plt.imshow(mask4[0, :, :, 0]);
    # plt.show()
