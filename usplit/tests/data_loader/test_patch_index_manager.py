from usplit.data_loader.patch_index_manager import GridAlignement, GridIndexManager


def test_grid_index_manager_idx_to_hwt_mapping():
    grid_size = 32
    patch_size = 64
    index = 13
    manager = GridIndexManager((5, 499, 469, 2), grid_size, patch_size, GridAlignement.Center)
    h_start, w_start = manager.get_deterministic_hw(index)
    print(h_start, w_start, manager.grid_count())
    print(manager.grid_rows(grid_size), manager.grid_cols(grid_size))

    for grid_size in [1, 2, 4, 8, 16, 32, 64]:
        hwt = manager.hwt_from_idx(index, grid_size=grid_size)
        same_index = manager.idx_from_hwt(*hwt, grid_size=grid_size)
        assert index == same_index, f'{index}!={same_index}'


if __name__ == '__main__':
    test_grid_index_manager_idx_to_hwt_mapping()
