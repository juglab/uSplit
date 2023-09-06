import gunpowder as gp

raw = gp.ArrayKey('RAW')
source = gp.ZarrSource(
    'data2.zarr',  # the zarr container
    {raw: 'raw'},  # which dataset to associate to the array key
    {raw: gp.ArraySpec(interpolatable=True)}  # meta-information
)

dsample = gp.ArrayKey('Dsample')
voxel_size = gp.Coordinate((1, 1, 1, 1))
downsample_factor = gp.Coordinate(1, 2, 2, 1)
target_voxel_size = voxel_size * downsample_factor
pipeline = source + gp.Resample(raw, target_voxel_size, dsample)
#+ gp.RandomLocation()
pipeline = source + gp.DownSample(raw, 2, dsample)
request = gp.BatchRequest()
request[raw] = gp.Roi((1, 33, 33, 1), (1, 64, 64, 1))
request[dsample] = gp.Roi((1, 2, 2, 1), (1, 128, 128, 1))
with gp.build(pipeline):
    batch = pipeline.request_batch(request)
batch[dsample].shape
batch[raw].data
batch[raw].data.shape
batch[dsample].data.shape
