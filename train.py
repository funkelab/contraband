import funlib.learn.torch
import gunpowder as gp
import logging
import math
import numpy as np
import torch
import zarr
from model import ContrastiveVolumeNet, contrastive_volume_loss

logging.basicConfig(level=logging.INFO)


class InspectBatch(gp.BatchFilter):

    def __init__(self, prefix):
        self.prefix = prefix

    def process(self, batch, request):
        for key, graph in batch.graphs.items():
            print(f"{self.prefix} ======== {key}: {graph}")


class RemoveChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[0]


class AddRandomPoints(gp.BatchFilter):

    def __init__(self, graph_key, for_array, density):
        self.graph_key = graph_key
        self.array_key = for_array
        self.n = density
        self.seed = 0

    def setup(self):
        self.graph_spec = gp.GraphSpec(roi=self.spec[self.array_key].roi)
        self.provides(self.graph_key, self.graph_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array_key] = request[self.array_key]
        return deps

    def process(self, batch, request):

        # ensure that parallel calls to this node produce the same
        # pseudo-random output
        rand_state = np.random.get_state()
        np.random.seed(self.seed)
        self.seed += 1

        # create random points in [0:1000,0:1000]
        points = np.random.random((int(self.n*1000*1000), 3))*1000
        points[:, 0] = request[self.graph_key].roi.get_begin()[0]

        # restore the RNG
        np.random.set_state(rand_state)

        keep = {}
        for i, point in enumerate(points):
            if request[self.graph_key].roi.contains(point):
                keep[i] = point

        batch[self.graph_key] = gp.Graph(
            [gp.Node(id=i, location=l) for i, l in keep.items()],
            [],
            gp.GraphSpec(roi=request[self.graph_key].roi))


class PrepareBatch(gp.BatchFilter):

    def __init__(
            self,
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1):
        self.raw_0 = raw_0
        self.raw_1 = raw_1
        self.points_0 = points_0
        self.points_1 = points_1
        self.locations_0 = locations_0
        self.locations_1 = locations_1

    def setup(self):
        self.provides(
            self.locations_0,
            gp.ArraySpec(nonspatial=True))
        self.provides(
            self.locations_1,
            gp.ArraySpec(nonspatial=True))

    def process(self, batch, request):

        ids_0 = set([n.id for n in batch[self.points_0].nodes])
        ids_1 = set([n.id for n in batch[self.points_1].nodes])
        common_ids = ids_0.intersection(ids_1)

        locations_0 = []
        locations_1 = []
        # get list of only xy locations
        # locations are in voxels, relative to output roi
        points_roi = request[self.points_0].roi
        voxel_size = batch[self.raw_0].spec.voxel_size
        for i in common_ids:
            location_0 = np.array(batch[self.points_0].node(i).location)
            location_1 = np.array(batch[self.points_1].node(i).location)
            if not points_roi.contains(location_0):
                print(f"skipping point {i} at {location_0}")
                continue
            if not points_roi.contains(location_1):
                print(f"skipping point {i} at {location_1}")
                continue
            location_0 -= points_roi.get_begin()
            location_1 -= points_roi.get_begin()
            location_0 /= voxel_size
            location_1 /= voxel_size
            locations_0.append(location_0[1:])
            locations_1.append(location_1[1:])

        # create point location arrays (with batch dimension)
        batch[self.locations_0] = gp.Array(
            np.array([locations_0], dtype=np.float32),
            self.spec[self.locations_0])
        batch[self.locations_1] = gp.Array(
            np.array([locations_1], dtype=np.float32),
            self.spec[self.locations_1])

        # add batch dimension to raw
        batch[self.raw_0].data = batch[self.raw_0].data[np.newaxis, :]
        batch[self.raw_1].data = batch[self.raw_1].data[np.newaxis, :]

        # make sure raw is float32
        batch[self.raw_0].data = batch[self.raw_0].data.astype(np.float32)
        batch[self.raw_1].data = batch[self.raw_1].data.astype(np.float32)


class AddSpatialDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def setup(self):

        upstream_spec = self.get_upstream_provider().spec[self.array]

        spec = upstream_spec.copy()
        if spec.roi is not None:
            spec.roi = gp.Roi(
                self.__insert_dim(spec.roi.get_begin(), 0),
                self.__insert_dim(spec.roi.get_shape(), 1))
        if spec.voxel_size is not None:
            spec.voxel_size = self.__insert_dim(spec.voxel_size, 1)
        self.spec[self.array] = spec

    def prepare(self, request):

        if self.array not in request:
            return

        upstream_spec = self.get_upstream_provider().spec[self.array]

        request[self.array].roi = gp.Roi(
            self.__remove_dim(request[self.array].roi.get_begin()),
            self.__remove_dim(request[self.array].roi.get_shape()))
        if request[self.array].voxel_size is not None:
            request[self.array].voxel_size = self.__remove_dim(
                request[self.array].voxel_size)

    def process(self, batch, request):

        if self.array not in request:
            return

        array = batch[self.array]

        array.spec.roi = gp.Roi(
            self.__insert_dim(array.spec.roi.get_begin(), 0),
            self.__insert_dim(array.spec.roi.get_shape(), 1))
        array.spec.voxel_size = self.__insert_dim(array.spec.voxel_size, 1)
        array.data = array.data[:, :, np.newaxis, :, :]

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

    def __insert_dim(self, a, s, dim=0):
        return a[:dim] + (s,) + a[dim:]


if __name__ == "__main__":

    num_iterations = int(1e6)

    unet = funlib.learn.torch.models.UNet(
        1, 12, 6,
        [(2, 2), (2, 2), (2, 2)],
        kernel_size_down=[[(3, 3), (3, 3)]]*4,
        kernel_size_up=[[(3, 3), (3, 3)]]*3,
        constant_upsample=True)
    model = ContrastiveVolumeNet(unet, 20, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    temperature = 1.0

    def loss(emb_0, emb_1, locations_0, locations_1):
        return contrastive_volume_loss(
            emb_0,
            emb_1,
            locations_0,
            locations_1,
            temperature)

    filename = 'data/ctc/Fluo-N2DH-SIM+.zarr'

    raw_0 = gp.ArrayKey('RAW_0')
    points_0 = gp.GraphKey('POINTS_0')
    locations_0 = gp.ArrayKey('LOCATIONS_0')
    emb_0 = gp.ArrayKey('EMBEDDING_0')
    raw_1 = gp.ArrayKey('RAW_1')
    points_1 = gp.GraphKey('POINTS_1')
    locations_1 = gp.ArrayKey('LOCATIONS_1')
    emb_1 = gp.ArrayKey('EMBEDDING_1')

    request = gp.BatchRequest()
    request.add(raw_0, (1, 260, 260))
    request.add(raw_1, (1, 260, 260))
    request.add(points_0, (1, 168, 168))
    request.add(points_1, (1, 168, 168))
    request[locations_0] = gp.ArraySpec(nonspatial=True)
    request[locations_1] = gp.ArraySpec(nonspatial=True)

    snapshot_request = gp.BatchRequest()
    snapshot_request[emb_0] = gp.ArraySpec(roi=request[points_0].roi)
    snapshot_request[emb_1] = gp.ArraySpec(roi=request[points_1].roi)

    source_shape = zarr.open(filename)['train/raw'].shape
    raw_roi = gp.Roi((0, 0, 0), source_shape)
    sources = tuple(
        gp.ZarrSource(
            filename,
            {
                raw: 'train/raw'
            },
            # fake 3D data
            array_specs={
                raw: gp.ArraySpec(
                    roi=raw_roi,
                    voxel_size=(1, 1, 1),
                    interpolatable=True)
            }) +
        gp.Normalize(raw, factor=1.0/4) +
        gp.Pad(raw, (0, 200, 200)) +
        AddRandomPoints(points, for_array=raw, density=0.0005) +
        gp.ElasticAugment(
            control_point_spacing=(1, 10, 10),
            jitter_sigma=(0, 0.1, 0.1),
            rotation_interval=(0, math.pi/2)) # +
        # gp.SimpleAugment(
            # mirror_only=(1, 2),
            # transpose_only=(1, 2)) +
        # gp.NoiseAugment(raw, var=0.01)

        for raw, points in zip([raw_0, raw_1], [points_0, points_1])
    )

    pipeline = (
        sources +
        gp.MergeProvider() +
        gp.Crop(raw_0, raw_roi) +
        gp.RandomLocation() +
        PrepareBatch(
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1) +
        gp.PreCache() +
        gp.torch.Train(
            model, loss, optimizer,
            inputs={
                'raw_0': raw_0,
                'raw_1': raw_1
            },
            loss_inputs={
                'emb_0': emb_0,
                'emb_1': emb_1,
                'locations_0': locations_0,
                'locations_1': locations_1
            },
            outputs={
                2: emb_0,
                3: emb_1
            },
            array_specs={
                emb_0: gp.ArraySpec(voxel_size=(1, 1)),
                emb_1: gp.ArraySpec(voxel_size=(1, 1))
            }) +
        # everything is 3D, except emb_0 and emb_1
        AddSpatialDim(emb_0) +
        AddSpatialDim(emb_1) +
        # now everything is 3D
        RemoveChannelDim(raw_0) +
        RemoveChannelDim(raw_1) +
        RemoveChannelDim(emb_0) +
        RemoveChannelDim(emb_1) +
        gp.Snapshot(
            output_filename='it{iteration}.hdf',
            dataset_names={
                raw_0: 'raw_0',
                raw_1: 'raw_1',
                points_0: 'points_0',
                points_1: 'points_1',
                emb_0: 'emb_0',
                emb_1: 'emb_1'
            },
            additional_request=snapshot_request,
            every=500) +
        gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):
        for i in range(num_iterations):
            batch = pipeline.request_batch(request)
