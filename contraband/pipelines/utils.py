import gunpowder as gp
import numpy as np
from collections.abc import Iterable
import skimage.filters as filters


class SetDtype(gp.BatchFilter):

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        array = batch.arrays[self.array]
        # array.data = array.data.astype(self.dtype)
        array.spec.dtype = self.dtype


class Blur(gp.BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function
    skimage.filters.gaussian
    See scikit-image documentation for more information.

    Args:

        array (:class:`ArrayKey`):

            The array to blur.

        sigma (``scalar or list``):

            The st. dev to use for the gaussian filter. If scalar it will be 
            projected to match the number of ROI dims. If give an list or numpy
            array, it must match the number of ROI dims.

    '''

    def __init__(self, array, sigma=1):
        self.array = array
        self.sigma = sigma
        self.filter_radius = np.ceil(np.array(self.sigma) * 3)

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        spec = request[self.array].copy()
        
        if isinstance(self.sigma, Iterable):
            assert spec.roi.dims() == len(self.sigma), \
                   ("Dimensions given for sigma (" 
                   + str(len(self.sigma)) + ") is not equal to the ROI dims (" 
                   + str(spec.roi.dims()) + ")")
        else:
            self.filter_radius = [self.filter_radius 
                                  for dim in range(spec.roi.dims())]

        self.grow_amount = gp.Coordinate([radius
                                          for radius in self.filter_radius])

        grown_roi = spec.roi.grow(
            self.grow_amount,
            self.grow_amount)
        grown_roi.snap_to_grid(self.spec[self.array].voxel_size)

        spec.roi = grown_roi
        deps[self.array] = spec
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]
        roi = raw.spec.roi
        
        if not isinstance(self.sigma, Iterable):
            self.sigma = [0 for dim in range(len(raw.data.shape) - roi.dims())] \
                + [self.sigma for dim in range(roi.dims())]

        raw.data = filters.gaussian(raw.data, sigma=self.sigma,
                                    mode='constant', preserve_range=True,
                                    multichannel=False)
        
        batch[self.array].crop(request[self.array].roi)


class InspectBatch(gp.BatchFilter):

    def __init__(self, prefix):
        self.prefix = prefix

    def process(self, batch, request):
        for key, array in batch.arrays.items():
            print(f"{self.prefix} ======== {key}: {array.data.shape}")
        for key, graph in batch.graphs.items():
            print(f"{self.prefix} ======== {key}: {graph}")


class RemoveChannelDim(gp.BatchFilter):

    def __init__(self, array, axis=0):
        self.array = array
        self.axis = axis

    def process(self, batch, request):

        if self.array not in batch:
            return
        data = batch[self.array].data
        shape = data.shape
        roi = batch[self.array].spec.roi

        assert self.axis < len(shape) - roi.dims(), "Axis given not is in ROI and not channel dim, " \
                "Shape:" + str(shape) + " ROI: " + str(roi)
        assert shape[self.axis] == 1, "Channel to delete must be size 1," \
                                       "but given shape " + str(shape)
        shape = self.__remove_dim(shape, self.axis) 
        batch[self.array].data = data.reshape(shape)

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

class RemoveSpatialDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def setup(self):

        upstream_spec = self.get_upstream_provider().spec[self.array]

        spec = upstream_spec.copy()
        if spec.roi is not None:
            spec.roi = gp.Roi(
                self.__remove_dim(spec.roi.get_begin()),
                self.__remove_dim(spec.roi.get_shape()))
        if spec.voxel_size is not None:
            spec.voxel_size = self.__remove_dim(spec.voxel_size)
        self.spec[self.array] = spec
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):

        if self.array not in request:
            return

        upstream_spec = self.get_upstream_provider().spec[self.array]

        request[self.array].roi = gp.Roi(
            self.__insert_dim(request[self.array].roi.get_begin(), 0),
            self.__insert_dim(request[self.array].roi.get_shape(), 1))
        if request[self.array].voxel_size is not None:
            request[self.array].voxel_size = self.__insert_dim(
                request[self.array].voxel_size, 1)

    def process(self, batch, request):
        if self.array not in batch:
            return
        data = batch[self.array].data
        shape = data.shape
        roi = batch[self.array].spec.roi
        assert shape[-roi.dims()] == 1, "Channel to delete must be size 1," \
                                       "but given shape " + str(shape)

        shape = self.__remove_dim(shape, len(shape) - roi.dims()) 
        batch[self.array].data = data.reshape(shape)
        batch[self.array].spec.roi = gp.Roi(
                self.__remove_dim(roi.get_begin()),
                self.__remove_dim(roi.get_shape()))
        batch[self.array].spec.voxel_size = \
            self.__remove_dim(batch[self.array].spec.voxel_size)

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

    def __insert_dim(self, a, s, dim=0):
        return a[:dim] + (s,) + a[dim:]

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
        self.updates(self.array, self.spec[self.array])

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


class AddChannelDim(gp.BatchFilter):

    def __init__(self, array, axis=0):
        self.array = array
        self.axis = axis

    def process(self, batch, request):

        if self.array not in batch:
            return
        batch[self.array].data = np.expand_dims(batch[self.array].data, self.axis)
