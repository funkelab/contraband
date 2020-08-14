import gunpowder as gp
import numpy as np
from collections.abc import Iterable
import skimage.filters as filters
from gunpowder import BatchProvider
from gunpowder.profiling import Timing
from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource
import copy

import logging
logger = logging.getLogger(__name__)

class SetDtype(gp.BatchFilter):

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.array].copy()
        spec.dtype = self.dtype
        self.updates(self.array, spec) 

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        array = batch.arrays[self.array]
        array.data = array.data.astype(self.dtype)
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
        self.sigma = sigma.copy()
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
            self.filter_radius = [self.filter_radius * spec.voxel_size[dim]
                                  for dim in range(spec.roi.dims())]

        self.grow_amount = gp.Coordinate([radius
                                          for radius in self.filter_radius])

        grown_roi = spec.roi.grow(
            self.grow_amount,
            self.grow_amount)
        grown_roi = grown_roi.snap_to_grid(self.spec[self.array].voxel_size)

        spec.roi = grown_roi
        deps[self.array] = spec
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]
        roi = raw.spec.roi
        
        if not isinstance(self.sigma, Iterable):
            sigma = [0 for dim in range(len(raw.data.shape) - roi.dims())] \
                + [self.sigma for dim in range(roi.dims())]
        else: 
            sigma = [0 for dim in range(len(raw.data.shape) - roi.dims())] \
                + self.sigma 


        raw.data = filters.gaussian(raw.data, sigma=sigma,
                                    mode='constant', preserve_range=True,
                                    multichannel=False)
        
        batch[self.array].crop(request[self.array].roi)


class InspectBatch(gp.BatchFilter):

    def __init__(self, prefix):
        self.prefix = prefix
    
    def prepare(self, request):
        for key, v in request.items():
            print(f"{self.prefix} ======== {key} ROI: {self.spec[key].roi}")

    def process(self, batch, request):
        for key, array in batch.arrays.items():
            print(f"{self.prefix} ======== {key}: {array.data.shape} ROI: {array.spec.roi}")
        for key, graph in batch.graphs.items():
            print(f"{self.prefix} ======== {key}: {graph.spec.roi} {graph}")


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

    def __init__(self, key):
        self.key = key

    def setup(self):

        upstream_spec = self.get_upstream_provider().spec[self.key]

        spec = upstream_spec.copy()
        if spec.roi is not None:
            spec.roi = gp.Roi(
                self.__remove_dim(spec.roi.get_begin()),
                self.__remove_dim(spec.roi.get_shape()))

        if isinstance(self.key, gp.ArrayKey):
            if spec.voxel_size is not None:
                spec.voxel_size = self.__remove_dim(spec.voxel_size)

        self.spec[self.key] = spec
        self.updates(self.key, self.spec[self.key])

    def prepare(self, request):

        if self.key not in request:
            return

        upstream_spec = self.get_upstream_provider().spec[self.key]

        request[self.key].roi = gp.Roi(
            self.__insert_dim(request[self.key].roi.get_begin(), 0),
            self.__insert_dim(request[self.key].roi.get_shape(), 1))

        if isinstance(self.key, gp.ArrayKey):
            if request[self.key].voxel_size is not None:
                request[self.key].voxel_size = self.__insert_dim(
                    request[self.key].voxel_size, 1)

    def process(self, batch, request):
        if self.key not in batch:
            return
        
        if isinstance(self.key, gp.ArrayKey):
            data = batch[self.key].data
            shape = data.shape
            roi = batch[self.key].spec.roi
            assert shape[-roi.dims()] == 1, "Channel to delete must be size 1," \
                                           "but given shape " + str(shape)

            shape = self.__remove_dim(shape, len(shape) - roi.dims()) 
            batch[self.key].data = data.reshape(shape)
            batch[self.key].spec.roi = gp.Roi(
                    self.__remove_dim(roi.get_begin()),
                    self.__remove_dim(roi.get_shape()))
            batch[self.key].spec.voxel_size = \
                self.__remove_dim(batch[self.key].spec.voxel_size)

        if isinstance(self.key, gp.GraphKey):
            roi = batch[self.key].spec.roi

            batch[self.key].spec.roi = gp.Roi(
                self.__remove_dim(roi.get_begin()),
                self.__remove_dim(roi.get_shape()))
            
            graph = gp.Graph([], [], spec=batch[self.key].spec)
            for node in batch[self.key].nodes:
                print(node)
                new_node = gp.Node(node.id, 
                                   node.location[1:],
                                   temporary=node.temporary,
                                   attrs=node.attrs)
                graph.add_node(new_node)
                print(node)
            print(graph.spec.roi)
            print(list(graph.nodes))
            batch[self.key] = graph

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

    def __insert_dim(self, a, s, dim=0):
        return a[:dim] + (s,) + a[dim:]


class RandomPointGenerator:

    def __init__(self, density=None, repetitions=1, num_points=None):
        '''Create random points in a provided ROI with the given density.

        Args:

            density (float):

                The expected number of points per world unit cube. If, for
                example, the ROI passed to `get_random_points(roi)` has a 2D
                size of (10, 10) and the density is 1.0, 100 uniformly
                distributed points will be returned.

            repetitions (int):

                Return the same list of points that many times. Note that in
                general only the first call will contain uniformly distributed
                points for the given ROI. Subsequent calls with a potentially
                different ROI will only contain the points that lie within that
                ROI.
        '''
        self.density = density
        self.repetitions = repetitions
        self.iteration = 0
        self.num_points = num_points

    def get_random_points(self, roi):
        '''Get a dictionary mapping point IDs to nD locations, uniformly
        distributed in the given ROI. If `repetitions` is larger than 1,
        previously sampled points will be reused that many times.
        '''

        ndims = roi.dims()
        volume = np.prod(roi.get_shape())

        if self.iteration % self.repetitions == 0:

            # create random points in the unit cube
            if self.num_points is None:
                self.points = np.random.random((int(self.density * volume), ndims))
            else:
                self.points = np.random.random((self.num_points, ndims))
            # scale and shift into requested ROI
            self.points *= np.array(roi.get_end() - roi.get_begin())
            self.points += roi.get_begin()

            ret = {i: point for i, point in enumerate(self.points)}

        else:

            ret = {}
            for i, point in enumerate(self.points):
                if roi.contains(point):
                    ret[i] = point

        self.iteration += 1
        return ret


class RandomPointSource(gp.BatchProvider):

    def __init__(
            self,
            graph_key,
            density=None,
            random_point_generator=None,
            shrink_by=None):
        '''A source creating uniformly distributed points.

        Args:

            graph_key (:class:`GraphKey`):

                The graph key to provide.

            density (float, optional):

                The expected number of points per world unit cube. If, for
                example, the ROI passed to `get_random_points(roi)` has a 2D
                size of (10, 10) and the density is 1.0, 100 uniformly
                distributed points will be returned.

                Only used if `random_point_generator` is `None`.

            random_point_generator (:class:`RandomPointGenerator`, optional):

                The random point generator to use to create points.

                One of `density` or `random_point_generator` has to be given.
        '''

        assert (density is not None) != (random_point_generator is not None), \
            "Exactly one of 'density' or 'random_point_generator' has to be " \
            "given"

        self.graph_key = graph_key
        if density is not None:
            self.random_point_generator = RandomPointGenerator(density=density)
        else:
            self.random_point_generator = random_point_generator
        self.shrink_by = shrink_by

    def setup(self):

        # provide points in an infinite ROI
        self.graph_spec = gp.GraphSpec(
            roi=gp.Roi(
                offset=(0, 0, 0),
                shape=(None, None, None)))

        self.provides(self.graph_key, self.graph_spec)

    def provide(self, request):

        roi = request[self.graph_key].roi
        if self.shrink_by is not None:
            roi = roi / self.shrink_by 

        random_points = self.random_point_generator.get_random_points(roi)

        batch = gp.Batch()
        if self.shrink_by is None:
            batch[self.graph_key] = gp.Graph(
                [gp.Node(id=i, location=l) for i, l in random_points.items()],
                [],
                gp.GraphSpec(roi=roi))
        else:
            print(list(random_points.items()))
            print(list(random_points.values())[0]* np.array(self.shrink_by))
            # Put graph back into original roi
            batch[self.graph_key] = gp.Graph(
                [gp.Node(id=i, location=l)
                 for i, l in random_points.items()],
                [],
                gp.GraphSpec(roi=roi))

        return batch


class PrepareBatch(gp.BatchFilter):

    def __init__(
            self,
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1,
            is_2d):
        self.raw_0 = raw_0
        self.raw_1 = raw_1
        self.points_0 = points_0
        self.points_1 = points_1
        self.locations_0 = locations_0
        self.locations_1 = locations_1
        self.is_2d = is_2d

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
            locations_0.append(location_0)
            locations_1.append(location_1)
        
        locations_0 = np.array(locations_0, dtype=np.float32)
        locations_1 = np.array(locations_1, dtype=np.float32)
        if self.is_2d:
            locations_0 = locations_0[:, 1:]
            locations_1 = locations_1[:, 1:]
        locations_0 = locations_0[np.newaxis]
        locations_1 = locations_1[np.newaxis]

        # create point location arrays (with batch dimension)
        batch[self.locations_0] = gp.Array(
            locations_0, self.spec[self.locations_0])
        batch[self.locations_1] = gp.Array(
            locations_1, self.spec[self.locations_1])

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

class RejectArray(gp.BatchFilter):

    def __init__(self, ensure_nonempty):
        self.ensure_nonempty = ensure_nonempty

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 2
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)

            if batch.arrays[self.ensure_nonempty].data.size != 0:
                have_good_batch = True
                logger.debug(f"Accepted batch with shape: {batch.arrays[self.ensure_nonempty].data.shape}")
            else:
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:
                    logger.info(
                        f"rejected {report_next_timeout} batches, been waiting for a good one "
                        "since {report_next_timeout}")
                    report_next_timeout *= 2

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

class RandomSourceGenerator:

    def __init__(self, num_sources, probabilities=None, repetitions=1):
        '''Create random points in a provided ROI with the given density.

        Args:

            repetitions (int):

                How many times the generator will be used in a pipeline.
                Only the first request to the RandomSource will have a 
                random source chosen. Future calls will use the same source.
        '''
        self.repetitions = repetitions
        self.num_sources = num_sources
        self.probabilities = probabilities

        # automatically normalize probabilities to sum to 1
        if self.probabilities is not None:
            self.probabilities = [float(x) / np.sum(probabilities) 
                                  for x in self.probabilities]

        if self.probabilities is not None:
            assert self.num_sources == len(
                self.probabilities), "if probabilities are specified, they " \
                                     "need to be given for each batch " \
                                     "provider added to the RandomProvider"
        self.iteration = 0

    def get_random_source(self):
        '''Get a randomly chosen source. If `repetitions` is larger than 1,
        the previously chosen source will be given. 
        '''
        if self.iteration % self.repetitions == 0:
            
            self.choice = np.random.choice(list(range(self.num_sources)),
                                           p=self.probabilities)
        self.iteration += 1
        return self.choice


class RandomMultiBranchSource(BatchProvider):
    '''Randomly selects one of the upstream providers based on a RandomSourceGenerator::
        (a + b + c) + RandomProvider()
    will create a provider that randomly relays requests to providers ``a``,
    ``b``, or ``c``. Array and point keys of ``a``, ``b``, and ``c`` should be
    the same. The RandomSourceGenerator will ensure that multiple branches of
    inputs will choose the same source.
    Args:
        The random source generator to sync random choice.  
    '''

    def __init__(self, random_source_generator):
        self.random_source_generator = random_source_generator

    def setup(self):

        assert len(self.get_upstream_providers()) > 0,\
            "at least one batch provider must be added to the RandomProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):
        source_idx = self.random_source_generator.get_random_source()
        source = self.get_upstream_providers()[source_idx]
        if isinstance(source, Hdf5LikeSource):
            logger.debug(f"Dataset chosen: {source.datasets}, {self.random_source_generator.iteration}")
        return source.request_batch(request)

class FillLocations(gp.BatchFilter):

    def __init__(
            self,
            raw,
            points,
            locations,
            is_2d,
            max_points=None):
        self.raw = raw
        self.points = points
        self.locations = locations
        self.is_2d = is_2d
        self.max_points = max_points

    def setup(self):
        self.provides(
            self.locations,
            gp.ArraySpec(nonspatial=True))

    def process(self, batch, request):

        locations = []
        # get list of only xy locations
        # locations are in voxels, relative to output roi
        points_roi = request[self.points].roi
        voxel_size = batch[self.raw].spec.voxel_size
        for i, node in enumerate(batch[self.points].nodes):
            if self.max_points is not None and i > self.max_points - 1:
                break

            location = node.location
            location -= points_roi.get_begin()
            location /= voxel_size
            locations.append(location)
        
        locations = np.array(locations, dtype=np.float32)
        print(locations)
        if self.is_2d:
            locations = locations[:, 1:]

        # create point location arrays 
        batch[self.locations] = gp.Array(
            locations, self.spec[self.locations])


class PointsLabelsSource(BatchProvider):
    '''Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point.

    Args:

        points (:class: `ArrayKey` or `GraphKey`):

            The key of the points correseponding to the gunpowder array
            or graph key. If points is an array key it will directly load the given 
            number of points into the array. If points is a graph key it will
            provide points in a ROI, in similar functinailty to CSVPointsSource.

        data (:class:`numpy array`):
            
            The data correseponding to the points. If points is an ArrayKey,
            data should be the actual data values (not the point locations). If
            points is a graphkey the data should be the point locations.

        labels (:class:`ArrayKey`, optional):
            
            The gunpowder ArrayKey for the labels for each point.

        label_data (:class: `Numpy Array`, optional):

            The actual label for each point, will be loaded into the labels key.

        num_points (:class: `int`, default=1):

            The number of points to return. If given an array key this will 
            specify the number of points that will be randomly selected to be 
            put into the points ArrayKey. If points is a GraphKey it does not 
            affect the number of points. 

        points_spec (:class:`GraphSpec` or `ArraySpec`, optional):

            An optional :class:`GraphSpec` or :class:`ArraySpec` to overwrite the points specs
            automatically determined from the points data. This is useful to set
            the :class:`Roi` manually.

        labels_spec (`ArraySpec`, optional):

            An optional :class:`ArraySpec` to overwrite the labels specs
            automatically given a voxel size of 1. This is useful to set
            the voxel_size manually.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the given points data.
            This is useful if the points refer to voxel positions to convert them to world units.
    '''

    def __init__(self,
                 points, 
                 data, 
                 labels=None, 
                 label_data=None, 
                 num_points=1, 
                 points_spec=None, 
                 labels_spec=None, 
                 scale=None):

        self.points = points
        self.labels = labels
        self.data = data
        self.label_data = label_data
        self.num_points = num_points
        self.points_spec = points_spec
        self.labels_spec = labels_spec
        self.scale = scale
        
        # Apply scale to given data
        if scale is not None:
            self.data = self.data * scale

    def setup(self):
        
        self.ndims = self.data.shape[1]

        if self.points_spec is not None:
            self.provides(self.points, self.points_spec)
        elif isinstance(self.points, gp.ArrayKey):
            self.provides(self.points, gp.ArraySpec(voxel_size=((1,))))
        elif isinstance(self.points, gp.GraphKey):
            print(self.ndims)
            min_bb = gp.Coordinate(np.floor(np.amin(self.data[:, :self.ndims], 0)))
            max_bb = gp.Coordinate(np.ceil(np.amax(self.data[:, :self.ndims], 0)) + 1)

            roi = gp.Roi(min_bb, max_bb - min_bb)
            logger.debug(f"Bounding Box: {roi}")

            self.provides(self.points, gp.GraphSpec(roi=roi))

        if self.labels is not None:
            assert isinstance(self.labels, gp.ArrayKey), \
                   f"Label key must be an ArrayKey, \
                     was given {type(self.labels)}"

            if self.labels_spec is not None:
                self.provides(self.labels, self.labels_spec)
            else:
                self.provides(self.labels, gp.ArraySpec(voxel_size=((1,))))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = gp.Batch()

        # If a Array is requested then we will randomly choose
        # the number of requested points
        if isinstance(self.points, gp.ArrayKey):
            points = np.random.choice(self.data.shape[0], self.num_points)
            data = self.data[points][np.newaxis]
            if self.scale is not None:
                data = data * self.scale
            if self.label_data is not None:
                labels = self.label_data[points]
            batch[self.points] = gp.Array(data, self.spec[self.points])

        else:
            # If a graph is request we must select points within the 
            # request ROI

            min_bb = request[self.points].roi.get_begin()
            max_bb = request[self.points].roi.get_end()

            logger.debug(
                "Points source got request for %s",
                request[self.points].roi)

            point_filter = np.ones((self.data.shape[0],), dtype=np.bool)
            for d in range(self.ndims):
                point_filter = np.logical_and(point_filter, self.data[:, d] >= min_bb[d])
                point_filter = np.logical_and(point_filter, self.data[:, d] < max_bb[d])

            points_data, labels = self._get_points(point_filter)
            logger.debug(f"Found {len(points_data)} points")
            points_spec = gp.GraphSpec(roi=request[self.points].roi.copy())
            batch.graphs[self.points] = gp.Graph(points_data, [], points_spec)
        
        # Labels will always be an Array
        if self.label_data is not None:
            batch[self.labels] = gp.Array(labels, self.spec[self.labels])

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):
        filtered = self.data[point_filter]

        if self.label_data is not None:
            filtered_labels = self.labels[point_filter]
        else:
            filtered_labels = None

        ids = np.arange(len(self.data))[point_filter]

        return (
            [gp.Node(id=i, location=p)
                for i, p in zip(ids, filtered)],
            filtered_labels
        )
