import os

import numpy as np
import torch

from data.utils import Data


class NeighborSampler:

    def __init__(
        self,
        adj_list: list,
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
        seed: int = None,
    ):
        """
        Neighbor sampler.

        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors,
                                                'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyperparameter that controls the sampling preference with time interval,
                                     a large time_scaling_factor tends to sample more on recent links,
                                    this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == "time_interval_aware":
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the
        # first position its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids,
        # self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures
            # the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == "time_interval_aware":
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors]))
                )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times

        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding
        # values in sampled_probabilities become nan (divided by zero), so we replace the nan by a very large negative
        # number -1e10 to denote the sampled probabilities

        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the
        overall interaction graph the returned interactions are sorted by time.

        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True)
                 with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                self.nodes_neighbor_sampled_probabilities[node_id][:i],
            )
        else:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                None,
            )

    def get_historical_neighbors(
        self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20
    ):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in
        node_interact_times

        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, "Number of sampled neighbors for each node should be greater than 0!"
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i]
        # with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i]
        # and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i]
        # and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened
        # before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = (
                self.find_neighbors_before(
                    node_id=node_id,
                    interact_time=node_interact_time,
                    return_sampled_probabilities=self.sample_neighbor_strategy == "time_interval_aware",
                )
            )

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling
                    # with node_neighbor_sampled_probabilities as None

                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors
                    # based on node_neighbor_sampled_probabilities

                    # for time_interval_aware sampling strategy, we additionally use softmax
                    # to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10,
                        # which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(
                            torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0
                        ).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(
                            a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities
                        )
                    else:
                        sampled_indices = self.random_state.choice(
                            a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities
                        )

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order,
                    # note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding,
                    # and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == "recent":
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids) :] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids) :] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times) :] = node_neighbor_times
                else:
                    raise ValueError(
                        f"Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!"
                    )

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(
        self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20
    ):
        """
        get historical neighbors of nodes in node_ids within num_hops hops

        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, "Number of sampled hops should be greater than 0!"

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors
        )
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
                node_ids=nodes_neighbor_ids_list[-1].flatten(),
                node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                num_neighbors=num_neighbors,
            )
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays,
        # each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number
        of neighbors (make the computation feasible)

        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information,
        # with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(
                node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=False
            )
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(
    data: Data, sample_neighbor_strategy: str = "uniform", time_scaling_factor: float = 0.0, seed: int = None
):
    """
    get neighbor sampler

    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors,
                                          'uniform', 'recent', or 'time_interval_aware'
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
                                    a large time_scaling_factor tends to sample more on recent links,
                                    this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    if (
        os.environ.get("TGL_CANONICAL_RECENT_NEIGHBORS", "0") == "1"
        and sample_neighbor_strategy == "recent"
    ):
        from train.prospective_sampler import (
            CanonicalRecentNeighborSampler,
            build_canonical_adjacency,
        )

        declared_nodes = int(os.environ.get("TGL_PROSPECTIVE_NODE_COUNT", "128"))
        adjacency = build_canonical_adjacency(
            data.src_node_ids,
            data.dst_node_ids,
            data.edge_ids,
            data.node_interact_times,
            event_priorities=data.edge_ids,
            num_nodes=declared_nodes,
            undirected=True,
        )
        return CanonicalRecentNeighborSampler(adjacency)

    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
        data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times
    ):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(
        adj_list=adj_list,
        sample_neighbor_strategy=sample_neighbor_strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
    )


class NegativeEdgeSampler(object):

    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        interact_times: np.ndarray = None,
        last_observed_time: float = None,
        negative_sample_strategy: str = "random",
        seed: int = None,
        directed: bool = True,
        doubled: bool = False,
        extra_avoid_edges: set = None,
    ):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".

        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        :param directed: whether all edges should be as they are (directed) or from the lower to higher node id
        :param doubled: whether all edges should be included in both directions
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time
        self.doubled = doubled
        self.extra_avoid_edges = set() if extra_avoid_edges is None else set(extra_avoid_edges)

        if directed and not doubled:
            self.possible_edges = set(
                (src_node_id, dst_node_id)
                for src_node_id in self.unique_src_node_ids
                for dst_node_id in self.unique_dst_node_ids
            )
        else:
            self.possible_edges = set(
                (src_node_id, dst_node_id)
                for src_node_id in self.unique_src_node_ids
                for dst_node_id in self.unique_dst_node_ids
                if src_node_id < dst_node_id
            )

        if self.negative_sample_strategy == "inductive":
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(
                self.earliest_time, self.last_observed_time
            )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time

        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set(
            (src_node_id, dst_node_id)
            for src_node_id, dst_node_id in zip(
                self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]
            )
        )

    def sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray = None,
        batch_dst_node_ids: np.ndarray = None,
        current_batch_start_time: float = 0.0,
        current_batch_end_time: float = 0.0,
        batch_times: np.ndarray = None,
    ):
        """
        sample negative edges, support random, historical and inductive sampling strategy

        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.doubled:
            # Half of the negative edges will come from reversing this sample
            # First, handle when size is odd (will drop a row later)
            if size % 2 != 0:
                odd_flag = True
                size += 1
            else:
                odd_flag = False

            size = int(size / 2)

        if self.negative_sample_strategy == "random":
            negative_src_node_ids, negative_dst_node_ids = self.random_sample_with_collision_check(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )
        elif self.negative_sample_strategy == "historical":
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )
        elif self.negative_sample_strategy == "inductive":
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )
        elif self.negative_sample_strategy == "explicit":
            negative_src_node_ids, negative_dst_node_ids = self.explicit_sample(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
                batch_times=batch_times,
            )
        else:
            raise ValueError(f"Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!")

        if self.doubled:
            new_src = []
            new_dst = []

            for n1, n2 in zip(negative_src_node_ids, negative_dst_node_ids):
                new_src.extend([n1, n2])
                new_dst.extend([n2, n1])

            negative_src_node_ids = np.array(new_src)
            negative_dst_node_ids = np.array(new_dst)

            if odd_flag:
                negative_src_node_ids = negative_src_node_ids[:-1]
                negative_dst_node_ids = negative_dst_node_ids[:-1]

        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works

        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return (
            self.unique_src_node_ids[random_sample_edge_src_node_indices],
            self.unique_dst_node_ids[random_sample_edge_dst_node_indices],
        )

    def random_sample_with_collision_check(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear
        in the current batch, used for historical and inductive sampling strategy

        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        avoid_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        avoid_edges |= self.extra_avoid_edges
        possible_random_edges = list(self.possible_edges - avoid_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times,
        # otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(
            len(possible_random_edges), size=size, replace=len(possible_random_edges) < size
        )
        return np.array(
            [possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]
        ), np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges

        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time, end_time=current_batch_start_time
        )
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges
        # with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(
                len(unique_historical_edges), size=size, replace=False
            )
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges
        and the current batch, if number of inductive edges is smaller than size, then fill in remaining edges
        with randomly sampled edges

        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time, end_time=current_batch_start_time
        )
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2)
        # not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges
        # with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(
                len(unique_inductive_edges), size=size, replace=False
            )
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)
    
    def explicit_sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
        batch_times: np.ndarray,
    ):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear
        in the current batch, and that non-appearing reverses of positive edges appear as negatives

        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """

        init_negative_src_node_ids, init_negative_dst_node_ids = self.random_sample_with_collision_check(
                size=size*2,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )
        
        time_step_edges = {}

        for i in range(int(current_batch_start_time), int(current_batch_end_time)+1):
            time_step_edges[i] = set(self.get_unique_edges_between_start_end_time(
                start_time=i, end_time=i
                ))
        
        for i, ts in enumerate(batch_times):
            if (batch_dst_node_ids[i], batch_src_node_ids[i]) not in time_step_edges[int(ts)]:
                init_negative_src_node_ids[i] = batch_dst_node_ids[i]
                init_negative_dst_node_ids[i] = batch_src_node_ids[i]

        return init_negative_src_node_ids, init_negative_dst_node_ids

        

    def reset_random_state(self):
        """
        reset the random state by self.seed

        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_negative_edge_samplers(
    full_data,
    train_data,
    val_data,
    doubled,
    directed,
    ns_train,
    train_seed,
    avoid_future_positives=False,
):

    prospective_schedule = os.environ.get("TGL_PROSPECTIVE_NEGATIVE_SCHEDULE")
    if prospective_schedule:
        if ns_train != "random" or not directed or doubled or avoid_future_positives:
            raise ValueError(
                "The prospective fixed schedule requires directed random training, "
                "no doubled edges, and no future-positive exclusion"
            )
        from train.prospective_sampler import FrozenNegativeScheduleSampler

        train_neg_edge_sampler = FrozenNegativeScheduleSampler(prospective_schedule)
        val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=0,
            interact_times=full_data.node_interact_times,
            negative_sample_strategy="random",
            directed=True,
            doubled=False,
        )
        test_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=2,
            interact_times=full_data.node_interact_times,
            negative_sample_strategy="random",
            directed=True,
            doubled=False,
        )
        return train_neg_edge_sampler, val_neg_edge_sampler, test_neg_edge_sampler

    future_avoid_edges = set()
    if avoid_future_positives:
        if directed and not doubled:
            future_avoid_edges = set(zip(full_data.src_node_ids, full_data.dst_node_ids))
        else:
            future_avoid_edges = {
                (min(src, dst), max(src, dst))
                for src, dst in zip(full_data.src_node_ids, full_data.dst_node_ids)
                if src != dst
            }

    if ns_train == "random":
        train_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            negative_sample_strategy=ns_train,
            seed=train_seed,
            directed=directed,
            doubled=doubled,
            extra_avoid_edges=future_avoid_edges,
        )
        val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=0,
            interact_times=full_data.node_interact_times,
            negative_sample_strategy=ns_train,
            directed=directed,
            doubled=doubled,
        )
        test_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=2,
            interact_times=full_data.node_interact_times,
            negative_sample_strategy=ns_train,
            directed=directed,
            doubled=doubled,
        )
    else:
        train_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            last_observed_time=train_data.node_interact_times[0],
            negative_sample_strategy=ns_train,
            seed=train_seed,
            directed=directed,
            doubled=doubled,
            extra_avoid_edges=future_avoid_edges,
        )
        val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            interact_times=full_data.node_interact_times,
            last_observed_time=train_data.node_interact_times[-1],
            seed=0,
            negative_sample_strategy=ns_train,
            directed=directed,
            doubled=doubled,
        )

        if ns_train == "explicit":
            test_neg_edge_sampler = NegativeEdgeSampler(
                src_node_ids=full_data.src_node_ids,
                dst_node_ids=full_data.dst_node_ids,
                seed=2,
                interact_times=full_data.node_interact_times,
                last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy="random",
                directed=directed,
                doubled=doubled,
            )

        else:
            test_neg_edge_sampler = NegativeEdgeSampler(
                src_node_ids=full_data.src_node_ids,
                dst_node_ids=full_data.dst_node_ids,
                seed=2,
                interact_times=full_data.node_interact_times,
                last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy=ns_train,
                directed=directed,
                doubled=doubled,
            )

    return train_neg_edge_sampler, val_neg_edge_sampler, test_neg_edge_sampler
