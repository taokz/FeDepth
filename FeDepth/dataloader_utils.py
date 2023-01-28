import numpy as np 
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
import tqdm
from torchvision.datasets import CIFAR10, CIFAR100
from utils import shuffle_sampler

class CifarDataset(CIFAR10):
    all_domains = ['cifar10']
    resorted_domains = {
        0: ['cifar10'],
    }
    num_classes = 10

    def __init__(self, domain='cifar10', train=True, transform=None, download=True):
        assert domain in self.all_domains, f"Invalid domain: {domain}"
        data_path = os.path.join('./data', domain)
        super().__init__(data_path, train=train, transform=transform, download=download)

class CifarDataset100(CIFAR100):
    all_domains = ['cifar100']
    resorted_domains = {
        0: ['cifar100'],
    }
    num_classes = 100

    def __init__(self, domain='cifar100', train=True, transform=None, download=True):
        assert domain in self.all_domains, f"Invalid domain: {domain}"
        data_path = os.path.join('./data', domain)
        super().__init__(data_path, train=train, transform=transform, download=download)

class Partitioner(object):
    """Class for partition a sequence into multiple shares (or users).
    Args:
        rng (np.random.RandomState): random state.
        partition_mode (str): 'dir' for Dirichlet distribution or 'uni' for uniform.
        max_n_sample_per_share (int): max number of samples per share.
        min_n_sample_per_share (int): min number of samples per share.
        max_n_sample (int): max number of samples
        verbose (bool): verbosity
    """
    def __init__(self, rng=None, partition_mode="dir",
                 max_n_sample_per_share=-1,
                 min_n_sample_per_share=2,
                 max_n_sample=-1,
                 verbose=True
                 ):
        assert max_n_sample_per_share < 0 or max_n_sample_per_share > min_n_sample_per_share, \
            f"max ({max_n_sample_per_share}) > min ({min_n_sample_per_share})"
        self.rng = rng if rng else np.random
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample
        self.verbose = verbose

    def __call__(self, n_sample, n_share, log=print):
        """Partition a sequence of `n_sample` into `n_share` shares.
        Returns:
            partition: A list of num of samples for each share.
        """
        assert n_share > 0, f"cannot split into {n_share} share"
        if self.verbose:
            log(f"  {n_sample} smp => {n_share} shards by {self.partition_mode} distr")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        if n_sample < self.min_n_sample_per_share * n_share:
            raise ValueError(f"Not enough samples. Require {self.min_n_sample_per_share} samples"
                             f" per share at least for {n_share} shares. But only {n_sample} is"
                             f" available totally.")
        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [1]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        # uniformly add residual to as many users as possible.
        for i in self.rng.choice(n_share, n_sample - np.sum(partition)):
            partition[i] += 1
            # partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition

class ClassWisePartitioner(Partitioner):
    """Partition a list of labels by class. Classes will be shuffled and assigned to users
    sequentially.
    Args:
        n_class_per_share (int): number of classes per share (user).
        rng (np.random.RandomState): random state.
        partition_mode (str): 'dir' for Dirichlet distribution or 'uni' for uniform.
        max_n_sample_per_share (int): max number of samples per share.
        min_n_sample_per_share (int): min number of samples per share.
        max_n_sample (int): max number of samples
        verbose (bool): verbosity
    """
    def __init__(self, n_class_per_share=2, **kwargs):
        super(ClassWisePartitioner, self).__init__(**kwargs)
        self.n_class_per_share = n_class_per_share
        self._aux_partitioner = Partitioner(**kwargs)

    def __call__(self, labels, n_user, log=print, user_ids_by_class=None,
                 return_user_ids_by_class=False, consistent_class=False):
        """Partition a list of labels into `n_user` shares.
        Returns:
            partition: A list of users, where each user include a list of sample indexes.
        """
        # reorganize labels by class
        idx_by_class = defaultdict(list)
        if len(labels) > 1e5:
            labels_iter = tqdm(labels, leave=False, desc='sort labels')
        else:
            labels_iter = labels
        for i, label in enumerate(labels_iter):
            idx_by_class[label].append(i)

        n_class = len(idx_by_class)
        assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                          f"{n_user} users when each user only " \
                                                          f"has {self.n_class_per_share} classes."

        # assign classes to each user.
        if user_ids_by_class is None:
            user_ids_by_class = defaultdict(list)
            label_sampler = shuffle_sampler(list(range(n_class)),
                                            self.rng if consistent_class else None)
            for s in range(n_user):
                s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
                for c in s_classes:
                    user_ids_by_class[c].append(s)

        # assign sample indexes to clients
        idx_by_user = [[] for _ in range(n_user)]
        if n_class > 100 or len(labels) > 1e5:
            idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
            log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
        else:
            idx_by_class_iter = idx_by_class
        for c in idx_by_class_iter:
            l = len(idx_by_class[c])
            log(f" class-{c} => {len(user_ids_by_class[c])} shares")
            l_by_user = self._aux_partitioner(l, len(user_ids_by_class[c]), log=log)
            base_idx = 0
            for i_user, tl in zip(user_ids_by_class[c], l_by_user):
                idx_by_user[i_user].extend(idx_by_class[c][base_idx:base_idx+tl])
                base_idx += tl
        if return_user_ids_by_class:
            return idx_by_user, user_ids_by_class
        else:
            return idx_by_user

def extract_labels(dataset: Dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    dl = DataLoader(dataset, batch_size=512, drop_last=False, num_workers=4, shuffle=False)
    labels = []
    dl_iter = tqdm(dl, leave=False, desc='load labels') if len(dl) > 100 else dl
    for _, targets in dl_iter:
        labels.extend(targets.cpu().numpy().tolist())
    return labels            