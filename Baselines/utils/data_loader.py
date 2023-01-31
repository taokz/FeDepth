import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from utils.data_utils import Partitioner, \
    CifarDataset, CifarDataset100, ClassWisePartitioner, extract_labels, EmnistDataset

def compose_transforms(trns, image_norm):
    if image_norm == '0.5':
        return transforms.Compose(trns + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif image_norm == 'torch':
        return transforms.Compose(trns + [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])
    elif image_norm == 'torch-resnet':
        return transforms.Compose(trns + [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    elif image_norm == 'none':
        return transforms.Compose(trns)
    elif image_norm == 'emnist':
        return transforms.Compose(trns + [transforms.Normalize(transforms.Normalize((0.1307,), (0.3081,)))])
    else:
        raise ValueError(f"Invalid image_norm: {image_norm}")


def get_central_data(name: str, domains: list, percent=1., image_norm='none',
                     disable_image_norm_error=False, enable_resize=False):
    if image_norm != 'none' and not disable_image_norm_error:
        raise RuntimeError(f"This is a hard warning. Use image_norm != none will make the PGD"
                           f" attack invalid since PGD will clip the image into [0,1] range. "
                           f"Think before you choose {image_norm} image_norm.")
    if percent != 1. and name.lower() != 'digits':
        raise RuntimeError(f"percent={percent} should not be used in get_central_data."
                           f" Pass it to make_fed_data instead.")
    
    if name.lower() == 'cifar10':
        if image_norm == 'default':
            image_norm = 'torch'
        for domain in domains:
            if domain not in CifarDataset.all_domains:
                raise ValueError(f"Invalid domain: {domain}")
        if enable_resize == False:
            trn_train = [transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
            trn_test = [transforms.ToTensor()]
        else:
            trn_train = [transforms.Resize([224,224]),
                        # transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
            trn_test = [transforms.Resize([224,224]),
                        transforms.ToTensor()]

        train_sets = [CifarDataset(domain, train=True,
                                   transform=compose_transforms(trn_train, image_norm))
                      for domain in domains]
        test_sets = [CifarDataset(domain, train=False,
                                  transform=compose_transforms(trn_test, image_norm))
                     for domain in domains]
    elif name.lower() == 'cifar100':
        if image_norm == 'default':
            image_norm = 'torch'
        for domain in domains:
            if domain not in CifarDataset100.all_domains:
                raise ValueError(f"Invalid domain: {domain}")
        if enable_resize == False:
            trn_train = [transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
            trn_test = [transforms.ToTensor()]
        else:
            trn_train = [transforms.Resize([224,224]),
                        # transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
            trn_test = [transforms.Resize([224,224]),
                        transforms.ToTensor()]

        train_sets = [CifarDataset100(domain, train=True,
                                   transform=compose_transforms(trn_train, image_norm))
                      for domain in domains]
        test_sets = [CifarDataset100(domain, train=False,
                                  transform=compose_transforms(trn_test, image_norm))
                     for domain in domains]
    elif name.lower() == 'emnist':
        if image_norm == 'default':
            image_norm = 'emnist'
        for domain in domains:
            if domain not in EmnistDataset.all_domains:
                raise ValueError(f"Invalid domain: {domain}")
        if enable_resize == False:
            trn_train = [transforms.Resize([32,32]),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]
            trn_test = [transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
                ]
        else:
            trn_train = [transforms.Resize([224,224]),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]
            trn_test = [transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
                ]

        train_sets = [EmnistDataset(domain, train=True,
                                   transform=compose_transforms(trn_train, image_norm))
                      for domain in domains]
        test_sets = [EmnistDataset(domain, train=False,
                                  transform=compose_transforms(trn_test, image_norm))
                     for domain in domains]
    else:
        raise NotImplementedError(f"name: {name}")
    return train_sets, test_sets


def make_fed_data(train_sets, test_sets, batch_size, domains, shuffle_eval=False,
                  n_user_per_domain=1, partition_seed=42, partition_mode='uni',
                  n_class_per_user=-1, val_ratio=0.2,
                  eq_domain_train_size=True, percent=1.,
                  num_workers=0, pin_memory=False, min_n_sample_per_share=128,
                  subset_with_logits=False,
                  test_batch_size=None, shuffle=True,
                  consistent_test_class=False, 
                  partition_method='iid', alpha=0.3):
    """Distribute multi-domain datasets (`train_sets`) into federated clients.

    Args:
        train_sets (list): A list of datasets for training.
        test_sets (list): A list of datasets for testing.
        partition_seed (int): Seed for partitioning data into clients.
        consistent_test_class (bool): Ensure the test classes are the same training for a client.
            Meanwhile, make test sets are uniformly splitted for clients.
    """
    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    SubsetClass = SubsetWithLogits if subset_with_logits else Subset
    clients = [f'{i}' for i in range(len(domains))]

    print(f" train size: {[len(s) for s in train_sets]}")
    print(f" test  size: {[len(s) for s in test_sets]}")

    train_len = [len(s) for s in train_sets]
    if eq_domain_train_size:
        train_len = [min(train_len)] * len(train_sets)
        # assert all([len(s) == train_len[0] for s in train_sets]), f"Should be equal length."

    if percent < 1:
        train_len = [int(tl * percent) for tl in train_len]

    print(f" trimmed train size: {[tl for tl in train_len]}")

    if n_user_per_domain > 1:  # split data into multiple users
        if partition_method == 'pat':
            if n_class_per_user > 0:  # split by class-wise non-iid
                split = ClassWisePartitioner(rng=np.random.RandomState(partition_seed),
                                            n_class_per_share=n_class_per_user,
                                            min_n_sample_per_share=min_n_sample_per_share,
                                            partition_mode=partition_mode,
                                            verbose=True)
                splitted_clients = []
                val_sets, sub_train_sets, user_ids_by_class = [], [], []
                for i_client, (dname, tr_set) in enumerate(zip(clients, train_sets)):
                    _tr_labels = extract_labels(tr_set)  # labels in the original order
                    _tr_labels = _tr_labels[:train_len[i_client]]  # trim
                    _idx_by_user, _user_ids_by_cls = split(_tr_labels, n_user_per_domain,
                                                        return_user_ids_by_class=True)
                    print(f" {dname} | train split size: {[len(idxs) for idxs in _idx_by_user]}")
                    _tr_labels = np.array(_tr_labels)
                    print(f"    | train classes: "
                        f"{[f'{np.unique(_tr_labels[idxs]).tolist()}' for idxs in _idx_by_user]}")

                    for i_user, idxs in zip(range(n_user_per_domain), _idx_by_user):
                        vl = int(val_ratio * len(idxs))

                        np.random.shuffle(idxs)
                        sub_train_sets.append(SubsetClass(tr_set, idxs[vl:]))

                        np.random.shuffle(idxs)
                        val_sets.append(Subset(tr_set, idxs[:vl]))

                        splitted_clients.append(f"{dname}-{i_user}")
                    user_ids_by_class.append(_user_ids_by_cls if consistent_test_class else None)

                if consistent_test_class:
                    # recreate partitioner to make sure consistent class distribution.
                    split = ClassWisePartitioner(rng=np.random.RandomState(partition_seed),
                                                n_class_per_share=n_class_per_user,
                                                min_n_sample_per_share=min_n_sample_per_share,
                                                partition_mode='uni',
                                                verbose=True)
                sub_test_sets = []
                for i_client, te_set in enumerate(test_sets):
                    _te_labels = extract_labels(te_set)
                    _idx_by_user = split(_te_labels, n_user_per_domain,
                                        user_ids_by_class=user_ids_by_class[i_client])
                    print(f"   test split size: {[len(idxs) for idxs in _idx_by_user]}")
                    _te_labels = np.array(_te_labels)
                    print(f"   test classes: "
                        f"{[f'{np.unique(_te_labels[idxs]).tolist()}' for idxs in _idx_by_user]}")

                    for idxs in _idx_by_user:
                        np.random.shuffle(idxs)
                        sub_test_sets.append(Subset(te_set, idxs))
        
        elif partition_method == 'dir':
            # reference: https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
            temp_trainset = train_sets[0]
            # temp_testset = test_sets[0]
            K = temp_trainset.num_classes
            # print('The value of K is ', K)

            temp_trainloader = torch.utils.data.DataLoader(
                temp_trainset, batch_size=len(temp_trainset.data), shuffle=False)
            # temp_testloader = torch.utils.data.DataLoader(
            #     temp_testset, batch_size=len(temp_testset.data), shuffle=False)

            for _, temp_train_data in enumerate(temp_trainloader, 0):
                temp_trainset.data, temp_trainset.targets = temp_train_data
            # for _, temp_test_data in enumerate(temp_testloader, 0):
            #     temp_testset.data, temp_testset.targets = temp_test_data

            dataset_image = []
            dataset_label = []

            dataset_image.extend(temp_trainset.data.cpu().detach().numpy())
            # dataset_image.extend(temp_testset.data.cpu().detach().numpy())
            dataset_label.extend(temp_trainset.targets.cpu().detach().numpy())
            # dataset_label.extend(temp_testset.targets.cpu().detach().numpy())
            dataset_image = np.array(dataset_image)
            dataset_label = np.array(dataset_label)   
            print("Done (image, label) extraction.")

            splitted_clients = []
            val_sets, sub_train_sets = [], []
            # X = [[] for _ in range(n_user_per_domain)]
            # y = [[] for _ in range(n_user_per_domain)]
            statistic = [[] for _ in range(n_user_per_domain)]
            dataidx_map = {}
            
            min_size = 0
            N = len(dataset_label)
            least_samples = 50
            # least_samples = batch_size / val_ratio # least samples for each client
            # if least_samples > (train_len[0]/n_user_per_domain):
            #     least_samples = (train_len[0]/n_user_per_domain)/5
            print("least samples for each client: ", least_samples)

            # rng = np.random.RandomState(partition_seed)

            while min_size < least_samples:
                idx_batch = [[] for _ in range(n_user_per_domain)]
                for k in range(K):
                    idx_k = np.where(dataset_label == k)[0]
                    # rng.shuffle(idx_k)
                    np.random.shuffle(idx_k)
                    # proportions = rng.dirichlet(np.repeat(alpha, n_user_per_domain))
                    proportions = np.random.dirichlet(np.repeat(alpha, n_user_per_domain))
                    proportions = np.array([p*(len(idx_j)<N/n_user_per_domain) for p,idx_j in zip(proportions,idx_batch)])
                    proportions = proportions/proportions.sum()
                    proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    # print(f'minimal size w.r.t label', k, ' is ', min_size)
                    # print(f'Finish dirichlet sampling w.r.t label', k)

            for j in range(n_user_per_domain):
                np.random.shuffle(idx_batch[j])
                dataidx_map[j] = idx_batch[j]

            # assign data
            for client in range(n_user_per_domain):
                idxs = dataidx_map[client]
                # X[client] = dataset_image[idxs]
                # y[client] = dataset_label[idxs]
                X = torch.Tensor(dataset_image[idxs])
                y = torch.from_numpy(dataset_label[idxs])
                tr_set = torch.utils.data.TensorDataset(X, y)

                tl = len(dataset_label[idxs])
                vl = int(val_ratio * tl)
                tl = tl - vl
                sub_train_sets.append(SubsetClass(tr_set, list(range(0, tl))))
                val_sets.append(Subset(tr_set, list(range(tl, tl+vl))))
                splitted_clients.append(f"User-0-{client}")

                for i in np.unique(dataset_label[idxs]):
                    statistic[client].append((int(i), int(sum(dataset_label[idxs]==i))))

                # print('Finish assigning data to client ', client)

            for client in range(n_user_per_domain):
                #print(f"Client {client}\t Size of training/valid data: {len(dataset_image[idxs])}\t Labels: ", np.unique(dataset_label[idxs]))
                print(f"Client {client}\t Samples of labels: ", [i for i in statistic[client]])
                print("-" * 50)

            # uniformly distribute test sets
            split = Partitioner(rng=np.random.RandomState(partition_seed),
                                min_n_sample_per_share=min_n_sample_per_share,
                                partition_mode=partition_mode)
            if consistent_test_class:
                split = Partitioner(rng=np.random.RandomState(partition_seed),
                                    min_n_sample_per_share=min_n_sample_per_share,
                                    partition_mode='uni')
            sub_test_sets = []
            for te_set in test_sets:
                _test_len_by_user = split(len(te_set), n_user_per_domain)

                base_idx = 0
                for tl in _test_len_by_user:
                    sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

            # rename
            train_sets = sub_train_sets
            test_sets = sub_test_sets
            clients = splitted_clients

        elif partition_method == 'balanced-dir':
            # reference: https://github.com/snow12345/FedAGM/blob/main/datasets/cifar.py
            temp_trainset = train_sets[0]
            K = temp_trainset.num_classes
            N = len(temp_trainset)
            
            temp_trainloader = torch.utils.data.DataLoader(
                temp_trainset, batch_size=len(temp_trainset.data), shuffle=True)

            for _, temp_train_data in enumerate(temp_trainloader, 0):
                temp_trainset.data, temp_trainset.targets = temp_train_data

            dataset_image = []
            dataset_label = []

            dataset_image.extend(temp_trainset.data.cpu().detach().numpy())
            dataset_label.extend(temp_trainset.targets.cpu().detach().numpy())
            dataset_image = np.array(dataset_image)
            dataset_label = np.array(dataset_label)   
            print("Done (image, label) extraction.")            
            
            net_dataidx_map = {i: np.array([],dtype='int64') for i in range(n_user_per_domain)}
            assigned_ids = []
            idx_batch = [[] for _ in range(n_user_per_domain)]
            num_data_per_client=int(N/n_user_per_domain)

            for i in range(n_user_per_domain):
                weights = torch.zeros(N)
                proportions = np.random.dirichlet(np.repeat(alpha, K))
                for k in range(K):
                    idx_k = np.where(dataset_label == k)[0]
                    weights[idx_k]=proportions[k]
                weights[assigned_ids] = 0.0
                idx_batch[i] = (torch.multinomial(weights, num_data_per_client, replacement=False)).tolist()
                assigned_ids+=idx_batch[i]

            dataidx_map = {}
            for j in range(n_user_per_domain):
                np.random.shuffle(idx_batch[j])
                dataidx_map[j] = idx_batch[j]
            
            
            splitted_clients = []
            val_sets, sub_train_sets = [], []
            statistic = [[] for _ in range(n_user_per_domain)]

            # assign data
            for client in range(n_user_per_domain):
                idxs = dataidx_map[client]
                X = torch.Tensor(dataset_image[idxs])
                y = torch.from_numpy(dataset_label[idxs])
                tr_set = torch.utils.data.TensorDataset(X, y)

                tl = len(dataset_label[idxs])
                vl = int(val_ratio * tl)
                tl = tl - vl
                sub_train_sets.append(SubsetClass(tr_set, list(range(0, tl))))
                val_sets.append(Subset(tr_set, list(range(tl, tl+vl))))
                splitted_clients.append(f"User-0-{client}")

                for i in np.unique(dataset_label[idxs]):
                    statistic[client].append((int(i), int(sum(dataset_label[idxs]==i))))

                # print('Finish assigning data to client ', client)

            for client in range(n_user_per_domain):
                #print(f"Client {client}\t Size of training/valid data: {len(dataset_image[idxs])}\t Labels: ", np.unique(dataset_label[idxs]))
                print(f"Client {client}\t Samples of labels: ", [i for i in statistic[client]])
                print("-" * 50)

            # uniformly distribute test sets
            split = Partitioner(rng=np.random.RandomState(partition_seed),
                                min_n_sample_per_share=min_n_sample_per_share,
                                partition_mode=partition_mode)
            if consistent_test_class:
                split = Partitioner(rng=np.random.RandomState(partition_seed),
                                    min_n_sample_per_share=min_n_sample_per_share,
                                    partition_mode='uni')
            sub_test_sets = []
            for te_set in test_sets:
                _test_len_by_user = split(len(te_set), n_user_per_domain)

                base_idx = 0
                for tl in _test_len_by_user:
                    sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

            # rename
            train_sets = sub_train_sets
            test_sets = sub_test_sets
            clients = splitted_clients

        else:  # class iid
            split = Partitioner(rng=np.random.RandomState(partition_seed),
                                min_n_sample_per_share=min_n_sample_per_share,
                                partition_mode=partition_mode)
            splitted_clients = []

            val_sets, sub_train_sets = [], []
            for i_client, (dname, tr_set) in enumerate(zip(clients, train_sets)):
                _train_len_by_user = split(train_len[i_client], n_user_per_domain)
                print(f" {dname} | train split size: {_train_len_by_user}")

                base_idx = 0
                for i_user, tl in zip(range(n_user_per_domain), _train_len_by_user):
                    vl = int(val_ratio * tl)
                    tl = tl - vl

                    sub_train_sets.append(SubsetClass(tr_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

                    val_sets.append(Subset(tr_set, list(range(base_idx, base_idx + vl))))
                    base_idx += vl

                    splitted_clients.append(f"{dname}-{i_user}")

            # uniformly distribute test sets
            if consistent_test_class:
                split = Partitioner(rng=np.random.RandomState(partition_seed),
                                    min_n_sample_per_share=min_n_sample_per_share,
                                    partition_mode='uni')
            sub_test_sets = []
            for te_set in test_sets:
                _test_len_by_user = split(len(te_set), n_user_per_domain)

                base_idx = 0
                for tl in _test_len_by_user:
                    sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

        # rename
        train_sets = sub_train_sets
        test_sets = sub_test_sets
        clients = splitted_clients
    else:  # single user
        assert n_class_per_user <= 0, "Cannot split in Non-IID way when only one user for one " \
                                      f"domain. But got n_class_per_user={n_class_per_user}"
        val_len = [int(tl * val_ratio) for tl in train_len]

        val_sets = [Subset(tr_set, list(range(train_len[i_client]-val_len[i_client],
                                              train_len[i_client])))
                    for i_client, tr_set in enumerate(train_sets)]
        train_sets = [Subset(tr_set, list(range(train_len[i_client]-val_len[i_client])))
                      for i_client, tr_set in enumerate(train_sets)]

    # check the real sizes
    # if partition_method != 'dir':
    print(f" split users' train size: {[len(ts) for ts in train_sets]}")
    print(f" split users' val   size: {[len(ts) for ts in val_sets]}")
    print(f" split users' test  size: {[len(ts) for ts in test_sets]}")
    if val_ratio > 0:
        for i_ts, ts in enumerate(val_sets):
            if len(ts) <= 0:
                raise RuntimeError(f"user-{i_ts} not has enough val data.")

    train_loaders = [DataLoader(tr_set, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=partition_mode != 'uni') for tr_set in train_sets]
    test_loaders = [DataLoader(te_set, batch_size=test_batch_size, shuffle=shuffle_eval,
                               num_workers=num_workers, pin_memory=pin_memory)
                    for te_set in test_sets]
    if val_ratio > 0:
        val_loaders = [DataLoader(va_set, batch_size=batch_size, shuffle=shuffle_eval,
                                  num_workers=num_workers, pin_memory=pin_memory)
                       for va_set in val_sets]
    else:
        val_loaders = test_loaders

    return train_loaders, val_loaders, test_loaders, clients


def prepare_cifar_data(args, domains=['cifar10'], shuffle_eval=False, n_class_per_user=-1,
                       n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                       eq_domain_train_size=True, subset_with_logits=False,
                       consistent_test_class=True, partition_method='iid', alpha=0.3, enable_resize=False,
                       ):
    train_sets, test_sets = get_central_data('cifar10', domains, enable_resize=enable_resize)

    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size, percent=args.percent,
        min_n_sample_per_share=10 if n_class_per_user > 3 else 16, subset_with_logits=subset_with_logits,
        n_class_per_user=n_class_per_user,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        consistent_test_class=consistent_test_class, partition_method=partition_method, alpha=alpha,
    )
    return train_loaders, val_loaders, test_loaders, clients

# consistent_test_class = False by default
def prepare_cifar100_data(args, domains=['cifar100'], shuffle_eval=False, n_class_per_user=-1,
                       n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                       eq_domain_train_size=True, subset_with_logits=False,
                       consistent_test_class=True, partition_method='iid', alpha=0.3, enable_resize=False,
                       ):
    train_sets, test_sets = get_central_data('cifar100', domains, enable_resize=enable_resize)

    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size, percent=args.percent,
        min_n_sample_per_share=3, subset_with_logits=subset_with_logits,
        n_class_per_user=n_class_per_user,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        consistent_test_class=consistent_test_class, partition_method=partition_method, alpha=alpha,
    )
    return train_loaders, val_loaders, test_loaders, clients

def prepare_emnist_data(args, domains=['emnist'], shuffle_eval=False, n_class_per_user=-1,
                       n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                       eq_domain_train_size=True, subset_with_logits=False,
                       consistent_test_class=False, partition_method='iid', alpha=0.3, enable_resize=False
                       ):
    train_sets, test_sets = get_central_data('emnist', domains, enable_resize=enable_resize)

    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size, percent=args.percent,
        min_n_sample_per_share=3, subset_with_logits=subset_with_logits,
        n_class_per_user=n_class_per_user,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        consistent_test_class=consistent_test_class, partition_method=partition_method, alpha=alpha,
    )
    return train_loaders, val_loaders, test_loaders, clients

class SubsetWithLogits(Subset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices) -> None:
        super(SubsetWithLogits, self).__init__(dataset, indices)
        self.logits = [0. for _ in range(len(indices))]

    def __getitem__(self, idx):
        dataset_subset = self.dataset[self.indices[idx]]
        if isinstance(dataset_subset, tuple):
            return (*dataset_subset, self.logits[idx])
        else:
            return dataset_subset, self.logits[idx]

    def update_logits(self, idx, logit):
        self.logits[idx] = logit
