import argparse
import copy
import numpy as np
from torch import nn
from federated.aggregation import ModelAccumulator

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class _Federation:
    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        # data
        parser.add_argument('--percent', type=float, default=1,
                            help='percentage of dataset for training')
        parser.add_argument('--val_ratio', type=float, default=0.2,
                            help='ratio of train set for validation')
        parser.add_argument('--batch', type=int, default=128, help='batch size')
        parser.add_argument('--test_batch', type=int, default=128, help='batch size for test')

        # federated
        parser.add_argument('--pd_nuser', type=int, default=8, help='#users per domain.')
        parser.add_argument('--pr_nuser', type=int, default=-1, help='#users per comm round '
                                                                     '[default: all]')
        parser.add_argument('--pu_nclass', type=int, default=-1, help='#class per user. -1 or 0: all')
        parser.add_argument('--domain_order', choices=list(range(5)), type=int, default=0,
                            help='select the order of domains')
        parser.add_argument('--partition_mode', choices=['uni', 'dir'], type=str.lower, default='uni',
                            help='the mode when splitting domain data into users: uni - uniform '
                                 'distribution (all user have the same #samples); dir - Dirichlet'
                                 ' distribution (non-iid sample sizes)')
        parser.add_argument('--con_test_cls', action='store_true',
                            help='Ensure the test classes are the same training for a client. '
                                 'Meanwhile, make test sets are uniformly splitted for clients. '
                                 'Mainly influence class-niid settings.')
        parser.add_argument('--resouce_w', type=str2bool, default=False, help='use client-local BN stats (valid if tracking stats)')
        parser.add_argument('--partition_method', type=str, default='iid', choices=['iid', 'pat', 'dir', 'balanced-dir'],  help='pat: pathological distribution; dir: dirichlet distribution')
        parser.add_argument('--alpha', type=float, default=0.3, help='hyperparameter for dirichlet distribution')

    @classmethod
    def render_run_name(cls, args):
        run_name = f'__pd_nuser_{args.pd_nuser}'
        if args.partition_method != 'iid': run_name += f'__pm_{args.partition_method}'
        if args.partition_method == 'dir': run_name += f'__dir_{args.alpha}'
        if args.partition_method == 'balanced-dir': run_name += f'__bdir_{args.alpha}'
        if args.percent != 0.3: run_name += f'__pct_{args.percent}'
        if args.pu_nclass > 0: run_name += f"__pu_nclass_{args.pu_nclass}"
        if args.pr_nuser != -1: run_name += f'__pr_nuser_{args.pr_nuser}'
        if args.domain_order != 0: run_name += f'__do_{args.domain_order}'
        if args.partition_mode != 'uni': run_name += f'__part_md_{args.partition_mode}'
        if args.con_test_cls: run_name += '__ctc'
        if args.resouce_w: run_name += '__rweight'
        return run_name

    def __init__(self, data, args):
        self.args = args
        
        if data == 'Cifar10':
            num_classes = 10
            from dataloader import prepare_cifar_data
            from dataloader_utils import CifarDataset
            prepare_data = prepare_cifar_data
            DataClass = CifarDataset

            all_domains = CifarDataset.resorted_domains[args.domain_order]

            train_loaders, val_loaders, test_loaders, clients = prepare_data(
                args, domains=all_domains,
                n_user_per_domain=args.pd_nuser,
                n_class_per_user=args.pu_nclass,
                partition_seed=args.seed + 1,
                partition_mode=args.partition_mode,
                val_ratio=args.val_ratio,
                eq_domain_train_size=args.partition_mode == 'uni',
                consistent_test_class=args.con_test_cls,
                partition_method=args.partition_method,
                alpha=args.alpha,
                enable_resize=args.enable_resize,
            )
            clients = [c + ' ' + ('noised' if hasattr(args, 'adv_lmbd') and args.adv_lmbd > 0.
                                  else 'clean') for c in clients]
            
            self.train_loaders = train_loaders
            self.val_loaders = val_loaders
            self.test_loaders = test_loaders
            self.clients = clients
            self.num_classes = num_classes
            self.all_domains = all_domains

        elif data == 'Cifar100':
            num_classes = 100
            from dataloader import prepare_cifar100_data
            from dataloader_utils import CifarDataset100
            prepare_data = prepare_cifar100_data
            DataClass = CifarDataset100

            all_domains = CifarDataset100.resorted_domains[args.domain_order]

            train_loaders, val_loaders, test_loaders, clients = prepare_data(
                args, domains=all_domains,
                n_user_per_domain=args.pd_nuser,
                n_class_per_user=args.pu_nclass,
                partition_seed=args.seed + 1,
                partition_mode=args.partition_mode,
                val_ratio=args.val_ratio,
                eq_domain_train_size=args.partition_mode == 'uni',
                consistent_test_class=args.con_test_cls,
                partition_method=args.partition_method,
                alpha=args.alpha,
                enable_resize=args.enable_resize,
            )
            clients = [c + ' ' + ('noised' if hasattr(args, 'adv_lmbd') and args.adv_lmbd > 0.
                                  else 'clean') for c in clients]
            
            self.train_loaders = train_loaders
            self.val_loaders = val_loaders
            self.test_loaders = test_loaders
            self.clients = clients
            self.num_classes = num_classes
            self.all_domains = all_domains

        # Setup fed
        self.client_num = len(self.clients)
        client_weights = [len(tl.dataset) for tl in train_loaders]
        self.client_weights = [w / sum(client_weights) for w in client_weights]

        pr_nuser = args.pr_nuser if args.pr_nuser > 0 else self.client_num
        self.args.pr_nuser = pr_nuser
        self.client_sampler = UserSampler([i for i in range(self.client_num)], pr_nuser, mode='uni')

    def get_data(self):
        return self.train_loaders, self.val_loaders, self.test_loaders

    def make_aggregator(self, running_model, local_bn=False):
        self._model_accum = ModelAccumulator(running_model, self.args.pr_nuser, self.client_num,  local_bn=local_bn, resource_weight=self.args.resouce_w)
        return self._model_accum

    @property
    def model_accum(self):
        if not hasattr(self, '_model_accum'):
            raise RuntimeError(f"model_accum has not been set yet. Call `make_aggregator` first.")
        return self._model_accum

    def download(self, running_model, client_idx, strict=True):
        """Download (personalized) global model to running_model."""
        self.model_accum.load_model(running_model, client_idx, strict=strict)

    def upload(self, running_model, client_idx):
        """Upload client model."""
        self.model_accum.add(client_idx, running_model, self.client_weights[client_idx], self.client_num)

    def aggregate(self):
        """Aggregate received models and update global model."""
        self.model_accum.update_server_and_reset()


class UserSampler(object):
    def __init__(self, users, select_nuser, mode='all'):
        self.users = users
        self.total_num_user = len(users)
        self.select_nuser = select_nuser
        self.mode = mode
        if mode == 'all':
            assert select_nuser == self.total_num_user, "Conflict config: Select too few users."

    def iter(self):
        if self.mode == 'all' or self.select_nuser == self.total_num_user:
            sel = np.arange(len(self.users))
        elif self.mode == 'uni':
            sel = np.random.choice(self.total_num_user, self.select_nuser, replace=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        for i in sel:
            yield self.users[i]

    def __len__(self):
        return self.total_num_user