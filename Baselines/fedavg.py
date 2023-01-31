"""FedAvg"""
import os, argparse, time
import numpy as np
import wandb
import torch
from torch import nn, optim
# federated
from federated.learning import train, test
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, LocalMaskCrossEntropyLoss, str2bool
from utils.config import CHECKPOINT_ROOT

# NOTE import desired federation
from federated.core import _Federation as Federation
from federated.core import AdversaryCreator


def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Cifar10': 'preresnet20', 'Cifar100': 'preresnet20'}[args.data]
    run_name = f'{args.model}'
    if args.width_scale != 1.: run_name += f'x{args.width_scale}'
    run_name += Federation.render_run_name(args)
    # log non-default args
    # if args.seed != 1: run_name += f'__seed_{args.seed}'
    # opt
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.lr != 1e-1: run_name += f'_lr_{args.lr}'
    if args.opt != 'sgd': run_name += f'__opt_{args.opt}'
    if args.batch != 128: run_name += f'__batch_{args.batch}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    # slimmable
    if args.no_track_stat: run_name += f"__nts"
    if args.no_mask_loss: run_name += f'__nml'
    # adv train
    if args.adv_lmbd > 0:
        run_name += f'__at{args.adv_lmbd}'

    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE


def get_model_fh(data, model, pretrained=False):
    if (data == 'Cifar10') or (data == 'Cifar100') or (data == 'EMNIST'):
        if model in ['preresnet20']:  # From heteroFL
            from nets.HeteFL.preresne import resnet20
            ModelClass = resnet20
        elif model in ['simple_vit']:
            from nets.HeteFL.svit import svit
            ModelClass = svit
        elif model in ['vit_tiny_timm']:
            # from nets.HeteFL import timm
            import timm
            if pretrained:
                ModelClass = timm.create_model('vit_tiny_patch16_224_fedavg', pretrained=True)
            else:
                ModelClass = timm.create_model('vit_tiny_patch16_224_fedavg', pretrained=False)    
        else:
            raise ValueError(f"Invalid model: {model}")
    return ModelClass


def fed_test(fed, running_model, val_loaders, verbose, adversary=None):
    mark = 's' if adversary is None else 'r'
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    for client_idx in range(fed.client_num):
        fed.download(running_model, client_idx)
        # Test
        val_loss, val_acc = test(running_model, val_loaders[client_idx], loss_fun, device,
                                 adversary=adversary)

        # Log
        val_loss_mt.append(val_loss)
        val_acc_list[client_idx] = val_acc
        if verbose > 0:
            print(' {:<19s} Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                'User-'+fed.clients[client_idx], mark.upper(), val_loss, mark.upper(), val_acc))
        wandb.log({
            f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
        }, commit=False)
    return val_acc_list, val_loss_mt.avg


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    #gpu
    parser.add_argument('--gpu', type=int, default=0, help='device idx')
    
    # basic problem setting
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data', type=str, default='Digits', help='data name')
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--width_scale', type=float, default=1., help='model width scale')
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    parser.add_argument('--no_mask_loss', type=str2bool, default=False, help='disable local mask cross entropy')
    parser.add_argument('--enable_resize', type=str2bool, default=False, help='enable image resize for vision transformer')
    parser.add_argument('--enable_pretrained', type=str2bool, default=False, help='enable loading pretrained vision transformer')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')
    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_sch', type=str, default='none', help='learning rate schedule')
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=300, help='#iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='#epochs in local train')
    # adversarial train
    parser.add_argument('--adv_lmbd', type=float, default=0.,
                        help='adv coefficient in [0,1]; default 0 for standard training.')
    parser.add_argument('--test_noise', choices=['none', 'LinfPGD'], default='none')
    # analysis
    parser.add_argument('--analysis', type=int, default=0, help='model performacne analysis before/after aggregation: 1 -- perform analysis')
    # additional information for naming
    parser.add_argument('--additional', type=str, default=None, help='for convinient naming')
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # set experiment files, wandb
    exp_folder = f'fedavg_{args.data}_iclr'
    if args.analysis == 1:
        exp_folder += f'_analysis'
    if args.enable_pretrained == False:
        exp_folder += f'_unpretrained'
    if args.additional is not None:
        exp_folder += f'_{args.additional}'
    # exp_folder = f'test'
    run_name, SAVE_FILE = render_run_name(args, exp_folder)
    wandb.init(group=run_name[:120], project=exp_folder,
               mode='offline' if args.no_log else 'online',
               config={**vars(args), 'save_file': SAVE_FILE})


    # /////////////////////////////////
    # ///// Fed Dataset and Model /////
    # /////////////////////////////////
    fed = Federation(args.data, args)
    # Data
    train_loaders, val_loaders, test_loaders = fed.get_data()
    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")

    # Model
    ModelClass = get_model_fh(args.data, args.model, args.enable_pretrained)
    if args.enable_resize == False:
        running_model = ModelClass(
            track_running_stats=not args.no_track_stat, num_classes=fed.num_classes,
            width_scale=args.width_scale,
        ).to(device)
    else:
        # TODO: add required slimmable variabels
        if args.data == 'Cifar10':
            # attempt 2
            ModelClass.width_scale=args.width_scale
            ModelClass.head = nn.Linear(ModelClass.head.weight.shape[1], 10)
            ModelClass.num_classes = 10
        elif args.data == 'Cifar100':
            # attempt 2
            ModelClass.width_scale=args.width_scale
            ModelClass.head = nn.Linear(ModelClass.head.weight.shape[1], 100)
            ModelClass.num_classes = 100
        elif args.data == 'EMNIST':
            # attempt 2
            ModelClass.width_scale=args.width_scale
            ModelClass.head = nn.Linear(ModelClass.head.weight.shape[1], 62)
            ModelClass.num_classes = 62
        running_model = ModelClass.to(device)

    # adversary]
    if args.adv_lmbd > 0. or args.test:
        make_adv = AdversaryCreator(args.test_noise if args.test else 'LinfPGD')
        adversary = make_adv(running_model)
    else:
        adversary = None

    # Loss
    if args.pu_nclass > 0 and not args.no_mask_loss:  # niid
        loss_fun = LocalMaskCrossEntropyLoss(fed.num_classes)
    else:
        loss_fun = nn.CrossEntropyLoss()

    # Use running model to init a fed aggregator
    fed.make_aggregator(running_model)


    # /////////////////
    # //// Resume /////
    # /////////////////
    # log the best for each model on all datasets
    best_epoch = 0
    best_acc = [0. for j in range(fed.client_num)]
    train_elapsed = [[] for _ in range(fed.client_num)]
    start_epoch = 0
    if args.resume or args.test:
        if os.path.exists(SAVE_FILE):
            print(f'Loading chkpt from {SAVE_FILE}')
            checkpoint = torch.load(SAVE_FILE)
            best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
            train_elapsed = checkpoint['train_elapsed']
            start_epoch = int(checkpoint['a_iter']) + 1
            fed.model_accum.load_state_dict(checkpoint['server_model'])

            print('Resume training from epoch {} with best acc:'.format(start_epoch))
            for client_idx, acc in enumerate(best_acc):
                print(' Best user-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                    fed.clients[client_idx], best_epoch, acc))
        else:
            if args.test:
                raise FileNotFoundError(f"Not found checkpoint at {SAVE_FILE}")
            else:
                print(f"Not found checkpoint at {SAVE_FILE}\n **Continue without resume.**")


    # ///////////////
    # //// Test /////
    # ///////////////
    if args.test:
        wandb.summary[f'best_epoch'] = best_epoch

        # Set up model with specified width
        print(f"  Test model: {args.model}x{args.width_scale}"
              + ('' if args.test_noise == 'none' else f'with {args.test_noise} noise'))

        # Test on clients
        test_acc_mt = AverageMeter()
        for test_idx, test_loader in enumerate(test_loaders):
            fed.download(running_model, test_idx)
            _, test_acc = test(running_model, test_loader, loss_fun, device,
                               adversary=adversary)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))
            
            wandb.summary[f'{fed.clients[test_idx]} test acc'] = test_acc
            test_acc_mt.append(test_acc)

        # Profile model FLOPs, sizes (#param)
        from nets.profile_func import profile_model

        flops, params = profile_model(running_model, device=device)
        wandb.summary['GFLOPs'] = flops / 1e9
        wandb.summary['model size (MB)'] = params / 1e6
        print('GFLOPS: %.4f, model size: %.4fMB' % (flops / 1e9, params / 1e6))

        print(f"\n Average Test Acc: {test_acc_mt.avg}")
        wandb.summary[f'avg test acc'] = test_acc_mt.avg
        wandb.finish()

        exit(0)


    # ////////////////
    # //// Train /////
    # ////////////////
    # LR scheduler
    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.iters, eta_max=args.lr, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step':
        lr_sch = MultiStepLR(args.lr, milestones=[150, 250], gamma=0.1, last_epoch=start_epoch)
    else:
        assert args.lr_sch == 'none', f'Invalid lr_sch: {args.lr_sch}'
        lr_sch = None
    for a_iter in range(start_epoch, args.iters):
        # set global lr
        global_lr = args.lr if lr_sch is None else lr_sch.step()
        wandb.log({'global lr': global_lr}, commit=False)

        # ----------- Train Client ---------------
        train_loss_mt, train_acc_mt = AverageMeter(), AverageMeter()
        # For analysis
        valid_acc_before_mt = AverageMeter()
        valid_acc_after_mt = AverageMeter()
        analysis_client_list = []        
        print("============ Train epoch {} ============".format(a_iter))
        for client_idx in fed.client_sampler.iter():
            analysis_client_list.append(client_idx)
            start_time = time.process_time()

            # Download global model to local
            fed.download(running_model, client_idx)

            # (Alg 3) Local Train
            if args.opt == 'sgd':
                optimizer = optim.SGD(params=running_model.parameters(), lr=global_lr,
                                      momentum=0.9, weight_decay=5e-4)
            elif args.opt == 'adam':
                optimizer = optim.Adam(params=running_model.parameters(), lr=global_lr)
            else:
                raise ValueError(f"Invalid optimizer: {args.opt}")
            train_loss, train_acc = train(
                running_model, train_loaders[client_idx], optimizer, loss_fun, device,
                max_iter=mean_batch_iters * args.wk_iters if args.partition_mode != 'uni'
                            else len(train_loaders[client_idx]) * args.wk_iters,
                progress=args.verbose > 0,
                adversary=adversary, adv_lmbd=args.adv_lmbd,
            )

            # Test before aggregation (only works for n_client = 8, partition_r = 100% in the paper)
            # Test on clients
            if args.analysis == 1:
                _, val_acc_before = test(running_model, val_loaders[client_idx], loss_fun, device, adversary=adversary) 
                print(' {:<11s}| Valid  Acc Before Agg: {:.4f}'.format(fed.clients[client_idx], val_acc_before))
                wandb.summary[f'{fed.clients[client_idx]} valid acc before agg'] = val_acc_before
                valid_acc_before_mt.append(val_acc_before)

            # Upload
            fed.upload(running_model, client_idx)

            # Log
            client_name = fed.clients[client_idx]
            elapsed = time.process_time() - start_time
            wandb.log({f'{client_name}_train_elapsed': elapsed}, commit=False)
            train_elapsed[client_idx].append(elapsed)

            train_loss_mt.append(train_loss), train_acc_mt.append(train_acc)
            print(f' User-{client_name:<10s} Train | Loss: {train_loss:.4f} |'
                  f' Acc: {train_acc:.4f} | Elapsed: {elapsed:.2f} s')
            wandb.log({
                f"{client_name} train_loss": train_loss,
                f"{client_name} train_acc": train_acc,
            }, commit=False)

        # Use accumulated model to update server model
        fed.aggregate()

        # Test after aggregation (only works for n_client = 8, partition_r = 100% in the paper)
        # Test on clients
        if args.analysis == 1:
            for analysis_client_idx in analysis_client_list:
                _, val_acc_after = test(running_model, val_loaders[analysis_client_idx], loss_fun, device, adversary=adversary) 
                print(' {:<11s}| Valid  Acc After Agg: {:.4f}'.format(fed.clients[analysis_client_idx], val_acc_after))
                wandb.summary[f'{fed.clients[analysis_client_idx]} valid acc before agg'] = val_acc_after
                valid_acc_after_mt.append(val_acc_after)

        # ----------- Validation ---------------
        val_acc_list, val_loss = fed_test(fed, running_model, val_loaders, args.verbose)
        if args.adv_lmbd > 0:
            print(f' Avg Val SAcc {np.mean(val_acc_list) * 100:.2f}%')
            wandb.log({'val_sacc': np.mean(val_acc_list)}, commit=False)
            val_racc_list, val_rloss = fed_test(fed, running_model, val_loaders, args.verbose,
                                                adversary=adversary)
            print(f' Avg Val RAcc {np.mean(val_racc_list) * 100:.2f}%')
            wandb.log({'val_racc': np.mean(val_racc_list)}, commit=False)

            val_acc_list = [(1-args.adv_lmbd) * sa_ + args.adv_lmbd * ra_
                            for sa_, ra_ in zip(val_acc_list, val_racc_list)]
            val_loss = (1-args.adv_lmbd) * val_loss + args.adv_lmbd * val_rloss

        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} Acc {train_acc_mt.avg*100:.1f}%'
              f' | Val Acc {np.mean(val_acc_list) * 100:.2f}%')
        wandb.log({
            f"train_loss": train_loss_mt.avg,
            f"train_acc": train_acc_mt.avg,
            f"val_loss": val_loss,
            f"val_acc": np.mean(val_acc_list),
        }, commit=False)

        # analysis log
        if args.analysis == 1:
            #print(len(valid_acc_before_mt))
            #print(len(valid_acc_after_mt))
            print(f' [Aggregation Analysis] Avg Valid Acc Before Agg {valid_acc_before_mt.avg*100:.2f}%'
                f' | Avg Val Acc After Agg {valid_acc_after_mt.avg*100:.2f}%')
            wandb.log({
                f"val_acc_before": valid_acc_before_mt.avg,
                f"val_acc_after": valid_acc_after_mt.avg,
            }, commit=False)

        # ----------- Save checkpoint -----------
        if np.mean(val_acc_list) > np.mean(best_acc):
            best_epoch = a_iter
            for client_idx in range(fed.client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                if args.verbose > 0:
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                          fed.clients[client_idx], best_epoch, best_acc[client_idx]))
            print(' [Best Val] Acc {:.4f}'.format(np.mean(val_acc_list)))

            # Save
            print(f' Saving the local and server checkpoint to {SAVE_FILE}')
            save_dict = {
                'server_model': fed.model_accum.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter,
                'all_domains': fed.all_domains,
                'train_elapsed': train_elapsed,
            }
            torch.save(save_dict, SAVE_FILE)
        wandb.log({
            f"best_val_acc": np.mean(best_acc),
        }, commit=True)
