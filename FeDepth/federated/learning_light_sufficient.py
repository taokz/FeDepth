import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from models.utils.dual_bn import set_bn_mode
import torch.nn.functional as F
from torch import optim, nn

class DML(nn.Module):
    '''
    Deep Mutual Learning
    https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
    '''
    def __init__(self):
        super(DML, self).__init__()

    def forward(self, out1, out2):
        loss = F.kl_div(F.log_softmax(out1, dim=1),
                        F.softmax(out2, dim=1),
                        reduction='batchmean')

        return loss

def train(model, data_loader, optimizer, loss_fun, device, client_idx, num_clients,
          adversary=None, adv_lmbd=0.5, start_iter=0, max_iter=np.inf, att_BNn=False, progress=True,
          aux_models=None, global_lr=None, config=None):
    
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    # data_iterator = iter(data_loader)
    tqdm_iters = tqdm(range(start_iter, max_iter), file=sys.stdout) \
        if progress else range(start_iter, max_iter)
    
    if adversary is None:
        # ordinary training.
        set_bn_mode(model, False)  # set clean mode
        
        #if client_idx < num_clients/4*1:
        # 4 here means number of budgets, e.g. [1/6, 1/3, 1/2, 1]
        if int(client_idx * 4 / num_clients) == 0: 
            model.comp_flag = 1
            for it in range(6):
                data_iterator = iter(data_loader)
                model.alter_flag = it + 1
                for name, param in model.named_parameters():
                    if model.alter_flag == 1:
                        if ("linear" in name) or ("first" in name)  or ("first_norm" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer1" in name) or ("link1" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer2" in name) or ("link1" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 3:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer3" in name) or ("link1" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 4:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer4" in name) or ("link2" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 5:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer5" in name) or ("layer6" in name) or ("link2" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer7" in name) or ("layer8" in name) or ("layer9" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                for step in tqdm_iters:
                # for data, target in tqdm(data_loader, file=sys.stdout):
                    try:
                        data, target = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(data_loader)
                        data, target = next(data_iterator)
                    optimizer.zero_grad()

                    data = data.to(device)
                    target = target.to(device)
            
                    output = model(data)
                    loss = loss_fun(output, target)

                    loss.backward()
                    optimizer.step()
                    if it == 5:
                        loss_all += loss.item() * target.size(0)
                        total += target.size(0)
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.view(-1)).sum().item()

        elif int(client_idx * 4 / num_clients) == 1:
            model.comp_flag = 2
            for it in range(3):
                data_iterator = iter(data_loader)
                model.alter_flag = it + 1
                for name, param in model.named_parameters():
                    if model.alter_flag == 1:
                        if ("linear" in name) or ("first" in name)  or ("first_norm" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer1" in name) or ("layer2" in name) or ("link1" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer3" in name) or ("layer4" in name) or ("link2" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer5" in name) or ("layer6" in name) or ("layer7" in name) or ("layer8" in name) or ("layer9" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                for step in tqdm_iters:
                # for data, target in tqdm(data_loader, file=sys.stdout):
                    try:
                        data, target = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(data_loader)
                        data, target = next(data_iterator)
                    optimizer.zero_grad()

                    data = data.to(device)
                    target = target.to(device)
            
                    output = model(data)
                    loss = loss_fun(output, target)

                    loss.backward()
                    optimizer.step()
                    if it == 2:
                        loss_all += loss.item() * target.size(0)
                        total += target.size(0)
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.view(-1)).sum().item()

        elif int(client_idx * 4 / num_clients) == 2:
            model.comp_flag = 3
            for it in range(2):
                data_iterator = iter(data_loader)
                model.alter_flag = it + 1
                for name, param in model.named_parameters():
                    if model.alter_flag == 1:
                        if ("linear" in name) or ("first" in name)  or ("first_norm" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer1" in name) or ("layer2" in name) or ("layer3" in name) or ("link1" in name): #or ("layer4" in name) or ("layer5" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear" in name) or ("bn9" in name):
                            param.requires_grad = True
                        elif ("layer4" in name) or ("layer5" in name) or ("layer6" in name) or ("layer7" in name) or ("layer8" in name) or ("layer9" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                for step in tqdm_iters:
                # for data, target in tqdm(data_loader, file=sys.stdout):
                    try:
                        data, target = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(data_loader)
                        data, target = next(data_iterator)
                    optimizer.zero_grad()

                    data = data.to(device)
                    target = target.to(device)
            
                    output = model(data)
                    loss = loss_fun(output, target)

                    loss.backward()
                    optimizer.step()
                    if it == 1:
                        loss_all += loss.item() * target.size(0)
                        total += target.size(0)
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.view(-1)).sum().item()

        elif int(client_idx * 4 / num_clients) == 3:
            try:
                aux_save_name = os.path.join(config.save_path, config.model+'_'+str(client_idx)+'.pt')
                aux_model_1 = torch.load_state_dict(torch.load(aux_save_name))
            except:
                aux_model_1 = aux_models[0]

            if config.opt == 'sgd':
                aux_opt_1 = optim.SGD(params=aux_model_1.parameters(), lr=global_lr,momentum=0.9, weight_decay=5e-4)
            elif config.opt == 'adam':
                aux_opt_1 = optim.Adam(params=aux_model_1.parameters(), lr=global_lr)
            set_bn_mode(aux_model_1, False)  # set clean mode
            loss_fn_kl = DML()
            lambda_kd = 1.0 # by default

            model.comp_flag = 4
            data_iterator = iter(data_loader)
            for name, param in model.named_parameters():
                param.requires_grad = True

            for step in tqdm_iters:
            # for data, target in tqdm(data_loader, file=sys.stdout):
                try:
                    data, target = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_loader)
                    data, target = next(data_iterator)
                optimizer.zero_grad()

                data = data.to(device)
                target = target.to(device)
                # calculate logits from global net0 and aux net1
                output = model(data)
                aux_output_1 = aux_model_1(data)

                # for global net
                cls0_loss = loss_fun(output, target)
                kd0_loss  = loss_fn_kl(output, aux_output_1.detach()) * lambda_kd
                net0_loss = cls0_loss + kd0_loss
                # for aux model
                cls1_loss = loss_fun(aux_output_1, target)
                kd1_loss  = loss_fn_kl(aux_output_1, output.detach()) * lambda_kd
                net1_loss = cls1_loss + kd1_loss
        
                output = model(data)
                loss = loss_fun(output, target)

                # update net0 & net1 (aux)
                optimizer.zero_grad()
                net0_loss.backward()
                aux_opt_1.zero_grad()
                net1_loss.backward()
                optimizer.step()
                aux_opt_1.step()

                # recording
                loss_all += loss.item() * target.size(0)
                total += target.size(0)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
            
            # save aux models locally
            aux_save_name = os.path.join(config.save_path, config.model+'_'+str(client_idx)+'.pt')
            torch.save(aux_model_1.state_dict(), aux_save_name)                        
    # else:
    # 	# TODO
    # 	pass
    return loss_all / total, correct / total


def test(model, data_loader, loss_fun, device, valid=True, depth_ratio=None, adversary=None, progress=False, ensemble=None, test_idx=None, num_client=None):
    """Run test single model.
    Returns: loss, acc
    """
    model.eval()

    # how to evaluate, it is a question.
    # full net? partial net?
    if valid or (depth_ratio is None):
        # if test_idx < num_client/4*1:
        #     model.comp_flag = 1
        #     model.alter_flag = 3
        # elif test_idx < num_client/4*2:
        #     model.comp_flag = 2
        #     model.alter_flag = 4
        # elif test_idx < num_client/4*3:
        #     model.comp_flag = 3
        #     model.alter_flag = 2
        # else:
        #     model.comp_flag = 4
        model.comp_flag = 4
    else:
        # if depth_ratio == 1:
        #     model.comp_flag = 1
        #     model.alter_flag = 3
        # elif depth_ratio == 2:
        #     model.comp_flag = 2
        #     model.alter_flag = 4
        # elif depth_ratio == 3:
        #     model.comp_flag = 3
        #     model.alter_flag = 2
        # else:
        #     model.comp_flag = 4
        model.comp_flag = 4

    loss_all, total, correct = 0, 0, 0
    for data, target in tqdm(data_loader, file=sys.stdout, disable=not progress):
        data, target = data.to(device), target.to(device)
        if adversary:
            with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                data = adversary.perturb(data, target)

        with torch.no_grad():
            output = model(data)
            loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total
