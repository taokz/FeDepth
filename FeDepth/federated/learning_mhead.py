import sys
import numpy as np
import torch
from tqdm import tqdm
from models.utils.dual_bn import set_bn_mode

def train(model, data_loader, optimizer, loss_fun, device, client_idx, num_clients,
          adversary=None, adv_lmbd=0.5, start_iter=0, max_iter=np.inf, att_BNn=False, progress=True):
    
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
                # print("client:", client_idx, "alter_flag:", model.alter_flag)
                for name, param in model.named_parameters():
                    if model.alter_flag == 1:
                        if ("linear1" in name) or ("first" in name)  or ("first_norm" in name):
                            param.requires_grad = True
                        elif ("layer1" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear2" in name):
                            param.requires_grad = True
                        elif ("layer2" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 3:
                        if ("linear3" in name):
                            param.requires_grad = True
                        elif ("layer3" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 4:
                        if ("linear4" in name):
                            param.requires_grad = True
                        elif ("layer4" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 5:
                        if ("linear6" in name):
                            param.requires_grad = True
                        elif ("layer5" in name) or ("layer6" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        if ("linear9" in name):
                            param.requires_grad = True
                        elif ("layer7" in name) or ("layer8" in name) or ("layer9" in name):
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
                        if ("linear2" in name) or ("first" in name)  or ("first_norm" in name):
                            param.requires_grad = True
                        elif ("layer1" in name) or ("layer2" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear4" in name):
                            param.requires_grad = True
                        elif ("layer3" in name) or ("layer4" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        if ("linear9" in name):
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
                        if ("linear3" in name) or ("first" in name)  or ("first_norm" in name):
                            param.requires_grad = True
                        elif ("layer1" in name) or ("layer2" in name) or ("layer3" in name): #or ("layer4" in name) or ("layer5" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    elif model.alter_flag == 2:
                        if ("linear9" in name):
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
        
                output = model(data)
                loss = loss_fun(output, target)

                loss.backward()
                optimizer.step()

                loss_all += loss.item() * target.size(0)
                total += target.size(0)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()                        
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
