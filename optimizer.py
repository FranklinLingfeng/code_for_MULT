import torch.optim as optim


def adjust_learning_rate(args, optimizer, epoch):

    if epoch >= 0 and epoch < 20:
        lr = args.lr 
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40 and epoch < 60:
        lr = args.lr * 0.1
    elif epoch >= 60 and epoch < 80:
        lr = args.lr * 0.01
    elif epoch >= 80 and epoch < 90:
        lr = args.lr * 0.001

    optimizer.param_groups[0]["lr"] = lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]["lr"] = lr

    return lr


def select_optimizer(args, main_net):

    if args.optim == 'adam':
        ignored_params = list(map(id, main_net.module.bottleneck.parameters())) 
        
        base_params = filter(lambda p: id(p) not in ignored_params, main_net.module.parameters())

        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.lr},
            {'params': main_net.module.bottleneck.parameters(), 'lr': args.lr}],
            weight_decay=5e-4)

    return optimizer
