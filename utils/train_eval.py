import torch

import torch

def train(model, trainloader, criterion, optimizer, scheduler, epoch, mixup_fn):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if mixup_fn:
            inputs, (targets_a, targets_b, lam) = mixup_fn(inputs, targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        if mixup_fn:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step(epoch)
    return train_loss/len(trainloader)

def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / len(testloader), 100. * correct / total



