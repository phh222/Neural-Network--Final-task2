import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.cifar100 import get_cifar100_dataloaders
from models.cnn import ResNet18
from models.transformer import VisionTransformer1
from utils.augmentations import CutMix
from utils.train_eval import train, test
from utils.scheduler import get_optimizer_and_scheduler

def main():
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()
    mixup_fn = CutMix(num_classes=100, alpha=1.0)

    trainloader, testloader = get_cifar100_dataloaders()

    #cnn_model = ResNet18().cuda()
    transformer_model = VisionTransformer1().cuda()


    #cnn_optimizer, cnn_scheduler = get_optimizer_and_scheduler(cnn_model, num_epochs)
    transformer_optimizer, transformer_scheduler = get_optimizer_and_scheduler(transformer_model, num_epochs)

    #初始化 TensorBoard writer
    writer = SummaryWriter()
    best_transformer_acc = 0
    best_cnn_acc = 0

    for epoch in range(num_epochs):
        #cnn_train_loss = train(cnn_model, trainloader, criterion, cnn_optimizer, cnn_scheduler, epoch, mixup_fn)
        transformer_train_loss = train(transformer_model, trainloader, criterion, transformer_optimizer, transformer_scheduler, epoch, mixup_fn)

        #cnn_val_loss, cnn_acc = test(cnn_model, testloader, criterion)
        transformer_val_loss, transformer_acc = test(transformer_model, testloader, criterion)

        #将训练和测试的损失及准确率记录到 TensorBoard
        # writer.add_scalar('训练集/CNN损失', cnn_train_loss, epoch)
        # writer.add_scalar('测试集/CNN损失', cnn_val_loss, epoch)
        # writer.add_scalar('测试集/CNN准确率', cnn_acc, epoch)
        
        writer.add_scalar('训练集/Transformer损失', transformer_train_loss, epoch)
        writer.add_scalar('测试集/Transformer损失', transformer_val_loss, epoch)
        writer.add_scalar('测试集/Transformer准确率', transformer_acc, epoch)
        

        print(f'第 {epoch+1}/{num_epochs} 轮')
        # print(f'CNN       - 训练损失: {cnn_train_loss:.4f}, 测试损失: {cnn_val_loss:.4f}, 准确率: {cnn_acc:.2f}%')
        print(f'Transformer - 训练损失: {transformer_train_loss:.4f}, 测试损失: {transformer_val_loss:.4f}, 准确率: {transformer_acc:.2f}%')
        #保存模型权重
        if transformer_acc>best_transformer_acc:
            best_transformer_acc = transformer_acc
            torch.save(transformer_model.state_dict(), f'results/Transformer/lr0.001warmup_t5warmup_lr_init1e-4/best_transformer_model.pth')

        # if cnn_acc>best_cnn_acc:
        #     best_cnn_acc = cnn_acc
        #     torch.save(cnn_model.state_dict(), f'results/CNN/lr0.001warmup_t5warmup_lr_init1e-4/best_cnn_model.pth')
    writer.close()

    #初始化 TensorBoard writer
    # writer = SummaryWriter()
    # best_acc = 0

    # for epoch in range(num_epochs):
        
    #     transformer_train_loss = train(transformer_model, trainloader, criterion, transformer_optimizer, transformer_scheduler, epoch, mixup_fn)

        
    #     transformer_val_loss, transformer_acc = test(transformer_model, testloader, criterion)

    #     # if epoch % 10 ==0:
    #     #     torch.save(transformer_model.state_dict(),"transformer.pth")
    #     if transformer_acc>best_acc:
    #         best_acc = transformer_acc
    #         torch.save(transformer_model.state_dict(), f'results/Transformer/lr0.0001stepsize10gamma0.5/best_transformer_model.pth')

    #     writer.add_scalar('训练集/Transformer损失', transformer_train_loss, epoch)
    #     writer.add_scalar('测试集/Transformer损失', transformer_val_loss, epoch)
    #     writer.add_scalar('测试集/Transformer准确率', transformer_acc, epoch)

    #     print(f'第 {epoch+1}/{num_epochs} 轮')
        
    #     print(f'Transformer - 训练损失: {transformer_train_loss:.4f}, 测试损失: {transformer_val_loss:.4f}, 准确率: {transformer_acc:.2f}%')
    # writer.close()

if __name__ == '__main__':
    main()


    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(count_parameters(VisionTransformer1))