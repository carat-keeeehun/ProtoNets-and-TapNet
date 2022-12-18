import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Runner(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

    def train(self, train_loader, model, optimizer, criterion, epoch):
        losses = AverageMeter()
        num_support = self.config.training.num_support_tr
        total_epoch = len(train_loader) * (epoch - 1)

        # switch to train mode
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = model(x)
            loss, acc1 = criterion(y_pred, y, num_support)

            losses.update(loss.item(), x.size(0))

            # compute gradient and do optimize step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[Train] Loss : {loss.item()} at {total_epoch + i}")
            print(f"[Train] Acc : {acc1.item()} at {total_epoch + i}")

        return losses.avg


    @torch.no_grad()
    def validate(self, val_loader, model, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        num_support = self.config.training.num_support_val
        total_epoch = len(val_loader) * (epoch - 1)

        # switch to evaluate mode
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = model(x)
            loss, acc1 = criterion(y_pred, y, num_support)

            losses.update(loss.item(), x.size(0))
            top1.update(acc1.item(), x.size(0))

            print(f"[Val] Loss : {loss.item()} at {total_epoch + i}")
            print(f"[Val] Acc : {acc1.item()} at {total_epoch + i}")

        return losses.avg, top1.avg
    
    @torch.no_grad()
    def test(self, test_loader, model, criterion):
        num_support = self.config.training.num_support_val
        losses = []
        top1 = []
        # switch to evaluate mode
        model.eval()

        for i, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = model(x)
            loss, acc1 = criterion(y_pred, y, num_support)

            losses.append(loss)
            top1.append(acc1)

            # probs = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(y, y_pred)]

            # predict_labels = y_pred.softmax(dim=1).argmax()

        return losses, top1