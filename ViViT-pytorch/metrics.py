class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean()


def precision(preds, labels):
    tp = (preds.argmax(dim=1) & labels).float().sum()
    fn = ((~preds.argmax(dim=1)) & labels).float().sum()
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


def recall(preds, labels):
    tp = (preds.argmax(dim=1) & labels).float().sum()
    fp = (preds.argmax(dim=1) & (~labels)).float().sum()
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def f1_score(preds, labels):
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    if prec + rec == 0:
        return 0
    else:
        return 2 * (prec * rec) / (prec + rec)