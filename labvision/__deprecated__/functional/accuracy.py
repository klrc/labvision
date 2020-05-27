def accuracy_topk(logits, y, k):
    acc = (logits.topk(max((1, k)), 1, True, True)[1] == y.view(-1, 1)).sum().float().item()/logits.shape[0]
    return acc


def accuracy(logits, y):
    return accuracy_topk(logits, y, 1)


def accuracy_top3(logits, y):
    return accuracy_topk(logits, y, 3)
