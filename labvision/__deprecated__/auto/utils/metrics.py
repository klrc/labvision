def acc_topk(logits, y, ks=(1, 3)):
    """
        Args:
            logits:
            y:
            ks: top-k accs to be evaluated.
    """
    metrics = {f'acc@top{k}': 0 for k in ks}
    for k in ks:
        metrics[f'acc@top{k}'] += (logits.topk(max((1, k)), 1, True, True)[1] == y.view(-1, 1)).sum().float().item()/logits.shape[0]
    return metrics
