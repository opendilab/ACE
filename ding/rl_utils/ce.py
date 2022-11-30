def l2_balance(label, loss):
    # loss: [batch, agent, dim]
    pos_mask = (label.abs() >= 1e-5).float()
    neg_mask = 1 - pos_mask
    pos_loss = (loss * pos_mask).sum((0,1)) / pos_mask.sum((0,1)).clamp(min=1.0)
    neg_loss = (loss * neg_mask).sum((0,1)) / neg_mask.sum((0,1)).clamp(min=1.0)
    return (pos_loss.mean() + neg_loss.mean()) / 2