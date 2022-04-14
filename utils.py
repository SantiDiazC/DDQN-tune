import torch
import numpy as np
import torch.nn.functional as F


def train(q, q_target, memory, optimizer, batch_size, device, gamma):
    for i in range(10):                     # Update using 320(32(bath_size)x10) samples per episode

        s, a, r, s_prime, done_mask = memory.sample(batch_size)     # 32(batch_size) random samples
        q_out = q(s.to(device))                # s-> Shape: [batch_size(32)][4], q(s)->Shape:[batch_size(32)][2]
        q_a = q_out.gather(1, a.to(device))    # Select only the q value of the action actually taken
        max_q_prime = q_target(s_prime.to(device)).max(1)[0].unsqueeze(1)
        target = r.to(device) + gamma*max_q_prime*done_mask.to(device)    # r and done are two-dimensional

        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # weights & Biases update


def get_moving_average(scores, period):

    return np.mean(scores)

