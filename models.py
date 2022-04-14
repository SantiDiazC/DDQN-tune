import torch
import collections
import random
import torch.nn.functional as F
import torch.nn as nn

class ReplayBuffer():
    def __init__(self, config):
        self.buffer = collections.deque(maxlen=config["buffer_limit"])  # .deque allows append and discard at both ends

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # Randomly extract n(batch_size = 32) data from self.buffer
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])  # adding dimension to make all same dimension (n=2)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float),\
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)    # TBD --> Design
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)    # TBD

    def forward(self, x):
        x = F.relu(self.fc1(x))     # TBD --> Design
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # TBD
        return x

    def sample_action(self, obs, epsilon):

        q_out = self.forward(obs)   # 2 action values - tensor
        r_value = random.random()   # 0~1 random value - float

        if r_value < epsilon:
            return random.randint(0, 1)
        else:
            return q_out.argmax().item()    # selects the index (action) with highest q value among tensor elements,
                                            # item() casts a single tensor to a python variable

