import gym
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter


env = gym.make("CartPole-v0")
env = env.unwrapped
mid_shape = 10
episode_num = 2500
observer_shape = env.observation_space.shape[0]
action_num = env.action_space.n
memory_capacity = 2000
epsilon = 0.01
epsilon_change_speed = 1
gamma = 0.8
lr = 0.0001
batchsize = 32
writer = SummaryWriter()


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(observer_shape, mid_shape),
            nn.ReLU(),
            nn.Linear(mid_shape, action_num),
        )

    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.target_replace = 100
        self.memory_counter = 0
        self.batchsize = batchsize
        self.lr = lr
        self.memory_capacity = memory_capacity
        self.observer_shape = observer_shape
        self.memory = np.zeros((self.memory_capacity, 2*self.observer_shape+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > epsilon:
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:
            action = np.random.randint(0, action_num)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(sample[:, self.observer_shape:self.observer_shape+1])
        r = torch.Tensor(sample[:, self.observer_shape+1:self.observer_shape+2])
        s_ = torch.Tensor(sample[:, self.observer_shape+2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + gamma * q_next.max(1, True)[0].data

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

for episode in range(episode_num):
    s = env.reset()
    print("#-------- episode:" + str(episode))
    avg_reward = 0
    episode_step = 0
    while True:
        episode_step += 1
        env.render()
        # choose action
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        avg_reward += r

        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > memory_capacity:
            dqn.learn()
        if done:
            print("avg_reward:" + str(r / episode_step))
            writer.add_scalar("scalar/reward", r / episode_step, episode)
            break
        s = s_

env.close()

torch.save({
    "eval_net": dqn.eval_net.state_dict(),
    "target_net": dqn.target_net.state_dict(),
    "memory": dqn.memory,
}, "./model.dat")

# reference: https://www.jianshu.com/p/18cbcca19837