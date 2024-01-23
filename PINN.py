import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

criteria =nn.MSELoss()

#損失関数の定義
def physics_informed_loss(x,t,net):
    
    u=net(x,t)
    
    #時間の一階微分
    u_t=torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True        
    )[0]
    #空間の一階微分
    u_x=torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True        
    )[0]
    
    #空間の二階微分
    u_xx=torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True        
    )[0]
    
    pinn_loss = u_t + u * u_x - (0.01 / np.pi) * u_xx #損失関数
    zeros_t=torch.zeros(pinn_loss.size()).to(device) #0を作る
    pinn_loss_ = criteria(pinn_loss, zeros_t) #zeros_tとpinn_lossの差を変えす（損失を導出する）
    
    return pinn_loss_

#初期条件に関する損失関数
def initial_condition_loss(x, t, net, u_ini):
    u = net(x, t)
    
    ini_condition_loss = criteria(u, u_ini)
    return ini_condition_loss

#境界条件に関する損失関数
def boundary_condition_loss(x, t, net, u_bc):
    u = net(x, t)
    
    bc_condition_loss = criteria(u, u_bc)
    return bc_condition_loss

#初期条件
ini_sample_size = 2000
x_ini = np.linspace(-1, 1, ini_sample_size)
X_ini = np.zeros([ini_sample_size, 2])
X_ini[:, 0] = x_ini
u_ini = -np.sin(np.pi * x_ini)
#plt.plot(x_ini, u_ini)

X_ini_t = torch.tensor(X_ini, requires_grad=True).float().to(device)
u_ini_t = torch.tensor(u_ini, requires_grad=True).float().to(device).unsqueeze(dim=1)


#境界条件
x_bc = np.array([-1.0, 1.0])

#サンプルサイズ
bc_sample_size=200

#時刻に関する情報
t_bc=np.linspace(0, 1.0, bc_sample_size)

# x = -1.0
X_bc1=np.zeros([bc_sample_size,2])
X_bc1[:, 0] = -1.0
X_bc1[:, 1] = t_bc

# x = 1.0
X_bc2=np.zeros([bc_sample_size,2])
X_bc2[:, 0] = 1.0
X_bc2[:, 1] = t_bc

# stack
X_bc_stack = np.vstack([X_bc1, X_bc2])
u_bc_stack = np.zeros(X_bc_stack.shape[0]) #すべての時刻において0、zerosを使ってshapeを使って活用

#Tensor
X_bc_t=torch.tensor(X_bc_stack, requires_grad=True).float().to(device)
u_bc_t=torch.tensor(u_bc_stack, requires_grad=True).float().to(device)

# sampling pointの設定
x_ = np.linspace(-1, 1, 100)
t_ = np.linspace(0, 1, 100)

X, T = np.meshgrid(x_, t_, indexing="ij")

x_flat = X.flatten()
t_flat = T.flatten()

# Sampling size
sampling_size = 5000
random_idx = np.random.choice(np.arange(x_flat.shape[0]), size= sampling_size, replace=False)

#
# Sampling
#

x_sampled = x_flat[random_idx]
#print(x_sampled)
t_sampled = t_flat[random_idx]

X_sampled = np.zeros([sampling_size, 2])
X_sampled[:, 0] = x_sampled
X_sampled[:, 1] = t_sampled

X_sample_t=torch.tensor(X_sampled, requires_grad=True).float().to(device)

#MLPの実装
class PINN(torch.nn.Module):
    def __init__(self,activation="relu"):
        super().__init__()
        self.regressor = nn.Linear(5,1)
        self.activation = activation
        self.linear1=self.linear(2, 5, activation=self.activation)
        self.linear2=self.linear(5, 20, activation=self.activation)
        self.linear3=self.linear(20, 40, activation=self.activation)
        self.linear4=self.linear(40, 40, activation=self.activation)
        self.linear5=self.linear(40, 40, activation=self.activation)
        self.linear6=self.linear(40, 20, activation=self.activation)
        self.linear7=self.linear(20, 10, activation=self.activation)
        self.linear8=self.linear(10, 5, activation=self.activation)
        
    def linear(self, in_features, out_features, activation="relu"):
        layers=[nn.Linear(in_features, out_features)]
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Sigmoid())
        net = nn.Sequential(*layers)
        return net
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        out = self.linear1(inputs)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        out = self.linear6(out)
        out = self.linear7(out)
        out = self.linear8(out)
        out = self.regressor(out)
        return out
    

net = PINN(activation = "tanh").to(device)
optimizer = optim.Adam(net.parameters(),lr=0.0005)

num_epochs = 9000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    #
    # PINN LOSS
    #
    x_sampled = X_sample_t[:, 0].unsqueeze(dim=-1).to(device)
    t_sampled = X_sample_t[:, 1].unsqueeze(dim=-1).to(device)
    pinn_loss = physics_informed_loss(x_sampled, t_sampled,net)
    #
    #initial loss
    #
    x_ini = X_ini_t[:, 0].unsqueeze(dim=-1).to(device)
    t_ini = X_ini_t[:, 1].unsqueeze(dim=-1).to(device)
    ini_loss = initial_condition_loss(x_ini, t_ini, net, u_ini_t)
    #
    #boundary loss
    #
    x_bnd = X_bc_t[:, 0].unsqueeze(dim=-1).to(device)
    t_bnd = X_bc_t[:, 1].unsqueeze(dim=-1).to(device)
    bnd_loss = boundary_condition_loss(x_bnd, t_bnd, net, u_bc_t)
    
    loss = pinn_loss + ini_loss + bnd_loss
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        loss_ = loss.item()
        pinn_loss_ = pinn_loss.item()
        ini_loss_ = ini_loss.item()
        bnd_loss = bnd_loss.item()
        print(f' epoch: {epoch:.3e}, loss: {loss_:.3e}, pinn:{ pinn_loss_:.3e}, ini: {ini_loss_:.3e},bnd: {bnd_loss:.3e}')