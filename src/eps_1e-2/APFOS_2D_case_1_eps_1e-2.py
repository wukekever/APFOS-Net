#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os


# In[ ]:


class Net(nn.Module):
    def __init__(self, input_size, hidden_width, hidden_num, output_size):
        super(Net, self).__init__()
        self.layer_in = nn.Linear(input_size, hidden_width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(hidden_num)]
        )
        self.output_layer = nn.Linear(hidden_width, output_size)

    def forward(self, xz):
        output = self.layer_in(xz)
        for i, h_i in enumerate(self.hidden_layers):
            output = self.activation(h_i(output))
        output = self.output_layer(output)
        return output

    def activation(self, o):
        return torch.tanh(o)


# In[ ]:


space_dimension = 2
d = space_dimension


# In[ ]:


input_size = d
hidden_width = 128
hidden_num = 4
output_size = 3


# In[ ]:


torch.set_default_tensor_type("torch.FloatTensor")

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device_ids = [7]

net = Net(input_size, hidden_width, hidden_num, output_size)

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net, device_ids=device_ids)

net.to(device)


# In[ ]:


# param_num = sum(x.numel() for x in net.parameters())
# print('Total number of paramerters in networks is: {}'.format(param_num))


# In[ ]:


# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def Xavier_initi(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


# In[ ]:


def save_param(net, path):
    torch.save(net.state_dict(), path)


def load_param(net, path):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        Xavier_initi(net)
        save_param(net, path="./pkl_uniform_case_1/Xavier_initi_net_params_case_1.pkl")


# In[ ]:


load_param(net, path="./pkl_uniform_case_1/Xavier_initi_net_params_case_1.pkl")
print("Initialization of neural network is finished!")


# In[ ]:


def model(xz):
    model_phi = net(xz)[:, 0:1]
    model_tau = net(xz)[:, 1:2]
    model_sigma = net(xz)[:, 2:3]
    temp = torch.cat((model_phi, model_tau), 1)
    model_temp = torch.cat((temp, model_sigma), 1)
    return model_temp.to(device)


# In[ ]:


# data sample
# data: [0, 1] \times [0, 1]
def generate_sample(data_size_temp):
    X, Z = 1.0, 1.0
    x_test = np.linspace(0, X, data_size_temp).reshape(data_size_temp, 1)
    z_test = np.linspace(0, Z, data_size_temp).reshape(data_size_temp, 1)
    x_test, z_test = np.meshgrid(x_test, z_test)
    X_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], 1)
    Z_test = z_test.reshape(z_test.shape[0] * z_test.shape[1], 1)
    xz_temp = np.concatenate((X_test, Z_test), axis=1)
    sample_temp = torch.tensor(xz_temp).float()
    return sample_temp.to(device)


# data: {0} \times [0, 1]
def generate_sample_bdy_left(boundary_data_size_temp):
    X, Z = 1.0, 1.0
    x_test = np.zeros((boundary_data_size_temp, 1))
    z_test = np.linspace(0, Z, boundary_data_size_temp).reshape(
        boundary_data_size_temp, 1
    )
    xz_temp = np.concatenate((x_test, z_test), axis=1)
    sample_temp = torch.tensor(xz_temp).float()
    return sample_temp.to(device)


# data: {1} \times [0, 1]
def generate_sample_bdy_right(boundary_data_size_temp):
    X, Z = 1.0, 1.0
    x_test = np.ones((boundary_data_size_temp, 1)) * X
    z_test = np.linspace(0, Z, boundary_data_size_temp).reshape(
        boundary_data_size_temp, 1
    )
    xz_temp = np.concatenate((x_test, z_test), axis=1)
    sample_temp = torch.tensor(xz_temp).float()
    return sample_temp.to(device)


# data: [0, 1] \times {0}
def generate_sample_bdy_down(boundary_data_size_temp):
    X, Z = 1.0, 1.0
    x_test = np.linspace(0, X, boundary_data_size_temp).reshape(
        boundary_data_size_temp, 1
    )
    z_test = np.zeros((boundary_data_size_temp, 1))
    xz_temp = np.concatenate((x_test, z_test), axis=1)
    sample_temp = torch.tensor(xz_temp).float()
    return sample_temp.to(device)


# data: [0, 1] \times {1}
def generate_sample_bdy_up(boundary_data_size_temp):
    X, Z = 1.0, 1.0
    x_test = np.linspace(0, X, boundary_data_size_temp).reshape(
        boundary_data_size_temp, 1
    )
    z_test = np.ones((boundary_data_size_temp, 1)) * Z
    xz_temp = np.concatenate((x_test, z_test), axis=1)
    sample_temp = torch.tensor(xz_temp).float()
    return sample_temp.to(device)


# In[ ]:


def b(xz, theta, m, w):
    (x, z) = torch.split(xz, 1, dim=1)
    B_x = m * np.pi * theta * (x**2 - x) * torch.sin(m * np.pi * z)
    B_z = np.pi + theta * (2 * x - 1.0) * torch.cos(m * np.pi * z)
    B_field = torch.cat((B_x, B_z), 1)
    B_norm = torch.norm(B_field, p=2, dim=1, keepdim=True)
    return (B_field / B_norm).to(device)


# In[ ]:


def b_orthogonal(xz, theta, m, w):
    b_temp = b(xz, theta, m, w)
    (b_x, b_z) = torch.split(b_temp, 1, dim=1)
    orthogonal = torch.cat((-b_z, b_x), 1)
    return orthogonal.to(device)


# In[ ]:


def exact_solution(xz, theta, m, w, eps):
    (x, z) = torch.split(xz, 1, dim=1)
    phi_0 = torch.sin(w * (np.pi * x + theta * (x**2 - x) * torch.cos(m * np.pi * z)))
    phi_temp = phi_0 + eps * torch.cos(2 * np.pi * z) * torch.sin(np.pi * x)
    return phi_temp.to(device)


# In[ ]:


def tensor_dot(tensor_x, tensor_y):
    dot_product = tensor_x * tensor_y
    return torch.sum(dot_product, dim=1, keepdim=True).to(device)


# In[ ]:


def f(xz, theta, m, w, eps):
    #      xz.requires_grad = True
    (x, z) = torch.split(xz, 1, dim=1)
    xz = torch.cat((x, z), 1)
    b_case = b(xz, theta, m, w)
    b_orth_case = b_orthogonal(xz, theta, m, w)

    solu = exact_solution(xz, theta, m, w, eps)

    gradient_solu = torch.autograd.grad(
        outputs=solu,
        inputs=xz,
        grad_outputs=torch.ones(solu.shape).to(device),
        create_graph=True,
    )[0]

    operator_para = tensor_dot(gradient_solu, b_case) * b_case
    operator_perp = tensor_dot(gradient_solu, b_orth_case) * b_orth_case

    operator_para_x = torch.autograd.grad(
        outputs=operator_para[:, 0:1],
        inputs=x,
        grad_outputs=torch.ones(operator_para[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]
    operator_para_z = torch.autograd.grad(
        outputs=operator_para[:, 1:2],
        inputs=z,
        grad_outputs=torch.ones(operator_para[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]
    operator_perp_x = torch.autograd.grad(
        outputs=operator_perp[:, 0:1],
        inputs=x,
        grad_outputs=torch.ones(operator_perp[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]
    operator_perp_z = torch.autograd.grad(
        outputs=operator_perp[:, 1:2],
        inputs=z,
        grad_outputs=torch.ones(operator_perp[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]

    divergence_para = operator_para_x + operator_para_z
    divergence_perp = operator_perp_x + operator_perp_z

    f_temp = -divergence_perp - 1.0 / eps * divergence_para
    return f_temp.to(device)


# In[ ]:


def error_function(xz, theta, m, w, eps):
    solu_ex = exact_solution(xz, theta, m, w, eps)
    solu_pred = model(xz)[:, 0:1]
    residual = solu_ex - solu_pred
    relative_error = torch.sum(residual**2) / torch.sum(solu_ex**2)
    return relative_error.to(device)


# In[ ]:


def loss_function(xz, xz_left, xz_right, xz_down, xz_up, theta, m, w, eps):

    (x, z) = torch.split(xz, 1, dim=1)
    xz = torch.cat((x, z), 1)

    model_hat = model(xz)
    phi_hat = model_hat[:, 0:1]
    tau_hat = model_hat[:, 1:2]
    sigma_hat = model_hat[:, 2:3]

    b_case = b(xz, theta, m, w)
    b_orth_case = b_orthogonal(xz, theta, m, w)

    multi_1_hat = tau_hat * b_orth_case  # tau * b_orthogonal
    multi_2_hat = sigma_hat * b_case  # sigma * b

    gradient_phi_hat = torch.autograd.grad(
        outputs=phi_hat,
        inputs=xz,
        grad_outputs=torch.ones(phi_hat.shape).to(device),
        create_graph=True,
    )[0]

    dot_1_hat = tensor_dot(
        gradient_phi_hat, b_orth_case
    )  # inner product of gradient_phi and b_orthogonal
    dot_2_hat = tensor_dot(
        gradient_phi_hat, b_case
    )  # inner product of gradient_phi and b

    multi_1_x_hat = torch.autograd.grad(
        outputs=multi_1_hat[:, 0:1],
        inputs=x,
        grad_outputs=torch.ones(multi_1_hat[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]
    multi_1_z_hat = torch.autograd.grad(
        outputs=multi_1_hat[:, 1:2],
        inputs=z,
        grad_outputs=torch.ones(multi_1_hat[:, 1:2].shape).to(device),
        create_graph=True,
    )[0]
    divergence_1 = multi_1_x_hat + multi_1_z_hat  # divergence of tau * b_orthogonal

    multi_2_x_hat = torch.autograd.grad(
        outputs=multi_2_hat[:, 0:1],
        inputs=x,
        grad_outputs=torch.ones(multi_2_hat[:, 0:1].shape).to(device),
        create_graph=True,
    )[0]
    multi_2_z_hat = torch.autograd.grad(
        outputs=multi_2_hat[:, 1:2],
        inputs=z,
        grad_outputs=torch.ones(multi_2_hat[:, 1:2].shape).to(device),
        create_graph=True,
    )[0]
    divergence_2 = multi_2_x_hat + multi_2_z_hat  # divergence of sigma * b

    part_1 = torch.sum(
        (divergence_1 + divergence_2 + f(xz, theta, m, w, eps)) ** 2
    ) / len(xz)

    part_2 = torch.sum((dot_1_hat - tau_hat) ** 2) / len(xz)

    part_3 = torch.sum((dot_2_hat - eps * sigma_hat) ** 2) / len(xz)

    # Boundary - Gamma_D
    (x_left, z_left) = torch.split(xz_left, 1, dim=1)
    xz_left = torch.cat((x_left, z_left), 1)
    model_left_hat = model(xz_left)
    phi_left_hat = model_left_hat[:, 0:1]

    (x_right, z_right) = torch.split(xz_right, 1, dim=1)
    xz_right = torch.cat((x_right, z_right), 1)
    model_right_hat = model(xz_right)
    phi_right_hat = model_right_hat[:, 0:1]

    part_4 = (
        torch.sum((phi_left_hat - exact_solution(xz_left, theta, m, w, eps)) ** 2)
        / len(xz_left)
        / 2.0
        + torch.sum((phi_right_hat - exact_solution(xz_right, theta, m, w, eps)) ** 2)
        / len(xz_right)
        / 2.0
    )

    # Boundary - Gamma_N
    (x_down, z_down) = torch.split(xz_down, 1, dim=1)
    xz_down = torch.cat((x_down, z_down), 1)
    model_down_hat = model(xz_down)
    phi_down_hat = model_down_hat[:, 0:1]
    tau_down_hat = model_down_hat[:, 1:2]
    sigma_down_hat = model_down_hat[:, 2:3]

    gradient_phi_down_hat = torch.autograd.grad(
        outputs=phi_down_hat,
        inputs=xz_down,
        grad_outputs=torch.ones(phi_down_hat.shape).to(device),
        create_graph=True,
    )[0]

    b_down_case = b(xz_down, theta, m, w)
    b_down_orth_case = b_orthogonal(xz_down, theta, m, w)

    operator_down_para = tensor_dot(gradient_phi_down_hat, b_down_case) * b_down_case
    operator_down_perp = (
        tensor_dot(gradient_phi_down_hat, b_down_orth_case) * b_down_orth_case
    )

    (x_up, z_up) = torch.split(xz_up, 1, dim=1)
    xz_up = torch.cat((x_up, z_up), 1)
    model_up_hat = model(xz_up)
    phi_up_hat = model_up_hat[:, 0:1]
    tau_up_hat = model_up_hat[:, 1:2]
    sigma_up_hat = model_up_hat[:, 2:3]

    gradient_phi_up_hat = torch.autograd.grad(
        outputs=phi_up_hat,
        inputs=xz_up,
        grad_outputs=torch.ones(phi_up_hat.shape).to(device),
        create_graph=True,
    )[0]

    b_up_case = b(xz_up, theta, m, w)
    b_up_orth_case = b_orthogonal(xz_up, theta, m, w)

    operator_up_para = tensor_dot(gradient_phi_up_hat, b_up_case) * b_up_case
    operator_up_perp = tensor_dot(gradient_phi_up_hat, b_up_orth_case) * b_up_orth_case

    part_5 = (
        torch.sum(
            (
                operator_down_para[:, 1:2] * (-1.0)
                + eps * operator_down_perp[:, 1:2] * (-1.0)
            )
            ** 2
        )
        / len(xz_down)
        / 2.0
        + torch.sum(
            (operator_up_para[:, 1:2] * (1.0) + eps * operator_up_perp[:, 1:2] * (1.0))
            ** 2
        )
        / len(xz_up)
        / 2.0
    )

    part_6 = (
        torch.sum(
            (
                b_down_orth_case[:, 1:2] * (-1.0) * tau_down_hat
                + b_down_case[:, 1:2] * (-1.0) * sigma_down_hat
            )
            ** 2
        )
        / len(xz_down)
        / 2.0
        + torch.sum(
            (
                b_up_orth_case[:, 1:2] * (1.0) * tau_up_hat
                + b_down_case[:, 1:2] * (1.0) * sigma_up_hat
            )
            ** 2
        )
        / len(xz_up)
        / 2.0
    )

    beta_D, beta_N = 1.0, 1.0
    summation = part_1 + part_2 + part_3 + beta_D * part_4 + beta_N * (part_5 + part_6)

    return summation


# In[ ]:


# set optimizer and learning rate decay
optimizer = optim.Adam(net.parameters())
scheduler = lr_scheduler.StepLR(
    optimizer, 5000, 0.9
)  # every 5000 epoch, learning rate * 0.9


# In[ ]:


Iter = 100000
theta, m, w, eps = 0, 1, 2, 0.01
xz = generate_sample(data_size_temp=100)
xz_left = generate_sample_bdy_left(boundary_data_size_temp=100)
xz_right = generate_sample_bdy_right(boundary_data_size_temp=100)
xz_down = generate_sample_bdy_down(boundary_data_size_temp=100)
xz_up = generate_sample_bdy_up(boundary_data_size_temp=100)
xz.requires_grad = True
xz_left.requires_grad = True
xz_right.requires_grad = True
xz_down.requires_grad = True
xz_up.requires_grad = True

loss_record = np.zeros(Iter)
error_record = np.zeros(Iter)
time_start = time.time()
for it in range(Iter):
    optimizer.zero_grad()
    loss = loss_function(xz, xz_left, xz_right, xz_down, xz_up, theta, m, w, eps)
    error = error_function(xz, theta, m, w, eps)
    loss_record[it] = float(loss)
    error_record[it] = float(error)
    if it % 1 == 0:
        print("")
        print(
            "[Iteration step: {}/{} - Loss: {:.2e} - Error: {:.2e}]".format(
                it + 1, Iter, loss.detach(), error.detach()
            )
        )
        #         print('[Learning rate: {:.2e}]'.format(optimizer.param_groups[0]['lr']))
        print("")
    path = "./pkl_uniform_case_1/net_params_case_1_{}.pkl".format(it)
    if it % 1000 == 0:
        save_param(net, path)

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()

save_param(net, path="./pkl_uniform_case_1/net_params_case_1.pkl")
np.save("loss_uniform_case_1.npy", loss_record)
np.save("error_uniform_case_1.npy", error_record)
time_end = time.time()
print("Total time for training is: ", time_end - time_start, "seconds")


# In[ ]:


# In[ ]:
