import torch

class LR_Wrapper:
    
    def __init__(self):
        pass
    
    
class Regularization(torch.nn.Module):
    def __init__(self, model, decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()

        self.model = model
        self.weight_decay = decay

        self.p = p
        self.weight_list = self.get_weight(model)
        self.idx_epoch = 0
        # self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        decay = self.weight_decay()
        reg_loss = self.regularization_loss(self.weight_list, decay, p=self.p)
        # print("reg", decay)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = {name: t.clone().detach()
                              for name, t in parameters}

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """

        one_minus_decay = 1. - self.decay
        with torch.no_grad():
            all_param = list(parameters)
            for name, t in all_param:
                # t1 = self.shadow_params[name]
                # t2 = t
                self.shadow_params[name].sub_(one_minus_decay * (self.shadow_params[name] - t))
                # self.shadow_params[name].sub_(t * 0.5)
                # t3 = self.shadow_params[name]
                # a = 0


    def copy_to(self, parameters):
        """
        Copies current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        self.key = 0
        for name, t in parameters:
            # if self.key == 0:
            #     self.key = name
            if name not in self.shadow_params and name.find("module") == -1:
                name = "module." + name
            t.data.copy_(self.shadow_params[name].data)


class PolynomialDecay:
    def __init__(self, learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0, begin_step = 0):
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.idx = 0
        self.decay = learning_rate
        self.begin_step = begin_step

    def __call__(self):
        self.idx += 1
        global_step = min(max(self.idx - self.begin_step, 0), self.decay_steps)
        self.decay = (self.learning_rate - self.end_learning_rate) *\
                                (1 - global_step / self.decay_steps) ** (self.power) + self.end_learning_rate

        return self.decay

class WarmupDecay:
    def __init__(self, warmup_step, learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0, begin_step = 0):

        self.decay_begin = PolynomialDecay(learning_rate / warmup_step, warmup_step, learning_rate, power, 0)
        self.decay_end = PolynomialDecay(learning_rate, decay_steps, end_learning_rate, power, 0)
        self.warmup_step = warmup_step
        self.decay = self.decay_begin.decay

    def __call__(self):
        return self.decay

    def step(self):
        if self.decay_begin.idx < self.warmup_step:
            self.decay = self.decay_begin()
        else:
            self.decay = self.decay_end()
# class StepDecay:
#     def __init__(self, learning_rate, decay_steps, end_learning_rate, num_step):
#
#         self.all_learning_rate = [learning_rate - v * (learning_rate - end_learning_rate) / num_step for v in range(num_step)]
#         self.decay_steps = decay_steps
#         self.step = decay_steps // num_step
#         self.idx = 0
#
#     def __call__(self):
#         global_step = min(self.idx, self.decay_steps)
#
#         self.decay = self.all_learning_rate[global_step // self.step]
#         self.idx += 1
#         return self.decay

def clip_grad_norm_layer_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    total_norm = 0
    num_clip = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)

        clip_coef = max_norm / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)
            # t = p.grad.data.norm(norm_type)
            num_clip += 1

    return num_clip


from torch.optim.optimizer import Optimizer, required
class SGDEx(torch.optim.SGD):
    def __init__(self, params, lambda_decay, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(SGDEx, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.lambda_decay = lambda_decay
        
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.lambda_decay()
        for group in self.param_groups:
            weight_decay = self.lambda_decay.decay * group['weight_decay_rate']
            weight_decay_type = group['weight_decay_type']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    if weight_decay_type == 2:
                        d_p.add_(weight_decay, p.data)
                    elif weight_decay_type == 1:
                        r = torch.zeros_like(p.data)
                        r[p.data < 0] = -1
                        r[p.data > 0] = 1
                        d_p.add_(weight_decay, r)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf


                p.data.add_(-group['lr'] * group["lr_rate"], d_p)
                pass

        return loss


# class LocallyConnected3d(nn.Module):
#     def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
#         super(LocallyConnected2d, self).__init__()
#         output_size = _pair(output_size)
#         self.weight = nn.Parameter(
#             torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
#         )
#         if bias:
#             self.bias = nn.Parameter(
#                 torch.randn(1, out_channels, output_size[0], output_size[1])
#             )
#         else:
#             self.register_parameter('bias', None)
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
# 
#     def forward(self, x):
#         _, c, h, w = x.size()
#         kh, kw = self.kernel_size
#         dh, dw = self.stride
#         x = x.unfold(2, kh, dh).unfold(3, kw, dw)
#         x = x.contiguous().view(*x.size()[:-2], -1)
#         # Sum in in_channel and kernel_size dims
#         out = (x.unsqueeze(1) * self.weight).sum([2, -1])
#         if self.bias is not None:
#             out += self.bias
#         return out