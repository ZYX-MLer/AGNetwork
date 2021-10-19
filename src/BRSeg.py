import os
import sys
import cv2

sys.path.append("../")
import tool.ToolBox as tBox
import tool.scheduler as scheduler
import tool.PytorchUtils as PytorchUtils
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy
import time
import copy
import socket

import torch
import torchvision
import BRDataTransform
import BRDataSet
import BRNet


from collections import Counter
from collections import Iterable, Iterator

import math
import scipy.ndimage
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib



class BRSeg:

    def get_net(self):
        if hasattr(self.net, "module"):
            return self.net.module
        else:
            return self.net

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __init__(self, param):

        torch.manual_seed(0)
        numpy.random.seed(0)

        if param.step == "train":
            self.init_train(param)
        elif param.step == "test":
            self.init_test(param)

    def init_test(self, param):
        self.param = param
        self.net = BRNet.BRNet(param)
        self.device = torch.device(0 if torch.cuda.is_available() and -1 not in self.param.all_gpu_idx else "cpu")

        ##data
        transform_test = torchvision.transforms.Compose([
            BRDataTransform.PadAndCrop(self.param.shape_global, random=False),
            # BRDataTransform.Subsample_global(self.param.rate_global)
        ])

        test_loader = torch.utils.data.DataLoader(BRDataSet.BRDataSet(param, self.param.test_path, transform=transform_test, name="test"),
                                                  batch_size=self.param.num_batch,
                                                  shuffle=True, num_workers=self.param.num_worker)

        self.dataloader = {
            "eval_nor": test_loader,
            "eval_ema": test_loader
        }

        self.param.num_epoch = 1

        self.epoch_begin = 0

        if self.device.type == "cpu":
            checkpoint = torch.load(self.param.model_path, map_location='cpu')
        else:
            checkpoint = torch.load(self.param.model_path)
        #
        self.net.load_state_dict(checkpoint['model_state_dict'])

        if self.device.type != "cpu":
            self.net = torch.nn.DataParallel(self.net, device_ids=[i for i in range(len(self.param.all_gpu_idx))])

        self.net.to(self.device)
        self.ema = PytorchUtils.ExponentialMovingAverage(self.net.named_parameters(), self.param.ema_rate, False)


    def init_train(self, param):
        self.param = param
        self.param.dice_weight = 0
        self.net = BRNet.BRNet(param)
        self.device = torch.device(0 if torch.cuda.is_available() and -1 not in self.param.all_gpu_idx else "cpu")
        self.param.self_flip = BRDataTransform.SelfFlipEx(self.param.kernel_size_head_local, self.param.opt_stride, self.param.shape_local, self.param.shape_brain, num_label=self.param.num_class_local)

        ##data
        transform_train_l = torchvision.transforms.Compose([
            # BRDataTransform.RandomRot(head_size=self.param.shape_global, brain_size=self.param.shape_brain, angle=self.param.angle),
            BRDataTransform.NonlinearTransformation(),
            BRDataTransform.PadAndCrop(self.param.shape_global, random=True),
            # BRDataTransform.Subsample_global(self.param.rate_global)
        ])

        transform_train_u = torchvision.transforms.Compose([
            # BRDataTransform.RandomRot(head_size=self.param.shape_global, brain_size=self.param.shape_brain, angle=self.param.angle),
            BRDataTransform.PadAndCrop(self.param.shape_global, random=True),
            self.param.self_flip
        ])

        transform_test = torchvision.transforms.Compose([
            BRDataTransform.PadAndCrop(self.param.shape_global, random=False),
            # BRDataTransform.Subsample_global(self.param.rate_global)
        ])

        all_key = [os.path.split(v)[1].split(".")[0] for v in tBox.FindAllFile(self.param.train_l_path, [], ["gz"]) if v.find("_glm.nii.gz") == -1]
        all_key = list(set(all_key))
        all_key.sort()
        numpy.random.shuffle(all_key)
        num_key = len(all_key)
        all_train_key_ignore = all_key[int(num_key * 0.9):]
        all_eval_key_ignore = all_key[:int(num_key * 0.9)]
        all_train_key_ignore = []
        all_eval_key_ignore = []


        train_l_loader = torch.utils.data.DataLoader(BRDataSet.BRDataSet(param, self.param.train_l_path, ignore_key=all_train_key_ignore,
                                                                         transform=transform_train_l, name="train_label", times=5 if self.param.num_class_local == 135 else 2),
                                                          batch_size=self.param.num_batch,
                                                          num_workers=self.param.num_worker,
                                                          drop_last=True, shuffle=True)


        train_eval_loader = torch.utils.data.DataLoader(BRDataSet.BRDataSet(param, self.param.train_l_path, transform=transform_test, name="train_eval"),
                                                          batch_size=self.param.num_batch, num_workers=self.param.num_worker,
                                                          drop_last=True, shuffle=True)
        self.dataloader = {}
        self.dataloader["train"] = train_l_loader
        self.dataloader["train_eval"] = train_eval_loader
        for key in self.param.test_path:
            test_loader = torch.utils.data.DataLoader(BRDataSet.BRDataSet(param, self.param.test_path[key], transform=transform_test, name="test"),
                                                      batch_size=self.param.num_batch, num_workers=self.param.num_worker,
                                                      drop_last=False, shuffle=True)
            self.dataloader[key + "_eval"] = test_loader
            self.dataloader[key + "_eval_ema"] = test_loader



        train_u_loader = torch.utils.data.DataLoader(BRDataSet.BRDataSet(param, self.param.train_u_path, ignore_key=all_train_key_ignore,
                                                                         transform=transform_train_u, name="train_unlabel"),
                                                          batch_size=max(1, self.param.num_batch // 2),
                                                          num_workers=self.param.num_worker,
                                                          drop_last=True,
                                                          shuffle=True)
        self.dataloader_endless = BRDataSet.BRDataLoaderEndless(train_u_loader)

        # if self.param.max_class_sample < 1:
        #     self.dataloader_endless = BRDataSet.BRDataLoaderEndless(train_u_loader)
        #     self.dataloader = {
        #         "train": train_l_loader,
        #         # "train_eval_out": train_eval_loader_out,
        #         "train_eval_in": train_eval_loader_in,
        #         "eval_nor": test_loader,
        #         "eval_ema": test_loader
        #     }
        # else:
        #     self.dataloader = {
        #         "train": train_l_loader,
        #         "eval_nor": train_eval_loader_in
        #     }


        self.param.num_iter = len(self.dataloader["train"]) * self.param.num_epoch

        multiplier = int(self.param.num_epoch * self.param.learning_rate_warmup)
        self.param.pre_epoch = multiplier
        weight_decay = PytorchUtils.PolynomialDecay(learning_rate=self.param.lambda_rate_begin,
                                                    decay_steps=len(self.dataloader["train"]) * (multiplier + self.param.num_epoch),
                                                    end_learning_rate=self.param.lambda_rate_end,
                                                    power=self.param.lambda_rate_param)


        param_group_act = [v for name, v in self.net.named_parameters() if name.find("act") != -1 or name.find("mlp.2") != -1]
        param_group_linear = [v for name, v in self.net.named_parameters() if name.find("act") == -1 and name.find("mlp.2") == -1]

        self.optimizer = PytorchUtils.SGDEx([
            {"params": param_group_act, 'weight_decay_rate': 0, 'weight_decay_type': 2, 'lr_rate': 1},
            {"params": param_group_linear, 'weight_decay_rate': 1, 'weight_decay_type': 2, 'lr_rate': 1}
        ],
            weight_decay, lr=self.param.learning_rate_begin / multiplier,
            momentum=0.9, weight_decay=self.param.lambda_rate_end)


        after_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.param.learning_rate_param)
        self.scheduler_warmup = scheduler.GradualWarmupScheduler(self.optimizer, multiplier=multiplier,
                                                                 total_epoch=multiplier,
                                                                 after_scheduler=after_scheduler)

        # for i in range(1000):
        #     lr = self.scheduler_warmup.get_lr()
        #
        #     for g_n in range(len(self.scheduler_warmup.optimizer.param_groups)):
        #         self.scheduler_warmup.optimizer.param_groups[g_n]["lr"] *= 0.5
        #
        #     lr2 = self.scheduler_warmup.get_lr()
        #
        #     self.scheduler_warmup.step()
        #     print(lr, lr2)

        self.param.num_epoch += multiplier
        self.param.original_num_epoch = self.param.num_epoch

        self.clip_grad_th = PytorchUtils.PolynomialDecay(learning_rate=self.param.clip_grad_th_begin,
                                                         decay_steps=self.param.num_epoch,
                                                         end_learning_rate=self.param.clip_grad_th_end, power=1)

        self.dropout_rate = PytorchUtils.PolynomialDecay(learning_rate=self.param.dropout_rate / self.param.num_epoch,
                                                         decay_steps=self.param.num_epoch,
                                                         end_learning_rate=self.param.dropout_rate, power=1)


        self.epoch_begin = 0
        self.performance_best = {key: {"dice_local": 0} for key in self.dataloader}
        # if self.device.type == "cpu":
        #     checkpoint = torch.load(self.param.model_path, map_location='cpu')
        # else:
        #     checkpoint = torch.load(self.param.model_path)
        # #
        # self.net.load_state_dict(checkpoint['model_state_dict'])


        if self.device.type != "cpu":
            self.net = torch.nn.DataParallel(self.net, device_ids=[i for i in range(len(self.param.all_gpu_idx))])

        self.net.to(self.device)
        self.ema = PytorchUtils.ExponentialMovingAverage(self.net.named_parameters(), self.param.ema_rate, False)

    def save_model(self, save_name):

        save_name = os.path.join(self.param.save_path, save_name)

        while True:
            try:
                torch.save({
                    "performance_best": self.performance_best,
                    "model_state_dict": self.net.module.state_dict() if torch.cuda.is_available() else self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "ema": self.ema.shadow_params if hasattr(self, "ema") else 0,
                    "lr": self.scheduler_warmup
                }, save_name)
            except Exception as e:
                print(e)
                input("press any key to try again")
                continue
            else:
                break

    def every_eopch(self, run_info_file):

        print(
              "\033[31m" + "t_l_dice :" + "\033[0m" + str({key: self.performance_best[key]["dice_local"] for key in self.performance_best})
              )

        mesg = "dice:" + str({key: self.performance_best[key]["dice_local"] for key in self.performance_best}) + " " + \
               "dice_weight:" + str(0 if isinstance(self.param.dice_weight, int) is True else self.param.dice_weight.detach().cpu().numpy())

        while True:
            try:

                run_info_file.write(mesg)
                run_info_file.write("\n")
                run_info_file.flush()
            except Exception as e:
                print(e)
                input("press any key to try again")
                continue
            else:
                break

    def train_model(self):
        since = time.time()

        run_info_path = os.path.join(self.param.save_path, "runInfo.txt")
        run_info_file = open(run_info_path, "a")

        print("train_iter:", len(self.dataloader["train"]))

        e_n = self.epoch_begin
        writer = SummaryWriter(os.path.join(self.param.save_path, "tensorboard"))

        if e_n > self.param.num_epoch - 1:
            self.param.num_epoch += int(input("additional epoch:"))

        all_sample = {"train_eval_in": [], "train_eval_out": [], "eval_nor": [], "eval_ema": []}
        all_sample_eval_train = []
        time_epoch = 0
        while e_n < self.param.num_epoch:

            self.every_eopch(run_info_file)

            t = self.param.original_num_epoch / 2
            rampup = math.exp(-5 * math.pow(1 - min(1, e_n / t), 2))
            rampup = 0.5 - math.cos(min(math.pi, 2 * math.pi * max(0, e_n) / self.param.original_num_epoch)) / 2
            rampup = rampup * self.param.semi_rate

            time_begin = time.time()

            for phase in self.dataloader.keys():
                if phase == "train":

                    if e_n == 50:
                        for g_n in range(len(self.scheduler_warmup.optimizer.param_groups)):
                            self.scheduler_warmup.optimizer.param_groups[g_n]["lr"] *= self.param.lr_decay
                    elif e_n == 75:
                        for g_n in range(len(self.scheduler_warmup.optimizer.param_groups)):
                            self.scheduler_warmup.optimizer.param_groups[g_n]["lr"] *= self.param.lr_decay
                    elif e_n == 100:
                        for g_n in range(len(self.scheduler_warmup.optimizer.param_groups)):
                            self.scheduler_warmup.optimizer.param_groups[g_n]["lr"] *= self.param.lr_decay

                    self.net.train()
                    # writer.add_scalars("dice_local", {"train": self.performance_best_train["dice_local"], "eval": self.performance_best_eval["dice_local"]}, e_n)

                    grad_sum = 0
                    # continue
                    # non_zero_sum = 0
                    # data_sum_act = 0
                elif phase == "train_eval":
                    if e_n % 5 != 0 or e_n < self.param.original_num_epoch:
                        continue
                    self.net.eval()
                elif phase.find("eval_ema") != -1:
                    if e_n < self.param.pre_epoch * 3:
                        continue
                    model_backup = copy.deepcopy(self.net.state_dict())
                    self.ema.copy_to(self.net.named_parameters())
                    self.net.eval()
                elif phase.find("eval") != -1:
                    self.net.eval()

                all_loss = []
                all_statistic = []
                # all_idx = []
                #
                # mAP_epoch = 0.
                # loss_dict_sum = {}
                # iter_in_batch = 0
                num_iter_in_batch = max(1 - e_n, 1)
                # num_iter = len(self.dataloader[phase])

                if phase in all_sample and len(all_sample[phase]) == len(self.dataloader[phase]):
                    loader = all_sample[phase]
                else:
                    loader = self.dataloader[phase]

                for s_n, sample in enumerate(loader, 0):

                    if phase in all_sample and len(all_sample[phase]) != len(self.dataloader[phase]):
                        all_sample[phase].append(sample)

                    time_iter = time.clock()

                    # sample_device["data_local"] = sample["data_local"].to(self.device)
                    # sample_device["seg_local"] = sample["seg_local"].to(self.device)
                    # sample_device["roi_local"] = sample["roi_local"].numpy()

                    # sample_device["data_global"] = sample["data_global"].to(self.device)
                    # sample_device["seg_global"] = sample["seg_global"].to(self.device)
                    # sample_device["roi_global"] = sample["roi_global"].numpy()

                    with torch.set_grad_enabled(phase == "train"):

                        # try:


                        mesg = "\r" + phase + ' per: {:.2%}'.format(s_n / len(self.dataloader[phase]))

                        if phase == "train":
                            sample["loss"] = self.net(sample, e_n=e_n, dropout_rate=self.dropout_rate.decay, type="label")

                            self.param.dice_weight = self.param.dice_weight * 0.99 + sample["loss"]["dice_weight"].mean(dim=0) * 0.01
                            loss_opt = torch.mean(sample["loss"]["loss_opt"])
                            loss_dict = {key: torch.mean(sample["loss"]["loss_dict"][key]).item() for key in sample["loss"]["loss_dict"]}
                            loss_opt.backward()

                            num_unlabel = 1 if self.param.semi_rate != 0 and rampup != 0 else 0
                            num_unlabel = num_unlabel if self.param.debug is False else 2
                            for i in range(num_unlabel):
                                sample_unlabel = self.dataloader_endless.get()

                                result_global, result_local = self.net(sample_unlabel, e_n=e_n, dropout_rate=self.dropout_rate.decay, type="unlabel_0")
                                sample_unlabel["counter_global"] = result_global.detach().flip(dims=[0])
                                sample_unlabel["result_global"] = result_global
                                sample_unlabel["counter_local"] = result_local.detach().flip(dims=[0])
                                sample_unlabel["result_local"] = result_local
                                loss_opt_u, loss_dict_u = self.net(sample_unlabel, e_n=e_n, dropout_rate=self.dropout_rate.decay, type="unlabel_1")

                                loss_dict_u = {key: torch.mean(loss_dict_u[key]).item() for key in loss_dict_u}
                                loss_dict.update(loss_dict_u)
                                loss_opt_u = torch.mean(loss_opt_u)
                                loss_opt_u = loss_opt_u / num_unlabel * rampup
                                loss_opt_u.backward()

                            all_grad = [(name, torch.norm(t.grad, p=1)) for name, t in self.net.named_parameters() if hasattr(t.grad, "dim")]
                            grad = sum([v[1] for v in all_grad])
                            grad_sum = grad_sum + grad
                            total_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_th.decay)

                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self.ema.update(self.net.named_parameters())

                            all_loss = {k: 0 for k in loss_dict} if len(all_loss) == 0 else all_loss
                            all_loss = {k: all_loss[k] + loss_dict[k] for k in loss_dict}
                            # shape = sample_device["brain"].shape[-3:]
                            # mesg = mesg + " shape:" + "{:3} {:3} {:3}".format(shape[0], shape[1], shape[2])
                            mesg = mesg + " grad:" + str(round(grad.item(), 1))
                            mesg = mesg + " rampup:" + str(round(rampup, 3))
                            mesg = mesg + " total_norm:" + str(round(total_norm.item(), 1))
                            mesg = mesg + " dice:" + str({key: round(float(loss_dict[key]), 3) for key in loss_dict})
                            # mesg = mesg + " roi_c:" + str(sample["loss"]["roi_c"].cpu().numpy())
                            mesg = mesg + "||"
                        else:

                            loss_dict = self.net(sample, e_n=e_n, dropout_rate=self.dropout_rate.decay, type="eval")

                            loss_dict = {key: list(torch.cat(loss_dict[key], dim=0).detach().cpu().numpy()) for key in loss_dict}
                            all_statistic = {k: [] for k in loss_dict} if len(all_statistic) == 0 else all_statistic

                            all_statistic = {k: all_statistic[k] + loss_dict[k] for k in loss_dict}



                        print(mesg, end="", flush=True)
                        # if s_n > 3:
                        #     break

                if phase == "train":

                    mesg = "\r" + \
                           time.strftime('%H:%M:%S', time.localtime(time.time())) + \
                           ' train {}/{}'.format(e_n, self.param.num_epoch - 1) + \
                           " remain:" + str(round((self.param.num_epoch - e_n) * time_epoch / 3600, 2)) + "h" + \
                           " lr: {:.5f} dr: {:.3f} grad: {:.0f} rampup: {:.5f}".format(
                               self.scheduler_warmup.get_lr()[0],
                               # self.optimizer.lambda_decay.decay,
                               # self.clip_grad_th.decay,
                               self.dropout_rate.decay,
                               grad_sum / len(self.dataloader[phase]),
                               rampup
                           ) + "|||\n\033[31m" + str(
                        {k: round(all_loss[k] * num_iter_in_batch / len(self.dataloader[phase]), 3) for k in all_loss}) + "\033[0m"

                    self.scheduler_warmup.step()
                    self.clip_grad_th()
                    self.dropout_rate()
                    # self.get_net().step()

                    # writer.add_scalars(phase + "_loss", {str(k): all_loss[k] / len(self.dataloader[phase]) for k in all_loss}, e_n)
                    #
                    # writer.add_scalar(phase + "_grad", grad_sum / len(self.dataloader[phase]), e_n)
                else:
                    performance = {k: round(sum(all_statistic[k]) / len(all_statistic[k]) * 100, 2) for k in all_statistic}

                    compare_key = self.get_net().compare_key
                    e2x = (sum([v for v in all_statistic[compare_key]]) / len(all_statistic[compare_key])) ** 2
                    ex2 = sum([v ** 2 for v in all_statistic[compare_key]]) / len(all_statistic[compare_key])
                    performance["dice_std"] = (ex2 - e2x) ** 0.5

                    if self.performance_best[phase][compare_key] < performance[compare_key]:
                        self.performance_best[phase] = performance
                        if phase.find("train") == -1:
                            self.save_model(save_name=str(e_n) + "_" + phase + "_" + str(performance[compare_key]) + "_best.tar")

                    max_val = max(all_statistic["max_dice"])
                    max_idx = all_statistic["max_dice_idx"][all_statistic["max_dice"].index(max_val)]
                    min_val = min(all_statistic["min_dice"])
                    min_idx = all_statistic["min_dice_idx"][all_statistic["min_dice"].index(min_val)]

                    performance.pop("max_dice_idx")
                    performance.pop("min_dice_idx")

                    phase_show = phase.replace("train_eval", "t_e").replace("eval_nor", "e_n").replace("eval_ema", "e_e")
                    mesg = "\r" + time.strftime('%H:%M:%S', time.localtime(time.time())) + " {:4}".format(phase_show) + str({key: str(round(performance[key], 3)) for key in performance }) + \
                           " max_dice/idx:" + str(round(max_val, 3)) + "/" + str(max_idx) + " min_dice/idx:" + str(round(min_val, 3)) + "/" + str(min_idx)

                    # all_scale[phase] = loss_metric

                print(mesg, flush=True)
                v = mesg.replace("\r", "")

                while True:
                    try:
                        run_info_file.write(v)
                        run_info_file.write("\n")
                        run_info_file.flush()
                    except Exception as e:
                        print(e)
                        input("press any key to try again")
                        continue
                    else:
                        break

                # deep copy the model
                if phase == 'train' and e_n % 2 == 0:
                    hostName = socket.gethostname()
                    save_name = os.path.join(self.param.save_path, "cpk.tar")
                    self.save_model("cpk.tar")

                if phase.find("eval_ema") != -1:
                    self.net.load_state_dict(model_backup)



            time_epoch = (time.time() - time_begin)

            if e_n + 1 == self.param.num_epoch:
                self.param.num_epoch += 100
                print("----------------num_epoch:", self.param.num_epoch, "--------------")
            else:
                e_n += 1

        print('performance_best: {:4f}'.format(self.performance_best))

    def nor_att(self, att, rate, slice_idx):
        att = numpy.squeeze(att)

        att_resize = scipy.ndimage.zoom(att, rate, order=2)

        att_slice = att_resize[slice_idx]

        max_value = numpy.max(att_slice)
        att_slice[att_slice == 0] = max_value
        min_value = numpy.min(att_slice)

        att_slice = (att_slice - min_value) / (max_value - min_value) * 255
        att_slice = att_slice.astype(numpy.uint8)

        att_slice_color = cv2.applyColorMap(att_slice, cv2.COLORMAP_JET)
        # att_slice_color = cv2.cvtColor(att_slice_color, cv2.COLOR_BGR2RGB)

        return att_slice, att_slice_color, min_value, max_value

    def test_model(self):
        since = time.time()
        time_epoch = 0


        for i in range(2):


            for phase in self.dataloader.keys():

                if phase == "train_eval_in":
                    self.net.eval()
                elif phase == "train_eval_out":
                    self.net.eval()
                elif phase == "eval_nor":
                    self.net.eval()
                elif phase == "eval_ema":
                    self.net.eval()

                all_loss = []
                all_statistic = []

                sample_iter = iter(self.dataloader[phase])
                all_time_spend = 0
                for s_n in range(len(self.dataloader[phase])):


                    with torch.set_grad_enabled(False):
                        time_begin = time.time()
                        sample = next(sample_iter)
                        loss_dict = self.net(sample, e_n=0, dropout_rate=0, type="eval")
                        t = time.time() - time_begin
                        all_time_spend += t

                        loss_dict = {key: list(torch.cat(loss_dict[key], dim=0).detach().cpu().numpy()) for key in loss_dict}
                        all_statistic = {k: [] for k in loss_dict} if len(all_statistic) == 0 else all_statistic
                        all_statistic = {k: all_statistic[k] + loss_dict[k] for k in loss_dict}

                    mesg = "\r" + phase + ' per: {:.2%}'.format(s_n / len(self.dataloader[phase])) + " speed:" + str(round(t, 3))
                    print(mesg, end="", flush=True)


                performance = {k: round(sum(all_statistic[k]) / len(all_statistic[k]) * 100, 2) for k in all_statistic}

                phase_show = phase.replace("train_eval", "t_e").replace("eval_nor", "e_n").replace("eval_ema", "e_e")
                mesg = "\r" + time.strftime('%H:%M:%S', time.localtime(time.time())) + " {:4}".format(phase_show) + str(
                    {key: str(round(performance[key], 3)) for key in performance}) + " speed:" + str(round(all_time_spend / len(self.dataloader[phase]), 3))

                print(mesg, flush=True)

        # print('performance_best_train: {:4f}'.format(self.performance_best_train), 'performance_best_eval: {:4f}'.format(self.performance_best_eval))