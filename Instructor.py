import time
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from MotionModel import MotionModel
from AppModel import AppModel
from PfBatchHandler import PfBatchHandler
from BatchMaker import BaseBatchMaker
from target_traj_func import make_trajs_mm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    WARNING2 = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'

optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD  # default lr=0.1
}

modelmode = {
    "attention": "attention"}

inferencemode = {
    "eval" : "eval",
    "paint": "paint"}

class Instructor:
    def __init__(self, opt, test_batch_maker:BaseBatchMaker):
        self.time_str_start = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
        self.opt = opt
        self.test_batch_maker = None
        self.testset          = None
        self.device = torch.device(self.opt.device_str) if self.opt.device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_fun = print
        if self.opt.make_new_trajs:
            return
        self.get_models_from_opt()
        nof_sets_to_take = self.opt.nof_batches_per_epoch * self.opt.batch_size
        self.test_batch_maker = test_batch_maker
        self.testset = self.test_batch_maker.get_sets(nof_sets_to_take)
        self.model = self.model.to(self.device)
        self._print_args()

    def load_active_nns(self,model, state_dict):
        if not self.opt.skip_nn3:
            model.nn3.load_state_dict(state_dict['nn3_sd'])

    def get_models_from_opt(self):
        attention_network = modelmode[self.opt.model_mode] in {"attention"}
        if attention_network:
            self.model = AppModel(opt=self.opt,  sensor_params=self.opt.sensor_params, mm_params=self.opt.mm_params, device=self.device)
            checkpoint_att = None
            if self.opt.att_load_checkpoint:
                if self.opt.attention_checkpoint:
                    att_load_model_path = self.opt.proj2ckpnts_load_path + modelmode["attention"] + '/{:s}'.format(self.opt.attention_checkpoint)
                    checkpoint_att = torch.load(att_load_model_path, map_location='cpu')
                    self.load_active_nns(self.model, checkpoint_att)
                    self.print_fun('attention_checkpoint {:s} has been loaded: ' + att_load_model_path)
                else:
                    exit('attention_checkpoint {:s} doesnt exist ')

        self.ckpnt = checkpoint_att
        self.pfbh = PfBatchHandler(model=self.model, opt=self.opt)
        self.criterion = self.pfbh.get_batch_loss
        return

    def string_update_flugs(self):
        out_str = 'simulation arguments:\n' + '\n'.join(
            ['>>> {0}: {1}'.format(arg, getattr(self.opt, arg)) for arg in [key for key, value in vars(self.opt).items()
                        if 'att' not in key.lower() and 'class'  not in key.lower() and 'debug'  not in key.lower() and 'thread' not in key.lower() and 'inference' not in key.lower()]])
        out_str +='\n'
        return out_str

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.info = 'n_trainable_params: {0}, n_nontrainable_params: {1}\n'.format(n_trainable_params, n_nontrainable_params)
        self.info += self.string_update_flugs()
        if self.device.type == 'cuda':
            self.print_fun('cuda memory allocated:', torch.cuda.memory_allocated(self.device.index))
        self.print_fun(self.info)

    def _evaluation(self, epoch, batch_maker, val_dataloader, criterion, paint_batch, time_to_stop=0):
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()

        self.model.eval()
        val_loss, val_dice, n_total, n_batch = 0, 0, 0, len(val_dataloader)
        val_loss_ts, val_dice_ts = 0,0
        total_nof_lost_trajs = 0
        nn3_time_per_batch_single_step, atrapp_time_per_batch_single_step, meas_time_per_batch_single_step = 0,0,0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(val_dataloader):
                input_target_cpu = batch_maker.make_batch_function(sample_batched, self.opt.true_sensor_model,self.opt.make_batch_device_str)
                inputs_cpu, target_cpu = input_target_cpu[0], input_target_cpu[1]
                curr_batch_size, curr_nof_steps = inputs_cpu.shape[0], inputs_cpu.shape[1]
                inputs = inputs_cpu.to(self.device)
                target = target_cpu.to(self.device)
                if modelmode[self.opt.model_mode] in {"attention"}:
                    loss_b_ts, ospa_batch_b_ts, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size_with_grad = criterion(z=inputs,x=target)
                    assert actual_batch_size_with_grad==curr_batch_size
                    lost_traj_mask = torch.sum(lost_targs_mask, axis=1) >= 1
                    nof_lost_trajs = torch.sum(lost_traj_mask >= 1).detach().cpu().numpy()
                    loss = torch.sum(loss_b_ts) / curr_nof_steps/ curr_batch_size
                    ospa_batch = torch.sum(-ospa_batch_b_ts) / curr_nof_steps/curr_batch_size
                    dice_item = ospa_batch.cpu().detach().numpy()
                    loss_item = loss.item()
                val_loss += loss_item * curr_batch_size
                val_dice += dice_item * curr_batch_size
                val_loss_ts += np.sum(loss_b_ts.cpu().detach().numpy(), axis=0) #* curr_batch_size
                val_dice_ts += np.sum(ospa_batch_b_ts.cpu().detach().numpy(), axis=0) #* curr_batch_size
                total_nof_lost_trajs += nof_lost_trajs
                n_total += curr_batch_size
                nn3_time_per_batch_single_step += nn3_time / curr_nof_steps
                atrapp_time_per_batch_single_step += atrapp_time / curr_nof_steps
                meas_time_per_batch_single_step += meas_time / curr_nof_steps
                if not self.opt.dont_print_progress: print_progress("eval "+self.epoch_string, i_batch, n_batch)
                if (not time_to_stop ==0) and (time.time() > time_to_stop):
                    break
        if not self.opt.dont_print_progress:
            sys.stdout.write("\r" + "");
            sys.stdout.flush()
        return val_loss / n_total, val_dice / n_total, val_loss_ts/n_total, val_dice_ts/n_total, total_nof_lost_trajs/n_total, i_batch+1, (atrapp_time_per_batch_single_step/(i_batch+1), nn3_time_per_batch_single_step/(i_batch+1), meas_time_per_batch_single_step/(i_batch+1))

    def get_dataloader(self, dataset, batch_size, num_workers):
        sampler = torch.utils.data.RandomSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,  shuffle=False)

    def set_nns_skip(self):
        self.model.nn3.skip = 1

    def inference(self):
        self.model.eval()
        if inferencemode[self.opt.inference_mode] in [inferencemode['paint'], ]:
            self.inference_paint()
        elif inferencemode[self.opt.inference_mode]==inferencemode['eval']:
            return self.inference_evaluate()

    def inference_paint(self):
        def on_move(event):
            if event.inaxes == ax:
                if ax.button_pressed in ax._rotate_btn:
                    ax2.view_init(elev=ax.elev, azim=ax.azim)
                elif ax.button_pressed in ax._zoom_btn:
                    ax2.set_xlim3d(ax.get_xlim3d())
                    ax2.set_ylim3d(ax.get_ylim3d())
                    ax2.set_zlim3d(ax.get_zlim3d())
            elif event.inaxes == ax2:
                if ax2.button_pressed in ax2._rotate_btn:
                    ax.view_init(elev=ax2.elev, azim=ax2.azim)
                elif ax2.button_pressed in ax2._zoom_btn:
                    ax.set_xlim3d(ax2.get_xlim3d())
                    ax.set_ylim3d(ax2.get_ylim3d())
                    ax.set_zlim3d(ax2.get_zlim3d())
            else:
                return
            fig.canvas.draw_idle()

        test_dataloader = self.get_dataloader(self.test_batch_maker.get_epoch_sets(self.testset), self.opt.batch_size, 0)
        max_nof_plots = 1
        big_break = False
        save_fig = False
        dpi0 = 1000
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_dataloader):
                do_second_round = self.opt.inference_do_compare and not self.opt.skip_nn3
                fig = plt.figure(figsize=(9,4.1))
                plt.show(block=False)
                if do_second_round:
                    title_str = "Tracking trajectory " + str(i_batch) + " at SNR=" + str(self.opt.snr0) + " N=" + str(self.opt.nof_parts) + " using a) APF and b) NA-APF"
                    ax = fig.add_subplot(122, projection='3d')
                else:
                    pf_str = "APF" if self.opt.skip_nn3 else "NA-APF"
                    title_str = "Tracking trajectory " + str(i_batch) + " at SNR=" + str(self.opt.snr0) + " N=" + str(self.opt.nof_parts) + " using" +pf_str
                    ax = fig.add_subplot(111, projection='3d')
                fig.subplots_adjust(left=0.05, right=0.98, bottom=-0.2, top=1.2, wspace=0.07, hspace=-0.1)
                plt.suptitle(title_str)                # ax.axis('scaled')  # this line fits your images to screen
                ax.autoscale(enable=True)
                elev0 = 90
                azim0 = 0
                curr_ax = ax
                not_was_skipped_all = not self.model.nn3.skip
                while (1):
                    input_target_cpu = self.test_batch_maker.make_batch_function(sample_batched, self.opt.true_sensor_model, self.opt.make_batch_device_str)
                    inputs_cpu, target_cpu = input_target_cpu[0], input_target_cpu[1]
                    inputs = inputs_cpu.to(self.device)
                    target = target_cpu.to(self.device)
                    loss, ospa_batch, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_batch_size_with_grad = self.criterion(z=inputs,x=target)
                    break_all = False
                    plts_count = 0
                    for set_idx in np.arange(sample_batched.shape[0]):
                        curr_ax.set_xlabel('x')
                        curr_ax.set_ylabel('y')
                        curr_ax.set_zlabel('k')
                        curr_ax.tick_params(axis='both', which='major', labelsize=6)
                        curr_ax.tick_params(axis='both', which='minor', labelsize=6)
                        curr_ax.view_init(elev=elev0, azim=azim0)
                        if break_all: break
                        curr_loss = np.sum(loss[set_idx].cpu().detach().numpy())
                        curr_val_dice = np.sum(-ospa_batch[set_idx].cpu().detach().numpy())
                        skip_str = f"skip nn3={self.model.nn3.skip:d}"
                        res_str = "    avg loss per ts: " + str(curr_loss/loss[set_idx].shape[0]) + ", avg ospa_batch per ts: " + str(curr_val_dice/ospa_batch[set_idx].shape[0])
                        str1 = "b)" if curr_ax==ax else "a)"
                        pf_str = "APF" if self.model.nn3.skip else "NA-APF"
                        str0 = pf_str+", OSPA="+str(-curr_val_dice/ospa_batch[set_idx].shape[0])
                        curr_ax.set_title(str1+str0)
                        self.print_fun(skip_str)
                        self.print_fun(res_str)
                        set_idx = 0
                        x_wt0_torch = torch.concat((target[:, 0:1], target), dim=1)
                        self.pfbh.plot_3d_particle_traj_with_particles_and_real_traj(x_wt0_torch, set_idx=0, title="inference_paint, set_idx " + str(set_idx), ax=curr_ax)
                        if 1 and not do_second_round:
                            time_idx = 0
                            self.test_batch_maker.paint_z_of_particles(inputs_cpu, target_cpu, [0, 1], [10, 40, 70], self.opt.true_sensor_model)
                        plts_count += 1
                        if plts_count >= max_nof_plots:
                            break_all = True
                            break

                    if do_second_round and self.opt.inference_do_compare and (not self.model.nn3.skip):
                        ax2 = fig.add_subplot(121, projection='3d')
                        curr_ax = ax2
                        do_second_round = False
                        self.set_nns_skip()
                        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
                    else:
                        if not_was_skipped_all:
                            self.model.nn3.skip = 0
                        break
                break
            if save_fig:
                sav_str = "NNAPF_and_APF_tracking_seed459566.png"
                plt.savefig('plot_sav_dir/' + sav_str, dpi=dpi0)

    def inference_evaluate(self):
        def print_progress(epoch_string, i_batch, n_batch):
            ratio = int((i_batch + 1) * 25 / n_batch)
            sys.stdout.write("\r" + epoch_string + "[" + ">" * ratio + " " * (25 - ratio) + "] {}/{} {:.2f}%".format(i_batch + 1, n_batch, (i_batch + 1) * 100 / n_batch));
            sys.stdout.flush()
        print("starting evaluation")
        do_second_round = True
        old_test_loss, old_test_ospa = 0, 0
        avg_loss = []
        avg_dice = []
        avg_fail_rate = []
        avg_loss_ts = []
        avg_dice_ts = []
        all_epochs_time = []
        avg_atrapp_time = []
        avg_nn3_time = []
        avg_meas_time = []
        ratio_loss, ratio_dice = None, None
        while (1):
            test_loss_accum = 0
            test_dice_accum = 0
            test_loss_ts_accum = 0
            test_dice_ts_accum = 0
            all_epochs_time_accum = 0
            epochs_count = 0
            atrapp_time_accum = 0
            nn3_time_accum = 0
            meas_time_accum = 0
            fail_rate_accum = 0
            atrapp_time_accum, nn3_time_accum, meas_time_accum
            for epoch in range(self.opt.nof_epochs):
                current_time = time.strftime("%H:%M:%S", time.localtime())
                self.epoch_string = "E" + str(epoch + 1) +"/"+str(self.opt.nof_epochs)+ "[" + current_time + "]"
                if epoch == 0 or not self.opt.same_batches_for_all_epochs:
                    test_dataloader = self.get_dataloader(self.test_batch_maker.get_epoch_sets(self.testset), self.opt.batch_size, 0)
                epoch_start_time = time.time()
                test_loss, test_dice, test_loss_ts, test_dice_ts, fail_rate, _nof_batches, (atrapp_time, nn3_time, meas_time) = self._evaluation(epoch, self.test_batch_maker, test_dataloader, self.criterion, False, time_to_stop=0)
                all_epochs_time_accum+= time.time() - epoch_start_time
                test_loss_accum += test_loss
                test_dice_accum += test_dice
                test_loss_ts_accum += test_loss_ts
                test_dice_ts_accum += test_dice_ts
                fail_rate_accum += fail_rate
                atrapp_time_accum += atrapp_time
                nn3_time_accum += nn3_time
                meas_time_accum += meas_time
                epochs_count+=1

            avg_dice.append(test_dice_accum/epochs_count)
            avg_loss.append(test_loss_accum/epochs_count)
            avg_fail_rate.append(fail_rate_accum/epochs_count)
            avg_loss_ts.append(test_loss_ts_accum/epochs_count)
            avg_dice_ts.append(test_dice_ts_accum/epochs_count)
            avg_atrapp_time.append(atrapp_time_accum/epochs_count)
            avg_nn3_time.append(nn3_time_accum/epochs_count)
            avg_meas_time.append(meas_time_accum/epochs_count)
            all_epochs_time.append(all_epochs_time_accum/epochs_count)
            skip_str = "skip_nn3=" + str(self.model.nn3.skip)
            self.print_fun(f"{bcolors.OKGREEN}<{epoch + 1:d}/{self.opt.nof_epochs:d}>  epochs:{epochs_count:d}|| {skip_str:s}"
                           f", avg time {all_epochs_time_accum/epochs_count:.4f} test loss: {test_loss_accum/epochs_count:.15f}, val dice: {test_dice_accum/epochs_count:.15f}, fail_rt: {fail_rate_accum/epochs_count:.3f}"
                           f", avg times per batch per ts, atrapp: {atrapp_time_accum/epochs_count:.6f}, nn3: {nn3_time_accum/epochs_count:.6f}, mearurments: {meas_time_accum/epochs_count:.6f}{bcolors.ENDC}")
            if do_second_round and self.opt.inference_do_compare and (not self.model.nn3.skip):
                do_second_round = False
                self.set_nns_skip()
                old_test_loss, old_test_ospa = test_loss_accum/epochs_count, test_dice_accum/epochs_count
            else:
                ratio_loss = old_test_loss /(test_loss_accum/epochs_count)
                ratio_dice = old_test_ospa / (test_dice_accum/epochs_count)
                self.print_fun(f"{bcolors.OKGREEN}epochs: {self.opt.nof_epochs:d} ratio loss: {ratio_loss:.4f}, ratio dice: {ratio_dice:.4f}{bcolors.ENDC}")
                break
        return avg_loss, avg_dice, all_epochs_time, ratio_loss, ratio_dice, avg_loss_ts, avg_dice_ts, (avg_atrapp_time, avg_nn3_time, avg_meas_time)

    def start(self):
        if self.opt.make_new_trajs:
            mm = MotionModel(opt=self.opt, device='cpu')
            mm.reset(opt=self.opt, device=self.device)
            make_trajs_mm(self.opt, mm)
            self.print_fun("finished making trajectories")
        else:
            return self.inference()




