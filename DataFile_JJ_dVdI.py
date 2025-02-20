import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import root
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import os
import pandas as pd
import numpy as np
import colormap
from DataFile_JJ import dataFile_JJ

# extract filter bias
filter_directory = 'D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/useful_data/Filter/'
filter_file_name = '193.dat'

filter_data = pd.read_csv(os.path.join(filter_directory, filter_file_name), delimiter='\t', comment='#', 
                names=['bias_volt', 'Keysight_34410A_2_volt', 'dc_current', 'sample_bias', 'time2', 'true_bias'])
filter_bias = interp1d(filter_data['dc_current'], filter_data['bias_volt'], bounds_error=False, fill_value=np.nan)


class dataFile_JJ_dVdI(dataFile_JJ):
    def __init__(self, directory, file, filter_bias, R_filter=None, Rc=0, delta_true_zero_bias=0, 
                 amplifier_factor=5e2, excitation=250e-6, sr1_X_column=1, sr2_X_column=1, get_dc_current=True,
                 column_x_index=4, column_y_index=1, column_z_index=2, current_column_index=3, reference_column=1, 
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1):
        
        super().__init__(directory, file, filter_bias, Rc, delta_true_zero_bias,
                         column_x_index, column_y_index, column_z_index, current_column_index, reference_column, 
                         Xname, Yname, Zname, x_scaling, y_scaling, z_scaling)
        
        self.R_filter = R_filter

        self.sr1_X = self.df.iloc[:, sr1_X_column-1].to_numpy().reshape(self.data_shape)
        self.sr2_X = self.df.iloc[:, sr2_X_column-1].to_numpy().reshape(self.data_shape)

        self.Ac_current = self.sr1_X / 1e6 # current pre-Amplifier factor
        self.Ac_4T_bias = self.sr2_X / amplifier_factor
        self.Circuit_resistance = excitation / self.Ac_current

        if get_dc_current: self.get_dc_current() # 将Xdata由applied bias替换为current(恒流法)
        
        return None
    
    def get_dc_current(self):
        '''
        直接把 Xdata 由 applied bias 替换为 dc_current
        '''
        self.Xdata = (self.Xdata/1e3) / self.Circuit_resistance # Xdata is applied bias, 1000:1

        return None
         
    def crop(self, xmin=None, xmax=None, ymin=None , ymax=None):
        '''
        xmin, xmax should be index,
        ymin, ymax should be value.
        '''
        if ymin is None:
            ymin = self.y_box.min()
        if ymax is None:
            ymax = self.y_box.max()
        
        y_slice = np.where((ymin <= self.y_box) & (self.y_box <= ymax))[0]
        
        self.Xdata = self.Xdata[y_slice, xmin:xmax]
        self.Ydata = self.Ydata[y_slice, xmin:xmax]
        self.Zdata = self.Zdata[y_slice, xmin:xmax]
        
        self.y_box = self.y_box[y_slice]
        self.y_len = len(self.y_box)

        self.data_shape = self.Xdata.shape

        return None
    
    def croptplr(self, left=0, right=None, top=0, bottom=None):
        self.Xdata = self.Xdata[top:bottom, left:right]
        self.Ydata = self.Ydata[top:bottom, left:right]
        self.Zdata = self.Zdata[top:bottom, left:right]
        
        self.y_box = self.y_box[top:bottom]
        self.y_len = len(self.y_box)

        self.data_shape = self.Xdata.shape

        return None
    
    def calibrate_dVdI(self):
        self.Zdata_uncalibrated = self.Zdata
        self.Zdata = self.Zdata - self.R_filter(self.Xdata)

        return None
    
    def linecuts(self, y_slice=[], y_list=None, z_shift=0, plot=True, alpha=0.5, mark_y=True, mark_y_ref=True,
                 xlim=(None, None), zlim=(None, None), vmin=None, vmax=None, cmin=0, cmax=1, gamma=1):
            '''
            You can specify y list with specific y value, or you can specify a direct slice.
            If neither of them is given, return all linecuts.
            注意! 需要给的是slice而不能是一个单个的int
            注意! 返回linecuts的copy而不是指针
            注意! 返回的是好多linecuts组成的矩阵, 如果只有一条linecut, 返回一个只有一行的矩阵[[*,*,*,*]]
            xlim和ylim可以限制画图范围, 不改变数据, 输入应该为一个元组, e.g. xlim=(xmin, xmax)
            '''
            if y_list is None: # if y list is not given
                y_list = self.y_box[y_slice]
            else: # if y_list is given, find corresponding y_silce
                y_slice = []
                for y in y_list:
                    y_slice.append(self.idx_y(y))

            if plot:
                cmap_generator = colormap.Colormap('D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/Seismic.npy',
                                                    min=cmin, max=cmax, gamma=gamma)
                fig, axs = plt.subplots(1, 2, figsize=(2*7, 1*4), dpi=100)
                fig.set_facecolor('white')

                for i,y in enumerate(y_list):
                    c = 'C' + str(i) # unify colors
                    idx_y = self.idx_y(y)
                    axs[0].plot(self.Xdata[idx_y]*self.x_scaling, self.Zdata[idx_y]*self.z_scaling+i*z_shift, 
                                label='y={:.3g}'.format(y), lw=0.5, c=c)
                    if mark_y: axs[1].axvline(y*self.y_scaling, ls='--', label='y = {}'.format(y), lw=1, c=c, alpha=alpha)
                ax0 = axs[1].pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                        cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
                if (mark_y_ref) & (~(np.isnan(self.y_reference))): 
                    idx_y = self.idx_y(self.y_reference)
                    axs[0].plot(self.Xdata[idx_y]*self.x_scaling, self.Zdata[idx_y]*self.z_scaling, 
                                label='y={:.3g}'.format(self.y_reference), lw=0.7, c='k')
                    axs[1].axvline(self.y_reference*self.y_scaling, ls='--', lw=0.7, c='k', alpha=1) # 特别标记 y reference
                axs[0].set_xlabel(self.Xname)
                axs[0].set_ylabel(self.Zname)
                axs[0].set_title(self.file + ', Rc={:.3g} $\Omega$, delta bias={:.3g}'.format(self.Rc, self.delta_true_zero_bias))
                axs[1].set_xlabel(self.Yname)
                axs[1].set_ylabel(self.Xname)
                axs[1].set_title(self.file + 
                                 ', Rc={:.3g} $\Omega$, y_ref={:.3g}, delta bias={:.3g}'
                                 .format(self.Rc, self.y_reference, self.delta_true_zero_bias))
                axs[0].legend(bbox_to_anchor=(0, 1), loc=2, prop={'size': 8}, framealpha=0.3)

                axs[0].set_xlim(*xlim) # Zoom in
                axs[1].set_ylim(*xlim)
                axs[0].set_ylim(*zlim)
                ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))

                fig.colorbar(ax0, ax=axs[1], label=self.Zname)
                # fig.subplots_adjust(wspace=0.25)
                plt.show()

            return np.copy(self.Xdata[y_slice]), np.copy(self.Zdata[y_slice])

    def remove_y_slice(self, y_slice, remove=False, plot=True):
        y_slice = np.arange(self.y_len)[y_slice] # 保证把 slice(*, *, *) 转化为list
        


# -------------------------- Finding Ic(B) -------------------------------
    # 找出来的Ic1, Ic2是负的, 但Ic1_box, Ic2_box都是正的

    def find_Ic(self, x, z, R_threshold, y=np.nan,
                plot=False):
        '''
        For a given I-dVdI curve, find Ic. 
        data must be a pandas dataFrame with two columns at least, one for I,  the other for V
        '''
        positive_slice = np.where(x >= 0)
        negative_slice = np.where(x <= 0)
        
        try:
            Ic1 = (x[negative_slice][ np.where(z[negative_slice] >= R_threshold) ])[-1] # 这里为了紧凑写得比较套娃, 思路是找左右半边的z的最大值对应的x
        except IndexError:
            Ic1 = 0         #如果遇到上面[-1]的索引失效,即数组长度为0, 则将Ic1置0
        
        try:
            Ic2 = (x[positive_slice][ np.where(z[positive_slice] >= R_threshold) ])[0]
        except IndexError:
            Ic2 = 0

        if plot:
            plt.plot(x, z)
            plt.axvline(Ic1, c='m', alpha=0.25)
            plt.axvline(Ic2, c='k', alpha=0.25)
            plt.title('Ic1={:.3g}, Ic2={:.3g}, y={:.3g}'.format(Ic1, Ic2, y))
            plt.show()
        return Ic1, Ic2
 
    def find_Ic_of_B(self, R_threshold, plot=True, plot_each=False,
                     alpha=1, vmin=None, vmax=None, cmin=0, cmax=1, gamma=1):
        self.Ic1_box = []
        self.Ic2_box = []

        for i,y in enumerate(self.y_box):
            Ic1, Ic2 = self.find_Ic(self.Xdata[i], self.Zdata[i], R_threshold=R_threshold, y=y,  plot=plot_each)
            self.Ic1_box.append(Ic1)
            self.Ic2_box.append(Ic2)
        self.Ic1_box = - np.array(self.Ic1_box)
        self.Ic2_box =   np.array(self.Ic2_box)

        if plot:
            cmap_generator = colormap.Colormap('D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/Seismic.npy',
                                                min=cmin, max=cmax, gamma=gamma)
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
            axs[0].plot(self.Xdata.T*1e9, self.Zdata.T, lw=0.3)
            axs[0].set_xlabel('I(nA)')
            axs[0].set_ylabel('Vbias(mV)')
            axs[0].set_title(self.file)
            
            ax0 = axs[1].pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                    cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
            ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(ax0, ax=axs[1], label=self.Zname)

            axs[1].plot(self.y_box*self.y_scaling,  self.Ic2_box*self.x_scaling, 
                        marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic2', c='k')
            axs[1].plot(self.y_box*self.y_scaling, -self.Ic1_box*self.x_scaling, 
                        marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic1', c='green')
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel(self.Xname)
            axs[1].set_title(self.file)
            axs[1].legend()
            # axs[1].ylim(0,500)
            # axs[1].plot(y_list, Ic2_list)
            plt.show()

        return None

    def update_Ic_in_y_slice(self, y_slice, R_threshold, update=True, update_Ic1=True, update_Ic2=True,
                             plot=True, mark_y=True, alpha=1, vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, plot_each=False):
        y_slice = np.arange(self.y_len)[y_slice] # 保证把 slice(*, *, *) 转化为list

        if update:
            for i,y in zip(y_slice, self.y_box[y_slice]):
                Ic1, Ic2 = self.find_Ic(self.Xdata[i], self.Zdata[i], R_threshold=R_threshold, y=y,  plot=plot_each)
                if update_Ic1: self.Ic1_box[i] = - Ic1
                if update_Ic2: self.Ic2_box[i] =   Ic2
        
        if plot:
            cmap_generator = colormap.Colormap('D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/Seismic.npy',
                                                min=cmin, max=cmax, gamma=gamma)
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
            for i in y_slice:
                axs[0].plot(self.Xdata[i]*self.x_scaling, self.Zdata[i]*self.z_scaling, lw=0.3, 
                            label='i={}, y={:.5g}'.format(i, self.y_box[i]))
                axs[0].axhline(R_threshold, ls='--', lw=0.3, c='k', alpha=0.3)
                axs[0].set_xlabel(self.Xname)
                axs[0].set_ylabel(self.Zname)
                axs[0].legend(prop={'size': 6}, framealpha=0.3)
                axs[0].set_title(self.file)
            
            ax0 = axs[1].pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                    cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
            ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(ax0, ax=axs[1], label=self.Zname)

            axs[1].plot(self.y_box*self.y_scaling,  self.Ic2_box*self.x_scaling, 
                        marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic2', c='k')
            axs[1].plot(self.y_box*self.y_scaling, -self.Ic1_box*self.x_scaling, 
                        marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic1', c='green')
            if mark_y: 
                axs[1].axvline(self.y_box[y_slice][ 0]*self.y_scaling, ls='--', lw=1, c='k', alpha=0.5)
                axs[1].axvline(self.y_box[y_slice][-1]*self.y_scaling, ls='--', lw=1, c='k', alpha=0.5)
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel(self.Xname)
            axs[1].set_title(self.file)
            axs[1].legend()
            # axs[1].ylim(0,500)
            # axs[1].plot(y_list, Ic2_list)
            plt.show()
        
        return None
