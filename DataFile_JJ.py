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

# extract filter bias
filter_directory = 'D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/useful_data/Filter/'
filter_file_name = '193.dat'

filter_data = pd.read_csv(os.path.join(filter_directory, filter_file_name), delimiter='\t', comment='#', 
                names=['bias_volt', 'Keysight_34410A_2_volt', 'dc_current', 'sample_bias', 'time2', 'true_bias'])
filter_bias = interp1d(filter_data['dc_current'], filter_data['bias_volt'], bounds_error=False, fill_value=np.nan)


# 我希望得到一个class，它吃进去一个dat文件，和指明需要输出的数据列索引，然后给我吐出xyz的2D数据格式，并且可以自己插值，或者根据我指定的x向量进行插值
# 我需要为它指定filter_bias, Rc, delta_true_zero_bias
# Principle: 
# 1.不要改变Original data which means self.Xdata,Ydata,Zdata. 
# 2.所有的操作几乎都是对interp数据直接进行的 
# 3.所有对Ydata_interp的操作都同步到y_box, y_reference
# 我希望得到一个class，它吃进去一个dat文件，和指明需要输出的数据列索引，然后给我吐出xyz的2D数据格式，并且可以自己插值，或者根据我指定的x向量进行插值
# 我需要为它指定filter_bias, Rc, delta_true_zero_bias
# Principle: 
# 1.不要改变Original data which means self.Xdata_original,Ydata_original,Zdata_original. 
# 2.所有的操作直接对Xdata, Ydata, Zdata进行, 包括calibrate_bias, deduct_true_zero_B, crop, swap_axes, xderiv, yderiv
# 3.所有对Ydata的操作都同步到y_box, y_reference
# 4.所有以上操作都有一份对data_interp的操作? to be fulfilled...
# 5.get() return None, calculate() return calculated results
# 6.除了bias的数据单位默认为mV(由于1000:1分压)以外其他物理量都是标准单位制, 此原则包括在所有的计算之中, 即默认bias单位为mV
# 7.一般来讲如果一个变量是xx_box, 那意味着它是y的函数, 和self.y_box对齐
# 8.find_xx()中搭配write_xx_in(), 每次手动找完之后把必要的结果存起来, 后续需要用到这些结果的函数搭配read_xx_from(), 这样手动寻找只需要找一次.

# ! 寻找Rc时的中点有问题

class dataFile_JJ(object):
    def __init__(self, directory, file, filter_bias, Rc=0, delta_true_zero_bias=0,
                 column_x_index=4, column_y_index=1, column_z_index=2, current_column_index=3, reference_column=1,
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1):
        '''
        除了bias的数据单位默认为mV(由于1000:1分压)以外其他物理量都是标准单位制
        '''
        self.directory = directory
        self.file = file
        self.filename, self.file_suffix = os.path.splitext(file)
        self.file_path = os.path.join(directory, file)  # str 

        self.filter_bias = filter_bias
        self.delta_true_zero_bias = delta_true_zero_bias
        if self.read_Rc_from(): 
            self.Rc = Rc
            self.y_reference = np.nan

        self.column_x_index = column_x_index  # int e.g. 1 # for 4-terminal bias vs xx scan, this is Agilent_2 (sample bias)
        self.column_y_index = column_y_index  # int e.g. 2 
        self.column_z_index = column_z_index  # int e.g. 6
        self.reference_column = reference_column # int 1 by default # for 4-terminal bias vs xx scan, this is bias column
        self.current_column_index = current_column_index

        self.df = pd.read_csv(self.file_path, delimiter='\t', comment='#', header=None)
        self.get_shape_and_names()
        self.data_shape = (self.y_len, self.x_len)
        self.Xdata = self.df.iloc[:, column_x_index-1].to_numpy().reshape(self.data_shape)
        self.Ydata = self.df.iloc[:, column_y_index-1].to_numpy().reshape(self.data_shape)
        self.Zdata = self.df.iloc[:, column_z_index-1].to_numpy().reshape(self.data_shape)
        self.Current_data = self.df.iloc[:, current_column_index-1].to_numpy().reshape(self.data_shape)

        self.Xdata_original = self.Xdata.copy()
        self.Ydata_original = self.Ydata.copy()
        self.Zdata_original = self.Zdata.copy()
        self.Current_data_original = self.Current_data.copy()

        self.xmax = np.max(abs(self.Xdata))
        self.y_box = self.Ydata[:, 0]
        self.y_step = (self.y_box.max() - self.y_box.min()) / (len(self.y_box)-1)

        if Xname is not None: # 如果制定了name就按照指定的来, 如果没有指定, 前面的self.get_shape_and_names()已经自动识别了name
            self.Xname = Xname
        if Yname is not None:
            self.Yname = Yname
        if Zname is not None:
            self.Zname = Zname

        self.x_scaling = x_scaling # 这些用于各种函数自己画图时对数据进行缩放
        self.y_scaling = y_scaling # 这些用于各种函数自己画图时对数据进行缩放
        self.z_scaling = z_scaling # 这些用于各种函数自己画图时对数据进行缩放
        
        self.Phi0 = 20.6783 # 20 Gauss*um^2

    def reset(self):
        self.__init__(self.directory, self.file, self.filter_bias, self.Rc, self.delta_true_zero_bias,
                      self.column_x_index, self.column_y_index, self.column_z_index, self.current_column_index, self.reference_column,
                      self.Xname, self.Yname, self.Zname, self.x_scaling, self.y_scaling, self.z_scaling)
        return None

    def idx_y(self, y): # find the index of y in y_box
        return np.where(np.isclose(self.y_box, y, atol=self.y_step/10)==True)[0][0]
    
    def y_list(self, y_slice):
        return self.y_box[y_slice]
    
    def y_slice(self, y_list):
        y_slice = []
        for y in y_list:
            y_slice.append(self.idx_y(y))
        return y_slice

    def remove_jumping_points_in_linecut(self, x, z, y_value='Not given', plot=False, prominence=2e-3):
        '''
        返回一个去除跳点的z, 可以选择是否plot去除跳点前后的效果对比图
        '''
        z_beauty = np.copy(z)
        peaks = find_peaks(z_beauty, prominence=prominence)[0]
        false_peaks_idx = []
        for i,j in enumerate(peaks):
            if (j-2 < 0) | (j+2 >= len(z_beauty)):
                z_beauty[j] = (z_beauty[j-1] + z_beauty[j+1])/2
            elif ((z_beauty[j] > z_beauty[j+2]) & (z_beauty[j] > z_beauty[j-2])): # notice that the left or right point of a dip may be indentified as a peak, which situation doesn't follow this judgement.
                z_beauty[j] = (z_beauty[j-1] + z_beauty[j+1])/2
            else:
                false_peaks_idx.append(i)
        peaks = np.delete(peaks, false_peaks_idx)

        # then remove jumping dips; once peaks are all removed, there won't be any false dips
        dips = find_peaks(-z_beauty, prominence=prominence)[0]
        for i in dips:
            z_beauty[i] = (z_beauty[i-1] + z_beauty[i+1])/2

        if plot:
            if y_value is None: # double check
                y_value = 'Not given'
            lw = 0.6
            markersize=3
            plt.plot(x, z, label='original',lw=lw)
            plt.plot(x, z_beauty, label='corrected', lw=lw)
            plt.plot(x.take(peaks), z.take(peaks), 'o', label='detected jumping peaks', markersize=markersize, alpha=0.25)
            plt.plot(x.take(dips), z.take(dips), 'o', label='detected jumping dips', markersize=markersize, alpha=0.25)
            plt.title('value of y={}'.format(y_value))
            plt.legend()
            plt.show()

        return z_beauty

    def original_linecuts(self, y_slice=[], y_list=None, z_shift=0, plot=False, alpha=0.5, mark_y=True, mark_y_ref=True):
                '''
                You can specify y list with specific y value, or you can specify a direct slice.
                If neither of them is given, return all linecuts.
                注意! 需要给的是slice而不能是一个单个的int
                注意! 返回linecuts的copy而不是指针
                注意! 返回的是好多linecuts组成的矩阵, 如果只有一条linecut, 返回一个只有一行的矩阵[[*,*,*,*]]
                '''
                if y_list is None: # if y list is not given
                    y_list = self.y_box[y_slice]
                else: # if y_list is given, find corresponding y_silce
                    y_slice = []
                    for y in y_list:
                        y_slice.append(self.idx_y(y))

                if plot:
                    fig, axs = plt.subplots(1, 2, figsize=(2*7, 1*4), dpi=100)
                    fig.set_facecolor('white')
                    for i,y in enumerate(y_list):
                        c = 'C' + str(i) # unify colors
                        idx_y = self.idx_y(y)
                        axs[0].plot(self.Xdata_original[idx_y]*self.x_scaling, self.Zdata_original[idx_y]*self.z_scaling+i*z_shift, 
                                    label='y={:.3g}'.format(y), lw=0.5, c=c)
                        if mark_y: axs[1].axvline(y*self.y_scaling, ls='--', label='y = {}'.format(y), lw=1, c=c, alpha=alpha)
                    ax0 = axs[1].pcolormesh(self.Ydata_original*self.y_scaling, self.Xdata_original*self.x_scaling, self.Zdata_original*self.z_scaling, 
                                            cmap='seismic', shading='nearest')
                    if (mark_y_ref) & (~(np.isnan(self.y_reference))): 
                        idx_y = self.idx_y(self.y_reference)
                        axs[0].plot(self.Xdata_original[idx_y]*self.x_scaling, self.Zdata_original[idx_y]*self.z_scaling, 
                                    label='y={:.3g}'.format(self.y_reference), lw=0.7, c='k')
                        axs[1].axvline(self.y_reference*self.y_scaling, ls='--', lw=0.7, c='k', alpha=1) # 特别标记 y reference
                    axs[0].set_xlabel(self.Xname)
                    axs[0].set_ylabel(self.Zname)
                    axs[0].set_title(self.file + ', Rc={:.3g} $\Omega$'.format(self.Rc))
                    axs[1].set_xlabel(self.Yname)
                    axs[1].set_ylabel(self.Xname)
                    axs[1].set_title(self.file + ', Rc={:.3g} $\Omega$, y_ref={:.3g}'.format(self.Rc, self.y_reference))
                    axs[0].legend(bbox_to_anchor=(0, 1), loc=2, prop={'size': 8}, framealpha=0.3)
                    axs[0].axhline(0,c='k',alpha=0.25)
                    fig.colorbar(ax0, ax=axs[1], label=self.Zname)
                    # fig.subplots_adjust(wspace=0.25)
                    plt.show()

                return np.copy(self.Xdata_original[y_slice]), np.copy(self.Zdata_original[y_slice])

    def find_Rc_at(self, y_value=None, idx_y_value=None, plot=True, plot_removejp=True,
                   correct_resistance_step=0.25, slope_threshold=30, num_points=30, iteration_limit=5000,
                   xlim=(None, None), zlim=(None, None)):
        '''
        使用的是原始数据，不需要差值
        以y=y_value对应的linecut为例寻找contact resistance, 如果不指定 y 或者idx of y, 用最中心的一条lincut
        先减掉filter bias,和初始的contact resistance的分压, 然后
        取中点前后num_points个点的数据做线性拟合, 不断修正contact resistance直到终点附近的斜率低于 slop_threshold或者迭代次数超过限制
        返回Rc, 同时自动修正self.Rc
        '''
        if not y_value is None: # 给了y value就以y value为准
            idx_y_value = self.idx_y(y_value)
        elif idx_y_value is None:  # 没给y value也没给idx
            idx_y_value = self.y_len//2 # 默认取中心的linecut
            y_value = self.y_box[idx_y_value]
        else: # 没给y value 但给了 idx
            y_value = self.y_box[idx_y_value]

        x, z = self.linecuts(y_slice=(idx_y_value), plot=False) # return linecuts, need to reshape
        x = x.reshape(-1)
        z = z.reshape(-1)
        # z = z - self.filter_bias(x) - x*self.Rc*1e3 # 可以更好地去除噪点
        z = self.remove_jumping_points_in_linecut(x, z, y_value=y_value, plot=plot_removejp)
        
        mid_idx = np.argmin(np.abs(x - 0))
        useful_slice = np.arange(mid_idx - num_points, mid_idx + num_points)
        x_useful = x[useful_slice]
        z_useful = z[useful_slice]

        correct_resistance = 0
        iteration_times = 0
        while(True):
            k, b = np.polyfit(x_useful, z_useful, 1)
            if abs(k) < slope_threshold:
                break
            elif iteration_times >= iteration_limit:
                print('Failed, iteration time out! y_value={}, k={}'.format(y_value, k))
                break
            elif k > slope_threshold:
                z_useful = z_useful - x_useful*correct_resistance_step*1e3
                correct_resistance += correct_resistance_step
            elif k < -slope_threshold:
                z_useful = z_useful + x_useful*correct_resistance_step*1e3
                correct_resistance -= correct_resistance_step
            
            iteration_times += 1

        self.Rc += correct_resistance
        self.y_reference = y_value

        if plot:
            plt.plot(x, z-x*correct_resistance*1e3, label='data')
            plt.plot(x, x*k + b, label='fit', lw=0.5, alpha=0.5)
            plt.title('y_value={}, k={:.2e}, iteration times={}, corrected Rc={}, $\Delta$Rc={}'
                      .format(y_value, k, iteration_times, self.Rc, correct_resistance))
            plt.xlim(*xlim)
            plt.ylim(*zlim)

            plt.show()

        return self.Rc, k

    def func_calibrate_sample_bias(self, bias, current, Rc, delta_true_zero_bias):
        return bias - self.filter_bias(current) - current*Rc*1e3 - delta_true_zero_bias*1e3

    def calibrate_sample_bias(self):
        '''
        For supercurrent I-V 2D, I is Xdata, V is Zdata
        '''
        self.Zdata_uncalibrated = self.Zdata.copy()
        self.Zdata = self.func_calibrate_sample_bias(self.Zdata, self.Xdata, self.Rc, self.delta_true_zero_bias)
        return None
    
    def uncalibrate_sample_bias(self):
        self.Zdata = self.Zdata_uncalibrated
        return None
    
    def crop(self, ymin=None , ymax=None):
        '''
        only y axis can be cropped for the original data.
        '''
        if ymin is None:
            ymin = self.y_box.min()
        if ymax is None:
            ymax = self.y_box.max()
        y_slice = np.where((ymin <= self.y_box) & (self.y_box <= ymax))[0]
        
        self.Xdata = self.Xdata[y_slice]
        self.Ydata = self.Ydata[y_slice]
        self.Zdata = self.Zdata[y_slice]
        
        self.y_box = self.y_box[y_slice]
        self.y_len = len(self.y_box)

        return None
    
    def swap_axes(self):

        return None
    
    def xderiv(self):
        for i,x in enumerate(self.Xdata):
            self.Zdata[i] = np.gradient(self.Zdata[i], x)
        return None

    def yderiv(self):
        
        return None

    def remove_jumping_points(self, prominence=2e-3):
        self.Zdata_unremoved = self.Zdata.copy()
        for i,z in enumerate(self.Zdata):
            self.Zdata[i] = self.remove_jumping_points_in_linecut(None, z, prominence=prominence)
        return None
    
    def unremove_jumping_points(self):
        self.Zdata = self.Zdata_unremoved
        return None

    def deduct_true_zero_B(self):
        if self.read_true_zero_B_from():
            self.true_zero_B = np.nan

        self.Ydata = self.Ydata - self.true_zero_B
        self.y_box = self.y_box - self.true_zero_B
        self.y_reference = self.y_reference - self.true_zero_B
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
                axs[0].axhline(0,c='k',alpha=0.25)

                axs[0].set_xlim(*xlim) # Zoom in
                axs[1].set_ylim(*xlim)
                axs[0].set_ylim(*zlim)
                ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))

                fig.colorbar(ax0, ax=axs[1], label=self.Zname)
                # fig.subplots_adjust(wspace=0.25)
                plt.show()

            return np.copy(self.Xdata[y_slice]), np.copy(self.Zdata[y_slice])

    def plot_in_colormap(self, boxes=[], labels=[], vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, alpha=1, figsize=(6,4)):
            cmap_generator = colormap.Colormap('D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/Seismic.npy',
                                                min=cmin, max=cmax, gamma=gamma)
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
            ax0 = ax.pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                    cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
            ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(ax0, ax=ax, label=self.Zname)
            for box,label in zip(boxes, labels):
                ax.plot(self.y_box*self.y_scaling,  box*self.x_scaling, marker='o', markersize=1, lw=0.5, label=label, alpha=alpha)
            ax.set_xlabel(self.Yname)
            ax.set_ylabel(self.Xname)
            ax.set_title(self.file)
            ax.legend()
            plt.show()

# ------------------------------- Write and read files -----------------------------------

    def get_shape_and_names(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[6:]
        f.close()
        self.y_len = int(lines[6][8:])
        self.x_len = int(lines[20][8:])
        self.Xname = lines[self.column_x_index*14 - 9 - 1][8:][:-1] # [i*14 - 9 - 1]选出第i行, 即name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n
        self.Yname = lines[self.column_y_index*14 - 9 - 1][8:][:-1] # [i*14 - 9 - 1]选出第i行, 即name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n
        self.Zname = lines[self.column_z_index*14 - 9 - 1][8:][:-1] # [i*14 - 9 - 1]选出第i行, 即name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n

        return None
  
    def write_Rc_in(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_Rc.txt'
        with open(file, 'w') as f:
            f.write(str(self.Rc) + '\n' + str(self.y_reference))
            f.close()
            print('Rc={} and y reference {} are written in \"{}\"'.format(self.Rc, self.y_reference, file))
        return None
    
    def write_true_zero_B_in(self, true_zero_B, file=None):
        self.true_zero_B = np.round(true_zero_B, 4)
        if file is None:
            file = self.directory + self.filename + '_true_zero_B.txt'
        with open(file, 'w') as f:
            f.write(str(self.true_zero_B))
            f.close()
            print('true zero B={}T are written in \"{}\"'.format(self.true_zero_B, file))
        return None       

    def read_Rc_from(self, file=None): # 如果成功读取
        if file is None:
            file = self.directory + self.filename + '_Rc.txt'
        try:
            f = open(file, 'r')
            self.Rc = float(f.readline())
            self.y_reference = float(f.readline())
            f.close()
            print('Rc={} Ohm and y reference={} are extracted from \"{}\"'.format(self.Rc, self.y_reference, file))
            return False
        except IOError:
            print( "Wrong! Rc file is not accessible. Use default Rc")
            return True
        
    def read_true_zero_B_from(self, file=None): # 如果成功读取
        if file is None:
            file = self.directory + self.filename + '_true_zero_B.txt'
        try:
            f = open(file, 'r')
            self.true_zero_B = float(f.readline())
            f.close()
            print('true zero B={}T is extracted from \"{}\"'.format(self.true_zero_B, file))
            return False
        except IOError:
            print( "Wrong! True zero B file is not accessible, use np.nan")
            return True 

# ----------------------------------- append ---------------------------------------

    def append(self, new_data):
        '''
        沿y轴添加新的数据, 用于拼接多次扫描的数据, 如果新数据沿x方向的长度和本数据不相等, 把短的数据用np.nan补齐;
        这样补齐之后就无法用self,linecuts()画出2D了, 只能用self.interp_linecuts();
        不改变new_data.data, 只改变self.data
        '''
        new_ylen, new_xlen = new_data.data_shape
        
        if new_xlen < self.x_len: # 如果新数据比较短
            nan_len = self.x_len - new_xlen
            X = np.pad(new_data.Xdata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把新数据用nan补齐
            Y = np.pad(new_data.Ydata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把新数据用nan补齐
            Z = np.pad(new_data.Zdata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把新数据用nan补齐
        elif new_xlen > self.x_len: # 如果新数据比较长
            nan_len = new_xlen - self.x_len
            self.Xdata = np.pad(self.Xdata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把本数据用nan补齐
            self.Ydata = np.pad(self.Ydata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把本数据用nan补齐
            self.Zdata = np.pad(self.Zdata, ((0, 0), (0, nan_len)), mode='constant', constant_values=np.nan) # 先把本数据用nan补齐
            X, Y, Z = new_data.Xdata, new_data.Ydata, new_data.Zdata
        else: # 两个数据一样长
            X, Y, Z = new_data.Xdata, new_data.Ydata, new_data.Zdata

        # 拼接
        self.Xdata = np.concatenate((self.Xdata, X), axis=0)
        self.Ydata = np.concatenate((self.Ydata, Y), axis=0)
        self.Zdata = np.concatenate((self.Zdata, Z), axis=0)

        # mask value
        if new_xlen != self.x_len:
            self.Xdata = np.ma.masked_array(self.Xdata)
            self.Ydata = np.ma.masked_array(self.Ydata)
            self.Zdata = np.ma.masked_array(self.Zdata)

        # 更新参数
        self.data_shape = np.shape(self.Xdata)
        self.y_len, self.x_len = self.data_shape
        self.y_box = self.Ydata[:, 0]
        self.xmax = np.max(abs(self.Xdata))

        return None
    
    def left_interpend(self, new_data):
        x_interp = self.Xdata[0]
        new_Xdata = np.tile(x_interp, (new_data.y_len, 1))
        new_Ydata = np.tile(new_data.y_box, (len(x_interp), 1)).T
        new_Zdata = np.full((new_data.y_len, len(x_interp)), np.nan)

        for i,x in enumerate(new_data.Xdata):
            new_Zdata[i] = interp1d(x, new_data.Zdata[i], bounds_error=False, fill_value=np.nan)(x_interp)

        # 拼接
        self.Xdata = np.vstack((new_Xdata, self.Xdata))
        self.Ydata = np.vstack((new_Ydata, self.Ydata))
        self.Zdata = np.vstack((new_Zdata, self.Zdata))

        # 更新参数
        self.data_shape = np.shape(self.Xdata)
        self.y_len, self.x_len = self.data_shape
        self.y_box = self.Ydata[:, 0]
        self.xmax = np.max(abs(self.Xdata))

        return None
    
    def right_interpend(self, new_data):
        x_interp = self.Xdata[-1]
        new_Xdata = np.tile(x_interp, (new_data.y_len, 1))
        new_Ydata = np.tile(new_data.y_box, (len(x_interp), 1)).T
        new_Zdata = np.full((new_data.y_len, len(x_interp)), np.nan)

        for i,x in enumerate(new_data.Xdata):
            new_Zdata[i] = interp1d(x, new_data.Zdata[i], bounds_error=False, fill_value=np.nan)(x_interp)

       # 拼接
        self.Xdata = np.vstack((self.Xdata, new_Xdata))
        self.Ydata = np.vstack((self.Ydata, new_Ydata))
        self.Zdata = np.vstack((self.Zdata, new_Zdata))

        # 更新参数
        self.data_shape = np.shape(self.Xdata)
        self.y_len, self.x_len = self.data_shape
        self.y_box = self.Ydata[:, 0]
        self.xmax = np.max(abs(self.Xdata))

        return None

# -------------------------------- interpolation -----------------------------------

    def interp(self, x_interp=None, multiplier=2, interp_kind='linear', filter_bias=None, 
               remove_jumping_points=False, calibrate_sample_bias=True):
        '''
        interpolating X,Y,Z data. x_interp and filter bias can be specified outside; if not given, use that from itself.
        '''
        if x_interp is None: # if x_interp is not given, then use self x_interp
            x_interp=np.linspace(-self.xmax, self.xmax, self.x_len*multiplier)

        if filter_bias is None:
            filter_bias = self.filter_bias

        self.x_interp = x_interp
        self.Zdata_interp = np.full((self.y_len, len(x_interp)), np.nan)
        self.Xdata_interp = np.tile(x_interp, (self.y_len, 1))
        self.Ydata_interp = np.tile(self.y_box, (len(x_interp), 1)).T

        for i,x in enumerate(self.Xdata):
            z = self.Zdata[i].copy() # copy a zdata
            if calibrate_sample_bias:
                z = self.func_calibrate_sample_bias(z, x, self.Rc, self.delta_true_zero_bias)
            if remove_jumping_points: # remove jumping points
                z = self.remove_jumping_points_in_linecut(x, z, plot=False, prominence=5e-3)
            z_interp = interp1d(x, z, kind=interp_kind, bounds_error=False, fill_value=np.nan)(x_interp) # interpolation
            self.Zdata_interp[i] = z_interp
            
        return None
    
    def crop_interp(self, 
                    left=0, right=None, bottom=0, top=None,
                    xmin=None, xmax=None, ymin=None, ymax=None):
        '''
        两种crop方法, 一种指定索引, 一种是指定值, 当两种都指定的时候会取两种crop的交集;
        需要注意的是如果bottom/top使用了self.idx_y(some value)来指定, 那可能会出错(如果y轴是降序排列的话)
        '''
        if self.y_box[0] > self.y_box[-1]: # 如果y轴降序 (注意x_interp必为升序)
            y_box = np.flip(self.y_box)[bottom:top] # 先取出真正想要的那部分y值
            bottom = self.idx_y(y_box[-1]) # 取出真正的bottom
            top = self.idx_y(y_box[0]) # 取出真正的top

        self.Xdata_interp = self.Xdata_interp[bottom:top, left:right]
        self.Ydata_interp = self.Ydata_interp[bottom:top, left:right]
        self.Zdata_interp = self.Zdata_interp[bottom:top, left:right]
        self.x_interp = self.x_interp[left:right]
        self.y_box = self.y_box[bottom:top]
        self.y_len = len(self.y_box)

        if xmin is None:
            xmin = self.x_interp.min()
        if xmax is None:
            xmax = self.x_interp.max()
        if ymin is None:
            ymin = self.y_box.min()
        if ymax is None:
            ymax = self.y_box.max()
        y_slice = np.where((ymin <= self.y_box) & (self.y_box <= ymax))[0]
        x_slice = np.where((xmin <= self.x_interp) & (self.x_interp <= xmax))[0]
        self.Xdata_interp = self.Xdata_interp[:, x_slice][y_slice, :] # 这里不知道为什么, 直接用[y_slice, x_slice]就是会报错, 烦死了
        self.Ydata_interp = self.Ydata_interp[:, x_slice][y_slice, :]
        self.Zdata_interp = self.Zdata_interp[:, x_slice][y_slice, :]
        self.y_box = self.y_box[y_slice]
        self.y_len = len(self.y_box)

        return None
    
    def swap_axes_interp(self):
        self.Xdata_interp, self.Ydata_interp = self.Ydata_interp, self.Xdata_interp
        return
 
    def deduct_true_zero_B_interp(self):
        if self.read_true_zero_B_from():
            self.true_zero_B = np.nan

        self.Ydata_interp = self.Ydata_interp - self.true_zero_B
        self.y_box = self.y_box - self.true_zero_B
        self.y_reference = self.y_reference - self.true_zero_B
        return None

    def interp_linecuts(self, y_slice=[], y_list=None, z_shift=0, plot=True, alpha=0.5, mark_y=True, mark_y_ref=True):
        '''
        You can specify y list with specific y value, or you can specify a direct slice.
        If neither of them is given, return all linecuts.
        注意! 需要给的是slice而不能是一个单个的int
        注意! 返回linecuts的copy而不是指针
        注意! 返回的是好多linecuts组成的矩阵, 如果只有一条linecut, 返回一个只有一行的矩阵[[*,*,*,*]]
        '''
        if y_list is None: # if y list is not given
            y_list = self.y_box[y_slice]
        else: # if y_list is given, find corresponding y_silce
            y_slice = []
            for y in y_list:
                y_slice.append(self.idx_y(y))

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(2*7, 1*4), dpi=100)
            fig.set_facecolor('white')
            for i,y in enumerate(y_list):
                c = 'C' + str(i) # unify colors
                idx_y = self.idx_y(y)
                axs[0].plot(self.Xdata_interp[idx_y]*self.x_scaling, self.Zdata_interp[idx_y]*self.z_scaling+i*z_shift, 
                            label='y={:.3g}'.format(y), lw=0.5, c=c)
                if mark_y: axs[1].axvline(y*self.y_scaling, ls='--', label='y = {}'.format(y), lw=1, c=c, alpha=alpha)
            ax0 = axs[1].pcolormesh(self.Ydata_interp*self.y_scaling, self.Xdata_interp*self.x_scaling, self.Zdata_interp*self.z_scaling, 
                                    cmap='seismic', shading='nearest')
            if (mark_y_ref) & (~(np.isnan(self.y_reference))): 
                idx_y = self.idx_y(self.y_reference)
                axs[0].plot(self.Xdata_interp[idx_y]*self.x_scaling, self.Zdata_interp[idx_y]*self.z_scaling, 
                            label='y={:.3g}'.format(self.y_reference), lw=0.7, c='k')
                axs[1].axvline(self.y_reference*self.y_scaling, ls='--', lw=0.7, c='k', alpha=1) # 特别标记 y reference
            axs[0].set_xlabel(self.Xname)
            axs[0].set_ylabel(self.Zname)
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel(self.Xname)
            axs[1].set_title(self.file + ', Rc={:.3g} $\Omega$, y_ref={:.3g}'.format(self.Rc, self.y_reference))
            axs[0].legend(bbox_to_anchor=(0, 1), loc=2, prop={'size': 8}, framealpha=0.3)
            axs[0].axhline(0,c='k',alpha=0.25)
            
            if z_shift != 0: text = ', z shift={:.3g}'.format(z_shift)
            else: text = ''
            axs[0].set_title(self.file + ', Rc={:.3g} $\Omega$'.format(self.Rc) + text)

            fig.colorbar(ax0, ax=axs[1], label=self.Zname)
            # fig.subplots_adjust(wspace=0.25)
            plt.show()

        return np.copy(self.Xdata_interp[y_slice]), np.copy(self.Zdata_interp[y_slice])

    def xderiv_interp(self):
        self.Zdata_interp = np.gradient(self.Zdata_interp, self.x_interp, axis=1)
        return None
    
    def yderiv_interp(self):
        self.Zdata_interp = np.gradient(self.Zdata_interp, self.y_box, axis=0)
        return None

# ------------------------------- changing Rc -----------------------------------
    # 注意事项: 
    # 1.使用前记得检查y_box和Rc_box的一致性, 数据的y_box状态必须和使用find_Rc_of_y()时的一致.

    def num_points_box_from_Ic2_box(self, contraction=2):
        right_idx_box = np.array([np.argmin(np.abs(I-Ic/contraction))  for I,Ic in zip(self.Xdata, self.Ic2_box)])
        mid_idx_box = np.array([np.argmin(np.abs(I - 0)) for I in self.Xdata])

        num_points_box = right_idx_box - mid_idx_box

        return num_points_box

    def get_num_points_box(self, num_points=5, num_points_box=None, plot=True,
                            vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, alpha=1):
        if num_points_box is None:
            num_points_box = np.ones(self.y_len, int) * num_points

        self.num_points_box = num_points_box

        mid_idx_box = np.array([np.argmin(np.abs(I - 0)) for I in self.Xdata])
        left_idx_box = mid_idx_box - num_points_box
        right_idx_box = mid_idx_box + num_points_box
        left_xpoint_box = np.array([self.Xdata[i, left_idx] for i,left_idx in enumerate(left_idx_box)])
        right_xpoint_box = np.array([self.Xdata[i, right_idx] for i,right_idx in enumerate(right_idx_box)])

        if plot:
            self.plot_in_colormap(boxes=[left_xpoint_box, right_xpoint_box], labels=['left x', 'right x'],
                                  vmin=vmin, vmax=vmax, cmin=cmin, cmax=cmax, gamma=gamma, alpha=alpha)
        return None
            
    def find_Rc_of_y(self, y_slice=slice(0, None, 1), y_list=None, 
                     plot=True, plot_each=False, plot_removejp=False,
                     correct_resistance_step=0.25, slope_threshold=30, iteration_limit=5000,
                     xlim=(None, None), zlim=(None, None)):
        '''
        默认的Rc(y)=self.Rc是一个常数, 本函数可以指定寻找某个特定y区间内的Rc(y), 默认遍历所有的y;
        可以给定num_points_box, 对每个不同的y都用不同的num_points去find_Rc_at_y(), 也可以只给定num_points用于所有的y;
        '''
        self.Rc_box = np.ones(self.y_len) * self.Rc
        self.k_box = np.ones(self.y_len) * np.nan

        if y_list is not None: # if y_list is given, find corresponding y_silce
            y_slice = self.y_slice(y_list)
        else:
            y_list = self.y_list(y_slice)
            y_slice = self.y_slice(y_list)


        Rc0 = self.Rc
        for i,num_points in zip(y_slice, self.num_points_box[y_slice]):
            self.Rc = Rc0 # reset Rc
            Rc, k = self.find_Rc_at(idx_y_value=i, plot=plot_each, plot_removejp=plot_removejp,
                                    num_points=num_points, 
                                    correct_resistance_step=correct_resistance_step, 
                                    slope_threshold=slope_threshold, iteration_limit=iteration_limit,
                                    xlim=xlim, zlim=zlim)
            self.Rc_box[i] = Rc
            self.k_box[i] = k

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(2*7, 1*4), dpi=100)
            axs[0].plot(self.y_box, self.Rc_box, 'o', markersize=1)
            axs[0].set_xlabel(self.Yname)
            axs[0].set_ylabel('Rc(Ohm)')
            axs[1].plot(self.y_box, self.k_box, 'o', markersize=1)
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel('k')
            plt.show()

        return None
    
    def calibrate_sample_bias_with_varying_Rc(self):
        self.read_Rc_box_from()
        self.Zdata = self.func_calibrate_sample_bias(self.Zdata, self.Xdata, self.Rc_box.reshape(-1, 1), self.delta_true_zero_bias)
        return None

    def write_Rc_box_in(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_Rc_box.txt'
        np.savetxt(file, np.vstack((self.y_box, self.Rc_box, self.num_points_box)).T)
        print('Rc_box etc. are written in \"{}\"'.format(file))
        return None

    def read_Rc_box_from(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_Rc_box.txt'
        try:
            self.y_box_for_Rc_box, self.Rc_box, self.num_points_box = np.loadtxt(file).T
            print('Rc_box etc. are extracted from \"{}\".'.format(file))
            # if crop:
            #     self.crop_with_Rc_box()
            #     print('Rc_box etc. are extracted from \"{}\", and data is cropped correspondingly'.format(file))
            # else:
            if len(self.y_box_for_Rc_box) != len(self.y_box):
                print('Attention! y_box may be inconsistent with Rc_box, you may need to crop().')
            return False
        except IOError:
            print( "Wrong! Rc_box file is not accessible, you need to find_Rc_of_y().")
            return True
        
    # def crop_with_Rc_box(self):
    #     self.crop(ymin=self.y_box_for_Rc_box.min(), ymax=self.y_box_for_Rc_box.max())
    #     return None

# ------------------------------- Excess current -----------------------------------

    def get_gap(self, gap):
        self.gap = gap
        return None

    def get_excess_current_at(self, idx_y_value=0, y_value=None, plot=True):
        '''
        gap的单位是mV
        '''
        if y_value is not None:
            idx_y_value = self.idx_y(y_value)
        else:
            y_value = self.y_box[idx_y_value]
        
        I = self.Xdata[idx_y_value]
        V = self.Zdata[idx_y_value]
        double_gap = self.gap*2

        pf_slice = np.where(V >  double_gap) # Positive fitting segment
        nf_slice = np.where(V < -double_gap) # Positive fitting segment
        plus_slice = np.where(V > 0) # Negative fitting segment
        minus_slice = np.where(V < 0) # Negative fitting segment
        Ipf = I[pf_slice]
        Vpf = V[pf_slice]
        Inf = I[nf_slice]
        Vnf = V[nf_slice]
        Vplus = V[plus_slice]
        Vminus = V[minus_slice]
        k_pf, b_pf = np.polyfit(Vpf, Ipf, 1)
        k_nf, b_nf = np.polyfit(Vnf, Inf, 1)

        Iex_pf = b_pf # positive fitting
        Rn_pf = 1/k_pf
        Iex_nf = b_nf # negative fitting
        Rn_nf = 1/k_nf

        eIRDpf = Iex_pf * Rn_pf / self.gap
        eIRDnf = Iex_nf * Rn_nf / self.gap
        solpf = root(self.Zequation, 0.5, args=(eIRDpf))
        solnf = root(self.Zequation, 0.5, args=(eIRDnf))
        Tpf = 1/(1+solpf.x**2)[0]
        Tnf = 1/(1+solnf.x**2)[0]

        if plot:
            plt.axhline(0, ls='--', c='g', alpha=0.7)
            plt.axhline(-double_gap, ls='--', c='k', alpha=0.3)
            plt.axhline( double_gap, ls='--', c='k', alpha=0.3)
            plt.plot((k_pf*Vplus+b_pf)*self.x_scaling, Vplus*self.z_scaling, ls='--', label='positive fit')
            plt.plot((k_nf*Vminus+b_nf)*self.x_scaling, Vminus*self.z_scaling, ls='--', label='negative fit')
            plt.plot(I*self.x_scaling, V, c='k')
            plt.title('Iexp={:.3g}, Rnp={:.1f} Ohm, Iexn={:.3g}, Rnn={:.1f} Ohm\nTpf={:.3g}, Tnf={:.3g}, y={:.3g}'
                      .format(Iex_pf*self.x_scaling, Rn_pf/1e3, Iex_nf*self.x_scaling, Rn_nf/1e3, Tpf, Tnf, y_value))
            plt.xlabel(self.Xname)
            plt.ylabel(self.Zname)
            plt.legend()
            # plt.savefig('D:/_Topological quantum computation/PbTe_Pb/S252/Optical images/' 
                        # + self.filename + '_{}.png'.format(idx_y_value))
            plt.show()
        return Tpf, Tnf, Iex_pf, Rn_pf, Iex_nf, Rn_nf
    
    def get_transparency_of_y(self, plot_Iex=False, plot=True):
        excess_result = []
        for i in range(self.y_len):
            excess_result.append(self.get_excess_current_at(idx_y_value=i, plot=plot_Iex))
        self.Tpf_box, self.Tnf_box, self.Iex_pf_box, self.Rn_pf_box, self.Iex_nf_box, self.Rn_nf_box = np.array(excess_result).T

        self.eIRDpf_box = self.Iex_pf_box * self.Rn_pf_box / self.gap
        self.eIRDnf_box = self.Iex_nf_box * self.Rn_nf_box / self.gap

        if plot:
            plt.plot(self.y_box, self.Tpf_box, 'o', markersize=2.5, c='r', label='positive fit')
            plt.plot(self.y_box, self.Tnf_box, 'o', markersize=2.5, c='b', label='negative fit')
            plt.xlabel(self.Yname)
            plt.ylabel('Transparency')
            plt.legend()
            plt.show()
        return None

    def Zequation(self, Z, eIRD):
        a = 2 * (1 + 2 * Z**2)
        b = np.arctanh(2 * Z * np.sqrt((1 + Z**2) / (1 + 6 * Z**2 + 4 * Z**4)))
        c = Z * np.sqrt((1 + Z**2) * (1 + 6 * Z**2 + 4 * Z**4))

        return a*b/c - 4/3 - eIRD

# -------------------------- Finding Ic(B) -------------------------------

    def write_Ic_box_in(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_Ic_box.txt'
        np.savetxt(file, np.vstack((self.y_box, self.Ic1_box, self.Ic2_box)).T)
        print('Ic_box are written in \"{}\"'.format(file))
        return None

    def read_Ic_box_from(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_Ic_box.txt'
        try:
            nothing, self.Ic1_box, self.Ic2_box = np.genfromtxt(file).T
            print('Ic_box etc. are extracted from \"{}\".'.format(file))
            return False
        except IOError:
            print( "Wrong! Ic_box file is not accessible, please use find_Ic_of_B().")
            return True

    def find_flattening_bias(self, left_cutoff, right_cutoff, y_value=None, idx_y_value=0, plot=True):
        if y_value is not None:
            idx_y_value = self.idx_y(y_value)
        else:
            y_value = self.y_box[idx_y_value]

        I = self.Xdata[idx_y_value]
        V = self.Zdata[idx_y_value]
        useful_slice = np.where((left_cutoff <= I) & (I <= right_cutoff))
        useful_I = I[useful_slice]
        useful_V = V[useful_slice]
        self.flattening_bias = interp1d(useful_I, useful_V, bounds_error=False, fill_value=0)

        if plot:
            plt.plot(I*self.x_scaling, V*self.y_scaling, lw=0.5, label='full linecut')
            plt.plot(useful_I*self.x_scaling, useful_V*self.y_scaling, marker='o', markersize=2, lw=0.5, label='flattening bias')
            plt.xlabel(self.Xname)
            plt.ylabel(self.Yname)
            plt.legend()
            plt.show()

        return None
    
    def flatten(self):
        self.Zdata_unflatten = self.Zdata.copy()
        self.Zdata = self.Zdata - self.flattening_bias(self.Xdata)
        return None
    
    def unflatten(self):
        self.Zdata = self.Zdata_unflatten
        return None

    def find_Ic(self, x, z, y=np.nan, bias_threshold=0.13, plot=False):
        '''
        For a given I-V curve, find Ic. 
        data must be a pandas dataFrame with two columns at least, one for I,  the other for V
        '''
        positive_slice = np.where(x >= 0)
        negative_slice = np.where(x <= 0)
        Ic1 = (x[negative_slice][ np.where(z[negative_slice] >= -bias_threshold) ])[0] # 这里为了紧凑写得比较套娃, 思路是
        Ic2 = (x[positive_slice][ np.where(z[positive_slice] <= bias_threshold) ])[-1] # 找左半边大于-threshold的第一个x和右半边小于threshold的最后一个x

        if plot:
            plt.plot(x, z)
            plt.axhline(-bias_threshold, c='m', alpha=0.25)
            plt.axhline(bias_threshold, c='k', alpha=0.25)
            plt.axvline(Ic1, c='m', alpha=0.25)
            plt.axvline(Ic2, c='k', alpha=0.25)
            plt.title('Ic1={:.3g}, Ic2={:.3g}, y={:.3g}'.format(Ic1, Ic2, y))
            plt.show()
        return Ic1, Ic2
    
    def find_Ic_of_B(self, bias_threshold=0.13, delta_bias=0, window_length=9, plot=True, plot_each=False,
                     alpha=1, vmin=None, vmax=None, cmin=0, cmax=1, gamma=1):
        self.Ic1_box = []
        self.Ic2_box = []
        self.remove_jumping_points()
        self.Zdata_smoothed = savgol_filter(self.Zdata, window_length, 0, axis=1)
        self.unremove_jumping_points()

        for i,y in enumerate(self.y_box):
            Ic1, Ic2 = self.find_Ic(self.Xdata[i], self.Zdata_smoothed[i] + delta_bias, y=y, 
                                    bias_threshold=bias_threshold,  plot=plot_each)
            self.Ic1_box.append(Ic1)
            self.Ic2_box.append(Ic2)
        self.Ic1_box = - np.array(self.Ic1_box)
        self.Ic2_box =   np.array(self.Ic2_box)

        if plot:
            cmap_generator = colormap.Colormap('D:/_Topological quantum computation/PbTe_Pb/S252/S252_data_analysis/Seismic.npy',
                                                min=cmin, max=cmax, gamma=gamma)
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
            axs[0].plot(self.Xdata.T*1e9, self.Zdata_smoothed.T + delta_bias, lw=0.3)
            axs[0].axhline(-bias_threshold, c='k', alpha=0.3, lw=0.5)
            axs[0].axhline( bias_threshold, c='k', alpha=0.3, lw=0.5)
            axs[0].set_xlabel('I(nA)')
            axs[0].set_ylabel('Vbias(mV)')
            axs[0].set_title(self.file + ', bias threshold={:.3g}mV, delta bias={:.3g}mV'.format(bias_threshold, delta_bias))
            
            ax0 = axs[1].pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                    cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
            ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(ax0, ax=axs[1], label=self.Zname)

            axs[1].plot(self.y_box*self.y_scaling,  self.Ic2_box*self.x_scaling, marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic2')
            axs[1].plot(self.y_box*self.y_scaling, -self.Ic1_box*self.x_scaling, marker='o', markersize=1, lw=0.5, alpha=alpha, label='Ic1')
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel(self.Xname)
            axs[1].set_title(self.file + ', bias threshold={:.3g}mV, delta bias={:.3g}mV'.format(bias_threshold, delta_bias))
            axs[1].legend()
            # axs[1].ylim(0,500)
            # axs[1].plot(y_list, Ic2_list)
            plt.show()

        return None

# -------------------------- Numerical calculation -------------------------------
    
    def use_Ic(self, use, read_from_file=True, plot=True, vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, alpha=1, figsize=(8,4)):
        if read_from_file: self.read_Ic_box_from()
        self.use = use
        if use == 1:
            self.Ic_box = self.Ic1_box
        elif use == 2:
            self.Ic_box = self.Ic2_box

        if plot:
            self.plot_in_colormap(boxes=[self.Ic2_box, -self.Ic1_box], labels=['Ic2', 'Ic1'],
                                  vmin=vmin, vmax=vmax, cmin=cmin, cmax=cmax, gamma=gamma, alpha=alpha, figsize=figsize)

        return None
    
    def get_device_parameters(self, width, penetration_depth, length, r_box=None, r_extend=3):
        '''
        注意! 在此之前的所有单位制都是SI, 但从这里开始, 长度的单位为um, 磁场的单位是Gauss, 
        也就是说所有使用到self.y_box和self.y_step的地方都要乘1e4将Tesla转化为Gauss
        '''
        self.width = width
        self.penetration_depth = penetration_depth
        self.length = length
        self.true_length = 2*penetration_depth + length

        self.beta_box = 2 * np.pi * self.true_length * self.y_box*1e4 / self.Phi0 # 2\pi(2\lambda +d) B /Phi0
        self.Phi_box = self.true_length * width * self.y_box*1e4 # Magnetic flux
        self.u_box = self.Phi_box/self.Phi0
        if r_box is None: self.r_box = np.linspace(-self.width*r_extend, self.width*r_extend, round(self.y_len*2*r_extend))
        else: self.r_box = r_box

        self.delta_beta = 2 * np.pi * self.true_length * self.y_step*1e4 / self.Phi0
        self.delta_r = (self.r_box.max() - self.r_box.min()) / (len(self.r_box)-1)

        self.b_box = self.beta_box[1:-1] - self.delta_beta/10 # generate b_box slightly mismatched with beta_box to avoid infinity in integrand

        self.Ic_func = interp1d(self.beta_box, self.Ic_box)

        return None

# --------------------------------- Even-odd FFT ---------------------------------

    def write_interfere_yn_in(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_interfere_yn.txt'
        np.savetxt(file, np.vstack((self.interfere_idx_yn, self.interfere_yn)).T, fmt=['%d', '%e'])
        print('interfere_yn are written in \"{}\"'.format(file))
        return None

    def read_interfere_yn_from(self, file=None):
        if file is None:
            file = self.directory + self.filename + '_interfere_yn.txt'
        try:
            interfere_idx_yn, self.interfere_yn = np.genfromtxt(file).T
            self.interfere_idx_yn = interfere_idx_yn.astype(int)
            print('interfere_yn etc. are extracted from \"{}\".'.format(file))
            return False
        except IOError:
            print( "Wrong! interfere_yn file is not accessible, please use find_interfere_yn().")
            return True
        
    def find_interfere_yn(self, idx_yn=[], plot=True, figsize=(8,4)):
        self.interfere_yn = self.y_box[idx_yn]
        self.interfere_idx_yn = np.array(idx_yn)
        
        if plot:
            plt.figure(figsize=figsize)
            plt.plot(self.y_box*1e3, -self.Ic1_box*1e9, marker='o', markersize=2, lw=0.5, label='Ic1')
            plt.plot(self.y_box*1e3,  self.Ic2_box*1e9, marker='o', markersize=2, lw=0.5, label='Ic2')
            for i,y in enumerate(self.interfere_yn):
                plt.axvline(y*1e3, label='idx={:.3g}, y={:.3g}'.format(self.interfere_idx_yn[i], y), alpha=0.5, lw=0.8)
            plt.xlabel('$B_{\perp}$(mT)')
            plt.ylabel('$I_c$(nA)')
            plt.title(self.file)
            plt.legend(bbox_to_anchor=(1, 1), loc=2, prop={'size': 8})
            plt.show()
        return None

    def get_flip_array(self):
        flip_array = np.ones(self.y_len) 

        if (len(self.interfere_yn)//2)%2 == 1: # parity of the number of self.interfere_idx_yn decides the first sign of the flip function
            start = 0
        else:
            start = 1

        for i in range(len(self.interfere_idx_yn)):
            if i%2 == start:
                flip_array[self.interfere_idx_yn[i-1]:self.interfere_idx_yn[i]] = -1

        if flip_array[np.argmin(np.abs(self.y_box - 0))] == -1: # make sure the central lobe is positive
            flip_array *= -1

        self.flip_array = flip_array

        return flip_array
    
    def get_CS_beta(self, parity=1, plot=True):
        '''
        interfere_yn的数目必须关于零点左右对称
        '''
        self.get_flip_array()
        self.C_beta = self.Ic_box*self.flip_array*parity

        S_beta_interp_func = interp1d(self.beta_box[self.interfere_idx_yn], self.C_beta[self.interfere_idx_yn], bounds_error=False, fill_value=0)
        self.S_beta = S_beta_interp_func(self.beta_box)

        if plot:
            plt.plot(self.u_box, self.C_beta*1e9, marker='o', markersize=2, lw=0.5, label='even(real) part')
            plt.plot(self.u_box, self.S_beta*1e9, marker='o', markersize=2, lw=0.5, label='odd(imaginary) part')
            # plt.plot(self.u_box, np.sqrt(self.S_beta**2 + self.C_beta**2), marker='o', markersize=2, lw=0.5, label='recovered Ic(B)')
            # plt.plot(self.u_box[self.interfere_idx_yn], self.C_beta[self.interfere_idx_yn], marker='o', markersize=2, lw=0.5)
            plt.xlabel('$\Phi/\Phi_0$')
            plt.ylabel('I(nA)')
            plt.title(self.file + ', use Ic{}'.format(self.use))
            plt.legend()
            plt.show()
        return None
     
    def get_Je_r(self, r=None, plot=True):
        if not r is None: # 如果没指定要计算的坐标空间的范围r, 就用默认的self.r_box; 给了r就把self.r_box变成r
            self.r_box = r
            self.delta_r = (r.max() - r.min()) / (len(r)-1)

        x = self.beta_box
        y = self.r_box
        delta_x = self.delta_beta
        
        N = len(x)
        M = len(y)
        X = np.tile(x, (M, 1))
        Y = np.tile(y, (N, 1)).T
        CX = np.tile(self.C_beta, (M, 1))
        delta_X = np.tile(delta_x, (N, 1))
        f_X_Y = (1/(2*np.pi)) * CX * np.cos(X*Y)
        # f_X_Y = (1/(2*np.pi)) * CX * np.exp(-1j*Y*X)
        # self.Je_r = np.abs(np.dot(f_X_Y, delta_X).reshape(-1))
        self.Je_r = np.dot(f_X_Y, delta_X).reshape(-1)
        
        self.Ic0_from_Je = np.dot(self.Je_r, self.delta_r*np.ones(M))
        if plot: 
            plt.plot(self.r_box/self.width, self.Je_r*1e9, marker='o', markersize=1, lw=0.5)
            plt.axvline(-0.5, c='k', ls='--', alpha=0.5)
            plt.axvline( 0.5, c='k', ls='--', alpha=0.5)
            plt.axhline( 0, c='k', ls='--', alpha=0.5)
            plt.title(self.file + ', Integrated Ic0 from Je={:.3g}nA'.format(self.Ic0_from_Je*1e9))
            plt.xlabel('r/width')
            plt.ylabel('Je(nA/um)')
            plt.show()

        return None
    
    def get_Jo_r(self, r=None, plot=True):
        if not r is None: # 如果没指定要计算的坐标空间的范围r, 就用默认的self.r_box; 给了r就把self.r_box变成r
            self.r_box = r
            self.delta_r = (r.max() - r.min()) / (len(r)-1)

        x = self.beta_box
        y = self.r_box
        delta_x = self.delta_beta       
        
        N = len(x)
        M = len(y)
        X = np.tile(x, (M, 1))
        Y = np.tile(y, (N, 1)).T
        SX = np.tile(self.S_beta, (M, 1))
        delta_X = np.tile(delta_x, (N, 1))
        f_X_Y = (1/(2*np.pi)) * SX * np.sin(X*Y)
        self.Jo_r = np.dot(f_X_Y, delta_X).reshape(-1)
        
        self.Ic0_from_Jo = np.dot(self.Jo_r, self.delta_r*np.ones(M)) 

        if plot: 
            plt.plot(self.r_box/self.width, self.Jo_r*1e9, marker='o', markersize=1, lw=0.5)
            plt.axvline(-0.5, c='k', ls='--', alpha=0.5)
            plt.axvline( 0.5, c='k', ls='--', alpha=0.5)
            plt.axhline( 0, c='k', ls='--', alpha=0.5)
            plt.title(self.file + ', Integrated Ic0 from Jo={:.3g}nA'.format(self.Ic0_from_Jo*1e9))
            plt.xlabel('r/width')
            plt.ylabel('Jo(nA/um)')
            plt.show()

        return None

    def get_Joe_r(self, plot=True, plot_CS_beta=False, plot_Je=False, plot_Jo=False, read_interfere_yn=True):
        if read_interfere_yn: self.read_interfere_yn_from()
        self.get_CS_beta(plot=plot_CS_beta)
        self.get_Je_r(plot=plot_Je)
        self.get_Jo_r(plot=plot_Jo)
        self.Joe_r = self.Je_r + self.Jo_r
        self.Ic0_from_Joe = self.Ic0_from_Jo + self.Ic0_from_Je
        if plot:
            plt.plot(self.r_box/self.width, self.Joe_r*1e9, marker='o', markersize=1, lw=0.5)
            # plt.xlim(-0.5, 0.5)
            plt.axvline(-0.5, ls='--', c='k', alpha=0.25)
            plt.axvline( 0.5, ls='--', c='k', alpha=0.25)
            plt.axhline( 0, c='k', ls='--', alpha=0.25)
            plt.fill_between(self.r_box/self.width, self.Joe_r*1e9, 0, facecolor='lightblue', alpha=0.5, interpolate=False)
            plt.title(self.file + ', Integrated Ic0 from J(r) = {:.3g}nA'.format(self.Ic0_from_Joe*1e9) + ', use Ic{}'.format(self.use))
            plt.xlabel('r/width')
            plt.ylabel('J(nA/um)')
            plt.show()
        return None

# ----------------------------- DHT + DFT ---------------------------------
        
    def Fourier_integral(self, x, y, delta_x, f_x):
        '''
        一般意义的傅里叶积分, g(y) = (1/2*pi) int f_x exp(-iyx) dx
        '''
        N = len(x)
        M = len(y)
        X = np.tile(x, (M, 1))
        Y = np.tile(y, (N, 1)).T
        f_X = np.tile(f_x, (M, 1))
        delta_X = np.tile(delta_x, (N, 1))

        f_X_Y = (1/(2*np.pi)) * f_X * np.exp(-1j * (X*Y))

        return np.abs(np.dot(f_X_Y, delta_X)).reshape(-1)


    def get_J_r_IFT(self, plot=True, plot_CS_beta=False, read_interfere_yn=True):
        '''
        IFT: integral Fourier transform;
        this is Even-odd method with J(beta) = C(beta) + i S(beta), then apply the Fourier integral to get J(r)
        '''
        if read_interfere_yn: self.read_interfere_yn_from()
        self.get_CS_beta(plot=plot_CS_beta)
        self.J_beta = self.C_beta + 1j*self.S_beta

        self.J_r_IFT = self.Fourier_integral(self.beta_box, self.r_box, self.delta_beta, self.J_beta)
    
        if plot:
            plt.plot(self.r_box/self.width, self.J_r_IFT*1e9, marker='o', markersize=1, lw=0.5)
            plt.axvline(-0.5, ls='--', c='k', alpha=0.25)
            plt.axvline( 0.5, ls='--', c='k', alpha=0.25)
            plt.axhline( 0, c='k', ls='--', alpha=0.25)
            plt.fill_between(self.r_box/self.width, self.J_r_IFT*1e9, 0, facecolor='lightblue', alpha=0.5, interpolate=False)
            plt.title(self.file + ', use Ic{}'.format(self.use))
            plt.xlabel('r/width')
            plt.ylabel('J(nA/um)')
            plt.show()
        
        return None

# ----------------------------- DHT + DFT ---------------------------------
# To be completed... lots of bugs.

    def delta(self, x):
        return (x[-1]-x[0]) / (len(x)-1)

    def get_DHT_parameters(self, width, penetration_depth, length, N=2**11):
        '''
        注意! 在此之前的所有单位制都是SI, 但从这里开始, 长度的单位为um, 磁场的单位是Gauss, 
        也就是说所有使用到self.y_box和self.y_step的地方都要乘1e4将Tesla转化为Gauss
        注意! 仅能用于磁场对称的数据！
        '''
        self.width = width
        self.penetration_depth = penetration_depth
        self.length = length
        self.true_length = 2*penetration_depth + length
        self.N = N

        self.B_box = np.linspace(self.y_box.min(), self.y_box.max(), self.N)
        self.beta_box = 2 * np.pi * self.true_length * self.B_box*1e4 / self.Phi0 # 2\pi(2\lambda +d) B /Phi0
        self.Phi_box = self.true_length * width * self.B_box*1e4 # Magnetic flux
        self.u_box = self.Phi_box/self.Phi0

        self.delta_B = self.delta(self.B_box)
        self.delta_beta = 2 * np.pi * self.true_length * self.delta_B*1e4 / self.Phi0
        
        # 根据FFT采样点数和采样精度生成x轴
        self.r_box = 2*np.pi*fftshift(fftfreq(self.N, self.delta_beta))
        self.delta_r = self.delta(self.r_box)

        # 插值Ic, 提高计算精度
        self.Ic_func = interp1d(self.y_box, self.Ic_box)

        return None
    
    def get_J_r_DHT(self, plot=True, xlim=(None, None)):
        # phase_factor = np.exp(-1j*self.beta_box[0]*2*np.pi*np.arange(self.N)/self.delta_beta/self.N)
        # self.J_r_DHT = self.delta_beta/(2*np.pi) * np.abs(fftshift(fft(np.exp(hilbert(np.log(self.Ic_func(self.beta_box)))))))
        self.J_r_DHT = self.delta_beta/(2*np.pi) * fftshift(fft(np.exp(hilbert(np.log(self.Ic_func(self.B_box))))))
        self.J_r = np.abs(self.J_r_DHT)
        if plot:
            plt.plot(self.r_box/self.width, self.J_r*1e9)
            plt.axvline(0, ls='--', c='k', alpha=0.25)
            plt.axvline(1, ls='--', c='k', alpha=0.25)
            plt.xlim(*xlim)
            plt.title(self.file + ', use Ic{}'.format(self.use))
            plt.xlabel('x(width)')
            plt.ylabel('J(nA/um)')
            plt.show()
        return None

# ----------------------------- iterative Ic(B) ---------------------------------

    def calculate_Ic_B(self, x, y, J_r, delta_x):
        '''
        x是空间座标, y是beta, J_r是电流密度分布, 均为行向量
        返回Ic(B), 是一个列向量
        '''
        N = len(x)
        M = len(y)
        X = np.tile(x, (M, 1))
        Y = np.tile(y, (N, 1)).T
        J_X = np.tile(J_r, (M, 1))
        delta_X = np.tile(delta_x, (N, 1))

        f_X_Y = J_X * np.exp(1j * (X*Y))

        return np.abs(np.dot(f_X_Y, delta_X)).reshape(-1)

    def get_Ic_B(self, method='hilbert', plot=True, figsize=(8,4)):
        if method == 'hilbert':
            J = self.J_r
            B = self.B_box # DHT会插值生成一个和y_box点数不同的磁场序列
        elif method == 'evenodd':
            J = self.Joe_r
            B = self.y_box
        else:
            print('method must be one of these: \'hilbert\' or \'evenodd\'!')
            return None
        
        self.Ic_B = self.calculate_Ic_B(self.r_box, self.beta_box, J, self.delta_r)

        if plot:
            plt.figure(figsize=figsize)
            plt.plot(self.y_box*self.y_scaling, self.Ic_box*self.x_scaling, label='original')
            # plt.xlabel(self.Yname)
            # plt.ylabel(self.Xname)
            # plt.legend()
            # plt.show()
            plt.plot(B*self.y_scaling, self.Ic_B*self.x_scaling, label='regenerated')
            plt.title(self.file + ', use Ic{}'.format(self.use) + ', {} method'.format(method))
            plt.xlabel(self.Yname)
            plt.ylabel(self.Xname)
            plt.legend()
            plt.show()
        return None
        

# to be realized functions:
# delta true zero bias? 注意在找Ic的时候没有用self.delta_true_zero_bias，而是外部指定了一个delta bias; 后面搞清楚物理之后需要修改这里
# automatically identify colomn names
# Gate vs. bias 2D 
# Stitch

class dataFile_JJ_4T(dataFile_JJ):
    def __init__(self, directory, file, filter_bias, R_filter=None, Rc=0, delta_true_zero_bias=0, excitation=20e-6,
                 column_x_index=4, column_y_index=1, column_z_index=2, current_column_index=3, reference_column=1, 
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1):
        
        super().__init__(directory, file, filter_bias, Rc, delta_true_zero_bias,
                         column_x_index, column_y_index, column_z_index, current_column_index, reference_column, 
                         Xname, Yname, Zname, x_scaling, y_scaling, z_scaling)
        self.Zdata = self.Zdata*1e3 # 为了把bias的标准单位修正到1mV
        return None

class dataFile_JJ_dIdV(dataFile_JJ):
    def __init__(self, directory, file, filter_bias, R_filter=None, Rc=0, delta_true_zero_bias=0, excitation=20e-6,
                 column_x_index=4, column_y_index=1, column_z_index=2, current_column_index=3, reference_column=1, 
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1):
        
        super().__init__(directory, file, filter_bias, Rc, delta_true_zero_bias,
                         column_x_index, column_y_index, column_z_index, current_column_index, reference_column, 
                         Xname, Yname, Zname, x_scaling, y_scaling, z_scaling)
        self.excitation = excitation
        self.R_filter = R_filter
        self.Xdata_calibrate = self.func_calibrate_sample_bias(self.Xdata, self.Current_data, self.Rc, self.delta_true_zero_bias)
        self.xmax_claibrate = self.Xdata_calibrate.max()
            
    def calculate_calibrate_G(self):
        self.Zdata_uncalibrated = self.Zdata.copy()
        self.Zdata = 1 / (1e6*self.excitation/self.Zdata - self.Rc - self.R_filter(self.Current_data)) / 7.748e-5 # in unit of 2e2/h
        return None
    
    def uncalculate_uncalibrate_G(self):
        self.Zdata = self.Zdata_uncalibrated
        return None

    def calibrate_sample_bias(self):
        '''
        For dI/dV 2D, V is Xdata, I is Current_data
        '''
        self.Xdata = self.func_calibrate_sample_bias(self.Xdata, self.Current_data, self.Rc, self.delta_true_zero_bias)
        return None

    def interp(self, x_interp=None, multiplier=2, interp_kind='linear', filter_bias=None, calibrate_sample_bias=True):
        '''
        For dI/dV 2D, x is bias, z is dI/dV, current is out of xyz.
        interpolating X,Y,Z data. x_interp and filter bias can be specified outside; if not given, use that from itself.
        '''
        if calibrate_sample_bias:
            Xdata = self.Xdata_calibrate
            xmax = self.xmax_claibrate
        else:
            Xdata = self.Xdata
            xmax = self.xmax
        
        if x_interp is None: # if x_interp is not given, then use self x_interp
            x_interp=np.linspace(-xmax, xmax, self.x_len*multiplier)

        if filter_bias is None:
            filter_bias = self.filter_bias

        self.x_interp = x_interp
        self.Zdata_interp = np.full((self.y_len, len(x_interp)), np.nan)
        self.Xdata_interp = np.tile(x_interp, (self.y_len, 1))
        self.Ydata_interp = np.tile(self.y_box, (len(x_interp), 1)).T

        for i,x in enumerate(Xdata):
            z = self.Zdata[i].copy() # copy a zdata
            z_interp = interp1d(x, z, kind=interp_kind, bounds_error=False, fill_value=np.nan)(x_interp) # interpolation
            self.Zdata_interp[i] = z_interp
            
        return None   

    def crop(self, ymin=None , ymax=None):
        '''
        only y axis can be cropped for the original data.
        '''
        if ymin is None:
            ymin = self.y_box.min()
        if ymax is None:
            ymax = self.y_box.max()
        y_slice = np.where((ymin <= self.y_box) & (self.y_box <= ymax))[0]
        
        self.Xdata = self.Xdata[y_slice]
        self.Ydata = self.Ydata[y_slice]
        self.Zdata = self.Zdata[y_slice]
        self.Current_data = self.Current_data[y_slice]

        self.y_box = self.y_box[y_slice]
        self.y_len = len(self.y_box)

        return None
    
    # def find_Ic(self, x, z, y=np.nan, bias_threshold=0.13, plot=False):
    #     '''
    #     For a given I-V curve, find Ic. 
    #     data must be a pandas dataFrame with two columns at least, one for I,  the other for V
    #     '''
    #     Ic1 = (x[ np.where(z <= -bias_threshold) ])[-1]
    #     Ic2 = (x[ np.where(z >= bias_threshold) ])[0]

    #     if plot:
    #         plt.plot(x, z)
    #         plt.axhline(-bias_threshold, c='m', alpha=0.25)
    #         plt.axhline(bias_threshold, c='k', alpha=0.25)
    #         plt.axvline(Ic1, c='m', alpha=0.25)
    #         plt.axvline(Ic2, c='k', alpha=0.25)
    #         plt.title('Ic1={:.3g}, Ic2={:.3g}, y={:.3g}'.format(Ic1, Ic2, y))
    #         plt.show()
    #     return Ic1, Ic2

    
# ----------------------------- Hilber transform ---------------------------------

    # def calculate_theta_b(self, x, y, delta_x, Ic_func):
    #     '''
    #     x, y, Ic都是行向量
    #     x代表beta, y代表b, delta_x代表delta_beta
    #     Ic_func是用Ic_box和beta_box插值出来的函数, 之所以要插值是因为, 为了保证积分计算的精度, beta向量和b向量取的点数得足够密集, 而一般测量数据是不太密集的
    #     y不能等于x, 虽然他们在物理上代表同样的变量beta, 但是这会使计算出现无穷大。必须使x和y叉开一点点
    #     返回theta_b, 是一个行向量
    #     '''
    #     Ic_x = Ic_func(x)
    #     Ic_y = Ic_func(y)
    #     N = len(x)
    #     M = len(y)
    #     X = np.tile(x, (M, 1))
    #     Y = np.tile(y, (N, 1)).T
    #     Ic_X = np.tile(Ic_x, (M, 1))
    #     Ic_Y = np.tile(Ic_y, (N, 1)).T
    #     delta_X = np.tile(delta_x, (N, 1))

    #     f_X_Y = (Y/np.pi) * ( np.log(Ic_X) -  np.log(Ic_Y)) / ( Y**2 - X**2 )
    #     # f_X_Y = (1/(2*np.pi)) * ( np.log(Ic_X) -  np.log(Ic_Y)) / ( Y**2 - X**2 )

    #     return np.dot(f_X_Y, delta_X).reshape(-1)

    # def get_theta_b(self, plot=True):
    #     self.theta_b = self.calculate_theta_b(self.beta_box, self.b_box, self.delta_beta, self.Ic_func)
    #     if plot:
    #         plt.plot(self.b_box, self.theta_b)
    #         plt.xlabel('beta')
    #         plt.ylabel('theta(rad)')
    #         plt.title(self.file + ', use Ic{}'.format(self.use))
    #         plt.show()
    #     return None

    # def calculate_J_r(self, x, y, delta_x, theta_x, Ic_func, width):
    #     '''
    #     x代表beta, y代表空间坐标r, delta_x是delta_beta, theta_x是theta_b
    #     返回J(r), 是一个行向量
    #     '''
    #     Ic_x = Ic_func(x)
    #     N = len(x)
    #     M = len(y)
    #     X = np.tile(x, (M, 1))
    #     Y = np.tile(y, (N, 1)).T
    #     Ic_X = np.tile(Ic_x, (M, 1))
    #     theta_X = np.tile(theta_x, (M, 1))
    #     delta_X = np.tile(delta_x, (N, 1))
    #     # print(np.shape(X), np.shape(Y), np.shape(Ic_X), np.shape(theta_X))
        
    #     f_X_Y = (1/(2*np.pi)) * Ic_X * np.exp(1j* theta_X - 1j*X*width/2) * np.exp(-1j * (X*Y))

    #     return np.abs(np.dot(f_X_Y, delta_X)).reshape(-1)
    #     # return np.real(np.dot(f_X_Y, delta_X)).reshape(-1)

    # def get_J_r(self, plot_theta=True, plot_J=True):
    #     self.get_theta_b(plot=plot_theta)
    #     self.J_r = self.calculate_J_r(self.b_box, self.r_box, self.delta_beta, 
    #                                   self.theta_b, self.Ic_func, self.width)
        
    #     self.Ic0_from_Jr = np.dot(self.J_r, np.ones(len(self.J_r)) * self.delta_r)

    #     if plot_J:
    #         plt.plot(self.r_box/self.width, self.J_r*1e9)
    #         plt.axvline(-0.5, ls='--', c='k', alpha=0.25)
    #         plt.axvline( 0.5, ls='--', c='k', alpha=0.25)
    #         plt.title(self.file + ', Integrated Ic0 from J(r) = {:.3g}nA'.format(self.Ic0_from_Jr*1e9) + ', use Ic{}'.format(self.use))
    #         plt.xlabel('x(width)')
    #         plt.ylabel('J(nA/um)')
    #         plt.show()

    #     return None