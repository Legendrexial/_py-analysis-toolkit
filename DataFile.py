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

class dataFile(object):
    def __init__(self, directory, file,
                 column_x_index=2, column_y_index=1, column_z_index=3,
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1,
                 data_shape=None,
                 filter_bias=None, Rc=None, delta_true_zero_bias=None):
        '''
        除了bias的数据单位默认为mV(由于1000:1分压)以外其他物理量都是标准单位制
        __init__不会对数据做任何处理, 只读取和整理数据
        '''
        self.directory = directory
        self.file = file
        self.file_name, self.file_suffix = os.path.splitext(file)
        self.file_path = os.path.join(directory, file)  # str

        self.filter_bias = filter_bias
        self.delta_true_zero_bias = delta_true_zero_bias
        if self.read_Rc_from():
            self.Rc = Rc
            self.y_reference = np.nan

        self.column_x_index = column_x_index  # int e.g. 1 # for 4-terminal bias vs xx scan, this is Agilent_2 (sample bias)
        self.column_y_index = column_y_index  # int e.g. 2 
        self.column_z_index = column_z_index  # int e.g. 6

        self.df = pd.read_csv(self.file_path, delimiter='\t', comment='#', header=None)
        self.num_columns = self.df.shape[1]
        self.get_comments()
        self.get_shape_and_names() # 自动读取datashape, 当然如果给定datashape下面会强制覆盖
        if data_shape is not None: self.y_len, self.x_len = data_shape
        self.data_shape = (self.y_len, self.x_len)
        self.Xdata = self.df.iloc[:, column_x_index-1].to_numpy().reshape(self.data_shape)
        self.Ydata = self.df.iloc[:, column_y_index-1].to_numpy().reshape(self.data_shape)
        self.Zdata = self.df.iloc[:, column_z_index-1].to_numpy().reshape(self.data_shape)

        self.Xdata_original = self.Xdata.copy()
        self.Ydata_original = self.Ydata.copy()
        self.Zdata_original = self.Zdata.copy()

        self.xmax = np.max(abs(self.Xdata))
        self.xmin = np.min(abs(self.Xdata))
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

# ------------------------------- Basic functions -----------------------------------
    def reset(self):
        self.__init__(self.directory, self.file, self.filter_bias, self.Rc, self.delta_true_zero_bias,
                      self.column_x_index, self.column_y_index, self.column_z_index, self.current_column_index, self.reference_column,
                      self.Xname, self.Yname, self.Zname, self.x_scaling, self.y_scaling, self.z_scaling)
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
        self.Xdata, self.Ydata = self.Ydata, self.Xdata
        return None
    
    def xderiv(self):
        for i,x in enumerate(self.Xdata):
            self.Zdata[i] = np.gradient(self.Zdata[i], x)
        return None

    def yderiv(self):
        
        return None

    def plot_heatmap(self, vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, figsize=(6,4), show=False):
        '''
        画出X,Y,Z的heatmap, 可指定colormap的一系列参数, 可指定画布大小
        若希望对图像做后续操作, 如添加辅助线等, 可以将show置为False(默认值), 并在函数外操作
        特别的, jupyter notebook的Cell结束时会自动show所有的图, 所以使用jupyter notebook时不用特别把show置为True
        '''
        cmap_generator = colormap.Colormap('Seismic.npy',
                                            min=cmin, max=cmax, gamma=gamma)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
        ax0 = ax.pcolormesh(self.Xdata*self.x_scaling, self.Ydata*self.y_scaling, self.Zdata*self.z_scaling, 
                                cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
        ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(ax0, ax=ax, label=self.Zname)
        ax.set_xlabel(self.Xname)
        ax.set_ylabel(self.Yname)
        ax.set_title(self.file)

        # 如果想对图像做后续操作, 如添加辅助线等, 可以将show置为False, 并在函数外操作
        if show: plt.show()

        return fig, ax

    def interp(self, x_interp=None, multiplier=1, interp_kind='linear'):
        '''
        interpolating X,Y,Z data
        x axis can be specified.
        '''
        if x_interp is None: # if x_interp is not given, then use self x_interp
            x_interp=np.linspace(self.xmin, self.xmax, self.x_len*multiplier)

        self.x_interp = x_interp
        self.Zdata_interp = np.full((self.y_len, len(x_interp)), np.nan)
        self.Xdata_interp = np.tile(x_interp, (self.y_len, 1))
        self.Ydata_interp = np.tile(self.y_box, (len(x_interp), 1)).T

        for i,x in enumerate(self.Xdata):
            z = self.Zdata[i].copy() # copy a zdata
            z_interp = interp1d(x, z, kind=interp_kind, bounds_error=False, fill_value=np.nan)(x_interp) # interpolation
            self.Zdata_interp[i] = z_interp

        self.Xdata, self.Xdata_uninterp = self.Xdata_interp, self.Xdata
        self.Ydata, self.Yata_uninterp = self.Ydata_interp, self.Ydata
        self.Zdata, self.Zdata_uninterp = self.Zdata_interp, self.Zdata

        return None

    def idx_y(self, y): # find the index of y in y_box
        return np.where(np.isclose(self.y_box, y, atol=self.y_step/10)==True)[0][0]
    
    def y_list(self, y_slice):
        '''
        给定几个y值的索引, 生成这几个y值
        '''
        return self.y_box[y_slice]
    
    def y_slice(self, y_list):
        '''
        给定几个y值, 生成这几个y值对应的索引y_slice
        '''
        y_slice = []
        for y in y_list:
            y_slice.append(self.idx_y(y))
        return y_slice

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
                cmap_generator = colormap.Colormap('Seismic.npy',
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

# ------------------------------- Write and read files -----------------------------------

    def get_head_comments(self):
        comments = []
        with open(self.file_path, 'r') as file:
            for i in range(6):
                comments.append(file.readline())
        return comments

    def get_comments_of_column(self, n):
        comments = []
        start_line = 7 + 14*(n - 1) # 第一个column开头是第7行, 之后每个column占14行

        with open(self.file_path, 'r') as file:
            # start line前面的行不读
            for i in range(start_line - 1):
                file.readline()
            # 从start line开始往后读14行
            for i in range(14):
                comments.append(file.readline())
        
        return comments
    
    def get_comments(self):
        comments = []
        
        # 读取全部表头注释
        comments.append(self.get_head_comments()) # head comments
        for i in range(self.num_columns): # comments for every column
            comments.append(self.get_comments_of_column(i + 1))

        self.comments = comments
        return None

    def get_shape_and_names(self):
        lines = self.comments
        self.y_len = int(lines[1][6][8:])
        self.x_len = int(lines[2][6][8:])
        self.Xname = lines[self.column_x_index][4][8:][:-1] # [4]选出name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n
        self.Yname = lines[self.column_y_index][4][8:][:-1] # [4]选出name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n
        self.Zname = lines[self.column_z_index][4][8:][:-1] # [4]选出name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n
        
        return None
    
    def write_Rc_in(self, file=None):
        if file is None:
            file = self.directory + self.file_name + '_Rc.txt'
        with open(file, 'w') as f:
            f.write(str(self.Rc) + '\n' + str(self.y_reference))
            f.close()
            print('Rc={} and y reference {} are written in \"{}\"'.format(self.Rc, self.y_reference, file))
        return None

    def read_Rc_from(self, file=None): # 如果成功读取
        if file is None:
            file = self.directory + self.file_name + '_Rc.txt'
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

    def save_dat_processed(self, directory=None, label='processed'):
        if directory is None:
            directory = self.directory

        file_name = self.file_name + '_' + label
        file_path =  os.path.join(directory, file_name + '.dat')  # str

        # 将二维矩阵转换为一维数组
        X_flat = self.Xdata.flatten()
        Y_flat = self.Ydata.flatten()
        Z_flat = self.Zdata.flatten()

        # 按列组合数据
        output_data = np.column_stack((Y_flat, X_flat, Z_flat))

        # 写入新文件
        with open(file_path, 'w') as file:
            # 先写入表头和对应数据列的comments
            for comment in self.comments[0]:
                file.write(comment)
            for comment in self.comments[self.column_y_index]:
                file.write(comment)
            for comment in self.comments[self.column_x_index]:
                file.write(comment)
            for comment in self.comments[self.column_z_index]:
                file.write(comment)

            # 再写入数据
            np.savetxt(file, output_data, fmt='%.6f', delimiter='\t')
        print('Data is saved in {}'.format(file_path))

        return None

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
