import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import colormap

class dataFile(object):
    def __init__(self, directory, file,
                 column_x_index=2, column_y_index=1, column_z_index=3,
                 Xname=None, Yname=None, Zname=None, x_scaling=1, y_scaling=1, z_scaling=1,
                 init_data_shape=None,
                 plot=True,):
        '''
        原则:
        - 除了bias的数据单位默认为mV(由于1000:1分压)以外其他物理量都是标准单位制
        - process data 模块前的所有函数不会对数据做任何处理, 只读取和整理数据
        - 处理任何数据都是append一列新的数据到self.all_data, 并更新表头、comments信息; 如果需要, 手动改变XYZ轴
        - 为了使用索引方便而引入了第0列空数据和列名, 但这导致了len(self.all_data) = len(self.column_names) = self.num_columns() + 1
        - 一旦读取完数据之后, 所有需要的数据信息都随时从数据本身获取(例如 self.y_box() ), 而不再提前指定好,
          这是为了避免对数据做各种处理时还要考虑数据信息的更新
        - 所有的处理应该时刻保持 XYZdata 是指向all_data某列数据的指针
        '''
        self.directory = directory
        self.file = file
        self.file_name, self.file_suffix = os.path.splitext(file)
        self.file_path = os.path.join(directory, file)

        self.column_x_index = column_x_index 
        self.column_y_index = column_y_index
        self.column_z_index = column_z_index

        # 提取数据, 提取表头所有的注释并从中读取data shape和列名, 并据此格式化每一列数据
        self.df = pd.read_csv(self.file_path, delimiter='\t', comment='#', header=None)
        self.init_num_columns = self.df.shape[1]
        self.get_comments()
        self.get_column_names()

        self.get_init_data_shape() 
        if init_data_shape is not None: self.init_y_len, self.init_x_len = init_data_shape
        self.init_data_shape = (self.init_y_len, self.init_x_len)
        self.format_all_data()

        # 如果制定了name就按照指定的来, 如果没有指定, 前面的self.get_column_names()已经自动识别了name
        if Xname is not None: 
            self.Xname = Xname
        if Yname is not None:
            self.Yname = Yname
        if Zname is not None:
            self.Zname = Zname

        # scaling用于画图时对数据进行缩放
        self.x_scaling = x_scaling
        self.y_scaling = y_scaling
        self.z_scaling = z_scaling
        
        self.Phi0 = 20.6783 # Gauss*um^2, quantum flux
        self.G0 = 7.748e-5 # Siemens, 2e^2/h

        self.print_column_names()
        if plot: self.plot_heatmap()

# -------------------------------- initiate dataFile --------------------------------

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
        for i in range(self.init_num_columns): # comments for every column
            comments.append(self.get_comments_of_column(i + 1))

        self.comments = comments
        return None

    def get_init_data_shape(self):
        self.init_y_len = int(self.comments[1][6][8:])
        self.init_x_len = int(self.comments[2][6][8:])
        
        return None
    
    def get_column_names(self):
        # 放一个0列的列名, 这样方便后面对其列名, 第i列就是column_names[i]
        column_names = ['nan']

        # 下面从comments中得到列名
        for i in range(self.init_num_columns):
            column_names.append(self.comments[i+1][4][8:][:-1]) # [4]选出name所在行, [8:]去掉前缀, [:-1]去掉末尾的\n, 注意i+1
        
        # 分别特别命名XYZ的names
        self.Xname = column_names[self.column_x_index]
        self.Yname = column_names[self.column_y_index]
        self.Zname = column_names[self.column_z_index]
        self.column_names = column_names

        return None
    
    def format_all_data(self):
        '''
        从self.df中提取所有列的数据并格式化为init_data_shape
        '''
        all_data = []
        all_data.append(np.full(self.init_data_shape, np.nan)) # 放一个0列的数据, 这样方便后面对其列名, 第i列数据就是all_data[i]
        for i in range(self.init_num_columns):
            all_data.append(self.df.iloc[:, i].to_numpy().reshape(self.init_data_shape))

        self.all_data = all_data
        self.Xdata = self.all_data[self.column_x_index]
        self.Ydata = self.all_data[self.column_y_index]
        self.Zdata = self.all_data[self.column_z_index]

        return None

    def print_column_names(self):
        for i, string in enumerate(self.column_names):
            # 使用字符串的长度作为对齐宽度
            print(f"{i:^{len(string) + 1}}", end=" ")
        print()  # 换行

        print(", ".join(self.column_names))

        return None
 
# -------------------------------- get data info --------------------------------

    def y_box(self):
        return np.copy(self.Ydata[:, 0])
    
    def y_step(self):
        return (self.y_box().max() - self.y_box().min()) / (len(self.y_box())-1)

    def data_shape(self):
        return self.Xdata.shape
    
    def y_len(self):
        return self.data_shape()[0]
    
    def x_len(self):
        return self.data_shape()[1]

    def num_columns(self):
        return len(self.all_data) - 1

# -------------------------------- Basic functions --------------------------------

    def idx_y(self, y): # find the index of y in y_box
        return np.where(np.isclose(self.y_box(), y, atol=self.y_step()/10)==True)[0][0]
    
    def y_slice(self, y_list):
        '''
        给定几个y值, 生成这几个y值对应的索引y_slice
        '''
        y_slice = []
        for y in y_list:
            y_slice.append(self.idx_y(y))
        return y_slice

    def change_X(self, index):
        '''重新指定 X data 为某列数据'''
        # 有时候会使用负数索引, 例如-1, 这一句代码保证给出的column_x_index总是正数
        self.column_x_index = (index + len(self.all_data)) % len(self.all_data)
        self.Xdata = self.all_data[self.column_x_index]

        print('X axis is now [{}]:[{}]'.format(self.column_x_index, self.column_names[self.column_x_index]))

        return None

    def change_Y(self, index):
        '''重新指定 Y data 为某列数据'''
        self.column_y_index = (index + len(self.all_data)) % len(self.all_data)
        self.Ydata = self.all_data[self.column_y_index]

        print('Y axis is now [{}]:[{}]'.format(self.column_y_index, self.column_names[self.column_y_index]))

        return None

    def change_Z(self, index):
        '''重新指定 Z data 为某列数据'''
        self.column_z_index = (index + len(self.all_data)) % len(self.all_data)
        self.Zdata = self.all_data[self.column_z_index]

        print('Z axis is now [{}]:[{}]'.format(self.column_z_index, self.column_names[self.column_z_index]))
        
        return None

    def change_XYZ(self, column_x_index, column_y_index, column_z_index):
        '''重新指定 XYZ data 为某列数据'''
        
        # 先计算对应的正数索引, 再按照索引指定XYZ data; 注意到这里考虑了索引为负数的情况
        self.column_x_index = (column_x_index + len(self.all_data)) % len(self.all_data)
        self.column_y_index = (column_y_index + len(self.all_data)) % len(self.all_data)
        self.column_z_index = (column_z_index + len(self.all_data)) % len(self.all_data)
        self.Xdata = self.all_data[self.column_x_index]
        self.Ydata = self.all_data[self.column_y_index]
        self.Zdata = self.all_data[self.column_z_index]

        print('X axis is now [{}]:[{}]'.format(self.column_x_index, self.column_names[self.column_x_index]))
        print('Y axis is now [{}]:[{}]'.format(self.column_y_index, self.column_names[self.column_y_index]))
        print('Z axis is now [{}]:[{}]'.format(self.column_z_index, self.column_names[self.column_z_index]))

        return None
    
# ----------------------------------- plot -------------------------------------

    def croptblr(self, top=0, bottom=None, left=0, right=None):
        '''
        - crop是一个非常特殊的操作, 它不是数据处理, 不产生新的数据
        - crop应该放在所有的数据处理过程完成之后, 仅用于画图之前!
        '''
        # 由于取slice操作会重新分配内存地址, 导致一系列指针问题, 所以就直接重新搞一套all_data
        cropped_all_data = []
        for data in self.all_data:
            cropped_all_data.append(data[top:bottom, left:right])
        
        self.all_data = cropped_all_data
        self.Xdata = self.all_data[self.column_x_index]
        self.Ydata = self.all_data[self.column_y_index]
        self.Zdata = self.all_data[self.column_z_index]

        return None
    
    def interp(self, x_interp=None, multiplier=1, interp_kind='linear'):
        '''
        - interpolation是一个非常特殊的操作, 它不是数据处理, 不产生新的数据
        - interpolation应该放在所有的数据处理过程完成之后, 仅用于画图之前!
        '''
        if x_interp is None: # if x_interp is not given, then use self x_interp
            x_interp=np.linspace(np.min(abs(self.Xdata)), np.max(abs(self.Xdata)), self.x_len()*multiplier)

        self.x_interp = x_interp
        self.Zdata_interp = np.full((self.y_len(), len(x_interp)), np.nan)
        self.Xdata_interp = np.tile(x_interp, (self.y_len(), 1))
        self.Ydata_interp = np.tile(self.y_box(), (len(x_interp), 1)).T

        for i,x in enumerate(self.Xdata):
            z = self.Zdata[i].copy() # copy a zdata
            z_interp = interp1d(x, z, kind=interp_kind, bounds_error=False, fill_value=np.nan)(x_interp) # interpolation
            self.Zdata_interp[i] = z_interp

        self.Xdata, self.Xdata_uninterp = self.Xdata_interp, self.Xdata
        self.Ydata, self.Yata_uninterp = self.Ydata_interp, self.Ydata
        self.Zdata, self.Zdata_uninterp = self.Zdata_interp, self.Zdata

        return None

    def swap_axes(self):
        '''
        仅用作画图前后使用, 否则会导致数据处理出现逻辑性错误
        画图前后应各使用一次
        '''
        self.Xdata, self.Ydata = self.Ydata, self.Xdata
        self.Xname, self.Yname = self.Yname, self.Xname
        self.x_scaling, self.y_scaling = self.y_scaling, self.x_scaling
        return None

    def plot_heatmap(self, cmap='Seismic', vmin=None, vmax=None, cmin=0, cmax=1, gamma=1, figsize=(6,4), show=False, swap_axes=False):
        '''
        画出X,Y,Z的heatmap, 可指定colormap的一系列参数, 可指定画布大小
        若希望对图像做后续操作, 如添加辅助线等, 可以将show置为False(默认值), 并在函数外操作
        特别的, jupyter notebook的Cell结束时会自动show所有的图, 所以使用jupyter notebook时不用特别把show置为True
        '''
        if swap_axes: self.swap_axes()

        cmap_generator = colormap.Colormap(cmap + '.npy',
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

        if swap_axes: self.swap_axes()

        return fig, ax

    def plot_waterfall(self, z_shift, figsize=(3, 6), show=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
        for i, (x, z) in enumerate(zip(self.Xdata, self.Zdata)):
            ax.plot(x, z + i*z_shift, c='black', linewidth=0.5)
        ax.set_xlabel(self.Xname)
        ax.set_ylabel(self.Zname)
        ax.set_title(self.file)

        if show: plt.show()

        return fig, ax

    def plot_linecuts(self, y_slice=[], y_list=None, z_shift=0, 
                      mark_y=True, mark_alpha=0.5, show=False, figsize=(14, 4),
                      cmap='Seismic', xlim=(None, None), zlim=(None, None), vmin=None, vmax=None, cmin=0, cmax=1, gamma=1):
            '''
            - 请使用 range(m, n ,q) 或 [i, j, k] 来给出 y_slice
            - 若给出 y_list, 则会覆盖 y_slice
            '''
            if y_list is not None: y_slice = self.y_slice(y_list)

            cmap_generator = colormap.Colormap(cmap + '.npy',
                                                min=cmin, max=cmax, gamma=gamma)
            fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=100)
            fig.set_facecolor('white')
            
            y_box = self.y_box()
            for i, idx_y in enumerate(y_slice):
                c = 'C' + str(i) # unify colors
                axs[0].plot(self.Xdata[idx_y]*self.x_scaling, self.Zdata[idx_y]*self.z_scaling + idx_y*z_shift, 
                            label='y={:.3g}, idx_y={}'.format(y_box[idx_y], idx_y), lw=0.5, c=c)
                if mark_y: axs[1].axvline(y_box[idx_y]*self.y_scaling, ls='--', 
                                          label='idx_y = {}'.format(idx_y), lw=1, c=c, alpha=mark_alpha)
            ax0 = axs[1].pcolormesh(self.Ydata*self.y_scaling, self.Xdata*self.x_scaling, self.Zdata*self.z_scaling, 
                                    cmap=cmap_generator.get_mpl_colormap(), shading='nearest')
            axs[0].set_xlabel(self.Xname)
            axs[0].set_ylabel(self.Zname)
            axs[0].set_title(self.file)
            axs[1].set_xlabel(self.Yname)
            axs[1].set_ylabel(self.Xname)
            axs[1].set_title(self.file)
            axs[0].legend(bbox_to_anchor=(0, 1), loc=2, prop={'size': 8}, framealpha=0.3)

            axs[0].set_xlim(*xlim) # Zoom in
            axs[1].set_ylim(*xlim)
            axs[0].set_ylim(*zlim)
            ax0.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))

            fig.colorbar(ax0, ax=axs[1], label=self.Zname)

            if show: plt.show()

            return fig, axs

# ----------------------------- Write and read files -------------------------------
           
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

# ------------------------------- process data ---------------------------------

# Principal for this module:
# - functions before this module do not change/create/delete any data
# - 这个模块不修改原始数据, 处理产生的数据是新数据
# - 产生新数据后必须更新data info, 即更新all_data, column_names和comments
# - 产生新数据后需手动self.change_XYZ()

    def add_data_column(self, data, name):
        '''
        每次处理出新数据都要update数据信息
        '''
        # column信息
        self.all_data.append(data)
        self.column_names.append(name)

        self.print_column_names()

        # comments信息
        # to be completed
        
        return None

    def calculate_sample_bias(self, bias_column_idx, current_volt_column_idx, 
                              current_amplifier_factor, filter_bias, Rc, true_zero_bias):
        '''
        - 计算sample bias需要bias和current数据列, filter bias的插值函数, Rc和true zero bias的值
        '''
        # self.Rc = Rc
        # self.filter_bias = filter_bias
        # self.true_zero_bias = true_zero_bias
        
        # 计算sample bias
        bias = self.all_data[bias_column_idx]
        current = self.all_data[current_volt_column_idx] / current_amplifier_factor
        sample_bias_data = bias - filter_bias(current) - current*Rc*1e3 - true_zero_bias
        
        # 更新数据信息
        self.add_data_column(sample_bias_data, 'sample bias (mV)')

        return None

    def calculate_differential_conductance(self, sr1_X_column_idx, current_volt_column_idx, 
                                           excitation, current_amplifier_factor, R_filter, Rc):
        '''
        - excitation is usually 20e-6
        - current amplifier factor is usually 1e6
        '''
        sr1_X = self.all_data[sr1_X_column_idx]
        current = self.all_data[current_volt_column_idx] / current_amplifier_factor

        # 计算微分电阻和微分电导, 这样计算的好处是不会出现除以0的情况
        series_resistance = R_filter(current) + Rc
        di = sr1_X/current_amplifier_factor # A
        dv = excitation - series_resistance*di # V
        didv = di/dv/self.G0 # 2e^2/h


        # 更新数据信息
        self.add_data_column(didv, 'dI/dV (2e^2/h)')

        return None
    

# ***********************************************************************************************************************************

class filter_IV(dataFile):
    def __init__(self, directory, file, 
                 current_amplifier_factor,
                 current_volt_column_idx=1, bias_column_idx=1, 
                 plot=True):
        '''
        - 只用到了父类的__init__来读入数据和其他信息
        - 使用时应调用self.IV_func()和self.IR_func()来给出插值函数
        - 这里为了简单起见, 没有遵循dataFile的原则, 在__init__里就直接计算了新数据self.resistance
        '''
        # 首先依赖父类读取文件并初始化数据, 打印表头
        super().__init__(directory, file, plot=False)

        # 读取数据 I-V
        self.current = (self.all_data[current_volt_column_idx] / current_amplifier_factor).reshape(-1)
        self.bias = self.all_data[bias_column_idx].reshape(-1)

        # 数值微分计算 I-R
        self.resistance = np.gradient(self.bias, self.current)/1e3

        if plot: 
            self.plot_IV()
            self.plot_IR()
        return None
    
    def smooth_IR(self, sgfilter_window_length):
        '''sgfilter_window_length推荐31'''
        self.resistance = savgol_filter(self.resistance, sgfilter_window_length, 0)

        return None

    def IV_func(self):
        return interp1d(self.current, self.bias, bounds_error=False, fill_value=np.nan)
    
    def IR_func(self):
        return interp1d(self.current, self.resistance, bounds_error=False, fill_value=np.nan)

    def plot_IV(self):
        plt.plot(self.current*1e6, self.bias)
        plt.xlabel('I(uA)')
        plt.ylabel('V(mV)')
        plt.title(self.file)
        plt.show()
        return None
    
    def plot_IR(self):
        plt.plot(self.current*1e6, self.resistance)
        plt.xlabel('I(uA)')
        plt.ylabel('R(Ohm)')
        plt.title(self.file)
        plt.show()
        return None

    
class filter_IR(dataFile):
    def __init__(self, directory, file, 
                 excitation, current_amplifier_factor,
                 current_volt_column_idx=1, sr1_X_column_idx=1, 
                 plot=True):
        '''
        - 只用到了父类的__init__来读入数据和其他信息
        - 使用时应调用self.R_of_current_func()给出插值函数!
        '''

        super().__init__(directory=directory, file=file, plot=False)

        # 计算电阻
        sr1_X = self.all_data[sr1_X_column_idx].reshape(-1)
        dvdi = excitation/(sr1_X/current_amplifier_factor)

        # 赋值
        self.resistance = dvdi.reshape(-1)
        self.current = (self.all_data[current_volt_column_idx] / current_amplifier_factor).reshape(-1)

        if plot: self.plot_IR()
        return None
    
    def IR_func(self):
        return interp1d(self.current, self.resistance, bounds_error=False, fill_value=np.nan)

    def plot_IR(self):
        plt.plot(self.current*1e6, self.resistance)
        plt.xlabel('I(uA)')
        plt.ylabel('R(Ohm)')
        plt.title(self.file)
        plt.show()
        return None
    
    
class constant_resistance_filter(object):
    def __init__(self, resistance):
        self.resistance = resistance

    def IV_func(self):
        return lambda current: self.resistance*current*1e3 # mV
    
    def IR_func(self):
        return lambda current: 3700 # Ohm