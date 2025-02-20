import os
import time
import math
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.constants as consts
from scipy import ndimage


class DatFile(object):
    """
    除了Island Analyasis方法为针对特定device测量数据的专门分析方法之外，其他的方法可以用于general的colormap类型数据分析（前提是.DAT文件是可以被QTPLOT给打开的格式）。
    """

    def __init__(self,
                 filename,
                 directory,
                 column_x_index=1,
                 column_y_index=2,
                 column_z_index=4,
                 reference_column=1,
                 circuit_current_column=3,
                 reshape=True):
        """
        该类为对原始的.dat文件的相关操作.Filename为包含后缀的完整名字.
        假设有201个不同的个x值，121个不同的y值，经过剪裁cropped之后变为100个x值，50个y值

        属性描述：
        坐标x所在datafile的列数，默认Column 1 为 x坐标 e.g. PG
        坐标y所在datafile的列数，默认Column 2 为 y坐标 e.g. B
        坐标z（也就是data）所在datafile的列数，默认Column 6 为 z坐标 e.g. Conductance

        坐标x那一列的所有值，包含每遍历一个y值所重复的所有数据 e.g. 第一列的201x121个元素。
        坐标y那一列的所有值。e.g. 第二列的201x121个元素。
        坐标z那一列的所有值。e.g. 第六列的201x121个元素
        返回数据矩阵，以上的xyz data都包含每一个小方块之间的空格

        返回x，y的尺寸（不是所有的数据的尺寸，而是有多少个不同的个x和多少个不同的个y）

        返回xy的最大和最小值，即xy从哪里开始到哪里结束

        文件名

        经过剪裁之后的XYZDATA，size分别为 1x100 1x50 50x100 可以直接作图colormap，无任何其他处理
        """

        self.file_path = os.path.join(directory, filename)  # str
        self.filename = filename

        self.column_x_index = column_x_index  # int e.g. 1
        self.column_y_index = column_y_index  # int e.g. 2
        self.column_z_index = column_z_index  # int e.g. 6
        self.reference_column = reference_column  # int 1 by default
        self.circuit_current_column = circuit_current_column

        self.Circut_current = self.__ObtainDataMatrix(
        )[:, self.circuit_current_column - 1]  #用于两端法测量时计算出样品上的分压所使用。
        self.XReference = self.__ObtainDataMatrix(
        )[:, self.reference_column -
          1]  #扫描x轴时候的参考点，等间距的扫描点。比如4terminal器件，reference为外加的等间距的bias，xdata就为在器件上的非等间距的bias
        self.XData = self.__ObtainDataMatrix(
        )[:,
          self.column_x_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.YData = self.__ObtainDataMatrix(
        )[:,
          self.column_y_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.ZData = self.__ObtainDataMatrix(
        )[:,
          self.column_z_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.DataMatrix = self.__ObtainDataMatrix(
        )  # <class 'numpy.ndarray'>; Shape = (24442, 14)

        self.XReference_RAW = self.__ObtainDataMatrix(
        )[:, self.reference_column -
          1]  #扫描x轴时候的参考点，等间距的扫描点。比如4terminal器件，reference为外加的等间距的bias，xdata就为在器件上的非等间距的bias
        self.XData_RAW = self.__ObtainDataMatrix(
        )[:,
          self.column_x_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.YData_RAW = self.__ObtainDataMatrix(
        )[:,
          self.column_y_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.ZData_RAW = self.__ObtainDataMatrix(
        )[:,
          self.column_z_index - 1]  # <class 'numpy.ndarray'>; Shape = (24442,)
        self.DataMatrix_RAW = self.__ObtainDataMatrix(
        )  # <class 'numpy.ndarray'>; Shape = (24442, 14)

        self.XSize = self.__VariableSizeObtain_unique()[0]  # int e.g. 201
        self.YSize = self.__VariableSizeObtain_unique()[1]  # int e.g. 121
        self.XStart = self.__VariableSizeObtain_unique()[2]  # -1019mV
        self.YStart = self.__VariableSizeObtain_unique()[3]  # 0T
        self.XEnd = self.__VariableSizeObtain_unique()[4]  # -1007mV
        self.YEnd = self.__VariableSizeObtain_unique()[5]  # 0.6T
        self.XStep = self.__VariableSizeObtain_unique()[6]
        self.YStep = self.__VariableSizeObtain_unique()[7]

        self.XName = self.__VariableSizeObtain_text()[6]
        self.YName = self.__VariableSizeObtain_text()[7]
        self.ZName = self.__VariableSizeObtain_text()[8]

        print('\'__init__\':Module has been loaded')

    def reshape(self):
        # 用于把本来是按照列向量排列的数据变成依照y值排列的矩阵形式（row=corected_y_size,col=x_uni_size），并且在中途停止扫描的时候用nan补全数据
        x_uni_size = np.shape(np.unique(self.XReference))[0]
        corrected_y_size = math.ceil(len(self.XData) / x_uni_size)

        if x_uni_size * corrected_y_size != len(
                self.XData):  # 如果在中途停止了扫描，则数据点不全，需要用nan补齐后再进行reshape
            print(
                '\'reshape\':The last scan is not completed, fill the lost scan with NaN'
            )
            fill_array = np.full(
                x_uni_size * corrected_y_size - len(self.XData), np.nan)
            self.XData = np.concatenate((self.XData, fill_array))
            self.YData = np.concatenate((self.YData, fill_array))
            self.ZData = np.concatenate((self.ZData, fill_array))
            self.XReference = np.concatenate((self.XReference, fill_array))
            self.Circut_current = np.concatenate(
                (self.Circut_current, fill_array))

        self.XData = np.reshape(self.XData, (corrected_y_size, x_uni_size))
        self.YData = np.reshape(self.YData, (corrected_y_size, x_uni_size))
        self.ZData = np.reshape(self.ZData, (corrected_y_size, x_uni_size))
        self.XReference = np.reshape(self.XReference,
                                     (corrected_y_size, x_uni_size))
        self.Circut_current = np.reshape(self.Circut_current,
                                         (corrected_y_size, x_uni_size))

        print('\'reshape\':Data has been reshaped into ', self.ZData.shape)

    def reorder(self):
        # 如果x轴data不是从小到大排列，而是出现跳变的情况，对x轴从小到大重新排列，通常为hard gap scan时候agilent2 voltage并不是单调随着bbias递增而递增
        for i, j in enumerate(self.YData[:, 0]):
            dictionary = dict(zip(self.XData[i, :], self.ZData[i, :]))
            re_order_X = np.sort(self.XData[i, :], axis=0)
            re_order_Z = [dictionary[v] for v in re_order_X]
            self.XData[i, :] = re_order_X
            self.ZData[i, :] = re_order_Z

    def VariableCheck(self):
        """
        逐个打印init里面所有属性的size和type。对于数字，打印其值
        """
        print('self.file_path =', self.file_path)
        print('self.column_x_index = {0}, self.XName = {1}'.format(
            self.column_x_index, self.XName))
        print('self.column_y_index = {0}, self.YName = {1}'.format(
            self.column_y_index, self.YName))
        print('self.column_z_index = {0}, self.ZName = {1}'.format(
            self.column_z_index, self.ZName))
        print('\n')
        print('self.XData: Type = {0}; Shape = {1}'.format(
            type(self.XData), self.XData.shape))
        print('self.YData: Type = {0}; Shape = {1}'.format(
            type(self.YData), self.YData.shape))
        print('self.ZData: Type = {0}; Shape = {1}'.format(
            type(self.ZData), self.ZData.shape))
        print('self.DataMatrix: Type = {0}; Shape = {1}'.format(
            type(self.DataMatrix), self.DataMatrix.shape))
        print('\n')
        print(
            'Unique method: self.XSize = {0}, self.XStart = {1}, self.XEnd = {2}, self.XStep = {3} '
            .format(self.XSize, self.XStart, self.XEnd, self.XStep))
        print(
            'Text method:   self.XSize = {0}, self.XStart = {1}, self.XEnd = {2} '
            .format(self.__VariableSizeObtain_text()[0],
                    self.__VariableSizeObtain_text()[2],
                    self.__VariableSizeObtain_text()[4]))
        print(
            'Unique method: self.YSize = {0}, self.YStart = {1}, self.YEnd = {2}, self.YStep = {3} '
            .format(self.YSize, self.YStart, self.YEnd, self.YStep))
        print(
            'Text method:   self.YSize = {0}, self.YStart = {1}, self.YEnd = {2} '
            .format(self.__VariableSizeObtain_text()[1],
                    self.__VariableSizeObtain_text()[3],
                    self.__VariableSizeObtain_text()[5]))
        print('\n')

        if self.XSize != self.__VariableSizeObtain_text(
        )[0] or self.YSize != self.__VariableSizeObtain_text()[1]:
            print(
                'Text method do not correspond to unique method, please check the original dat file.'
            )

        print('VARIABLE CHECK COMPLETE')

    def __TextObtain(self):
        """
        获取.dat文件里面的每一行的信息
        :return: list，元素为每一行的文本
        """
        f = open(self.file_path, 'r')
        lines = f.readlines()
        f.close()

        return lines

    def __VariableSizeObtain_unique(self):
        """
        使用unique方法获得xy size start end等数据。该方法为主要方法，而从text里得到的为辅助方法
        :return:tuple，变量x和变量y的size end start
        """

        x_size = len(np.unique(self.XData))
        x_start = np.unique(self.XData)[0]
        x_end = np.unique(self.XData)[-1]
        x_step = (x_end - x_start) / (x_size - 1)

        y_size = len(np.unique(self.YData))
        y_start = np.unique(self.YData)[0]
        y_end = np.unique(self.YData)[-1]
        y_step = (y_end - y_start) / (y_size - 1)

        return x_size, y_size, x_start, y_start, x_end, y_end, x_step, y_step

    def __VariableSizeObtain_text(self):
        """
          用于从.dat文件获取变量xy的尺寸，默认为Column 1和Column 2为变量x和y
          size,end start的获取是基于匹配到Column字样后其下面第几行第几个元素开始之后的字符串的识别
          :return:tuple，变量x和变量y的size end start
          """
        lines = self.__TextObtain()

        def is_number(str):
            try:
                int(str)
                return True
            except ValueError:
                return False

        for i, j in enumerate(lines):  # 有空格为682 无空格为341
            if 'Column {}:'.format(self.column_x_index) in j:
                if is_number(lines[i + 3][8:]):
                    x_size = int(lines[i + 3][8:])
                    x_start = float(lines[i + 4][9:])
                    x_end = float(lines[i + 1][7:])
                    x_name = lines[i + 2][8:]
                elif is_number(lines[i + 6][8:]):
                    x_size = int(lines[i + 6][8:])
                    x_start = float(lines[i + 8][9:])
                    x_end = float(lines[i + 2][7:])
                    x_name = lines[i + 4][8:]
                else:
                    print('Can not obtain X varizble size from text.')
                    x_size = 'NaN'
                    x_start = 'NaN'
                    x_end = 'NaN'
                    x_name = 'N/A'
                    # os.kill()

            if 'Column {}:'.format(self.column_y_index) in j:
                if is_number(lines[i + 3][8:]):
                    y_size = int(lines[i + 3][8:])
                    y_start = float(lines[i + 4][9:])
                    y_end = float(lines[i + 1][7:])
                    y_name = lines[i + 2][8:]
                elif is_number(lines[i + 6][8:]):
                    y_size = int(lines[i + 6][8:])
                    y_start = float(lines[i + 8][9:])
                    y_end = float(lines[i + 2][7:])
                    y_name = lines[i + 4][8:]
                else:
                    print('Can not obtain Y varizble size from text.')
                    y_size = 'NaN'
                    y_start = 'NaN'
                    y_end = 'NaN'
                    y_name = 'N/A'
                    # os.kill()

            if 'Column {}:'.format(self.column_z_index) in j:
                if is_number(lines[i + 3][8:]):
                    z_name = lines[i + 2][8:]
                elif is_number(lines[i + 6][8:]):
                    z_name = lines[i + 4][8:]
                else:
                    print('Can not obtain Z varizble name from text.')
                    z_name = 'N/A'

        return x_size, y_size, x_start, y_start, x_end, y_end, x_name, y_name, z_name

    def __isfloat(self, value):
        """
        判断value是不是浮点数。是浮点数返回True，否则返回False。
        """
        try:
            float(value)
            return True
        except:
            return False

    def __FindFirstLineOfDataMatrix(self):
        """
        判断数据从第几行开始：第一个可以变成浮点数的那一行即为数据开始的行
        :return: int 数据第一行的行数
        """
        lines = self.__TextObtain()
        n = 1
        for i in range(0, len(lines)):
            line = lines[i].split()
            if len(line) != 0:
                if self.__isfloat(line[0][1:]):
                    if n == 1:
                        first_line = i
                        n = 0

        return first_line

    def __ObtainDataMatrix(self):
        """
        用于读取.datfile除去title后的纯数据矩阵,不包含不同y之之间存在的空行
        :return: Type = <class 'numpy.ndarray'>; Shape = (24442, 14)
        """
        lines = self.__TextObtain()
        start = self.__FindFirstLineOfDataMatrix()
        end = len(lines)
        RowNumber = end - start
        DataPerRow = np.zeros(RowNumber).tolist()
        j = 0
        for i in range(start, end):
            DataPerRow[j] = lines[i].split(
            )  # 把每一行拆分，放入list中。List中每一个元素对应某一行的所有数据。
            j += 1

        # ColumnNmuber=len(DataPerRow[0]) # 列数
        ColumnNmuber = len(DataPerRow[1])  # 列数
        DataMatrix = np.zeros((RowNumber, ColumnNmuber))
        blank_row = np.array([], dtype='int16')  # 空白行的行index
        # print(len(DataPerRow[0]))
        for n in range(0, RowNumber):
            if DataPerRow[n]:
                pass
            else:
                blank_row = np.append(blank_row, n)
                # print(False,n)
            for m in range(0, len(DataPerRow[n])):
                # print(m)
                DataMatrix[n][m] = float(DataPerRow[n][m])
        DataMatrix = np.delete(DataMatrix, blank_row, axis=0)  # 删除空白行

        return DataMatrix

    def unique_xy(self):
        # 导出可以用于画colormap的xyzdata，适用于
        tmpx = np.unique(self.XData)
        tmpy = np.unique(self.YData)

        if (self.XData[0] - self.XData[-1]) > 0:
            tmpx = tmpx[::-1]
            print('3232')
        if (self.YData[0] - self.YData[-1]) > 0:
            tmpy = tmpy[::-1]
            print('4343')
        self.unique_ZData = self.ZData.reshape((len(tmpy), len(tmpx)))

        self.unique_XData = tmpx
        self.unique_YData = tmpy

        return self

    def offset_correction(self, bias_offset=0):
        # 校准系统offset偏置电压
        self.XData = self.XData - bias_offset

    def offset_correction_z(self, bias_offset=0):
        # 校准系统offset偏置电压
        self.ZData = self.ZData - bias_offset

    def ref_correction(self, bias_offset=0):
        # 校准ref column，通常是vbias的偏置
        self.XReference = self.XReference - bias_offset

    def x_rescale(self, scaling_factor):
        # 放大或者缩小xdata
        self.XData = self.XData * scaling_factor

    def z_rescale(self, scaling_factor):  # MODIFIED HERE
        # 放大或者缩小zdata
        self.ZData = self.ZData * scaling_factor

    def bias_correction_from_rseries(self, R_series):
        # 用于两端发测量中通过bias减去出串联电阻乘以电流来算出样品真实的分压 r_series in kohm unit
        self.Circut_current = signal.savgol_filter(self.Circut_current,
                                                   window_length=3,
                                                   polyorder=1)
        print(self.XData[0, 0])
        self.XData = self.XData - R_series * self.Circut_current  # 注意单位
        print(self.XData[0, 0], self.Circut_current[0, 0])

    def bias_correction_from_rseries_tmp(self, R_series):  # MODIFIED HERE
        self.Circut_current = signal.savgol_filter(self.Circut_current,
                                                   window_length=3,
                                                   polyorder=1)
        print(self.ZData[0, 0])
        self.ZData = self.ZData - R_series * self.Circut_current  # 注意单位
        print(self.ZData[0, 0], self.Circut_current[0, 0])

    def bias_correction_from_rcontact(self, rcontact):  # MODIFIED HERE
        self.Circut_current = signal.savgol_filter(self.Circut_current,
                                                   window_length=3,
                                                   polyorder=1)
        self.ZData = self.ZData - rcontact * self.Circut_current

    def bias_correction_from_rcontact_x(self, rcontact):  # MODIFIED HERE
        self.Circut_current = signal.savgol_filter(self.Circut_current,
                                                   window_length=3,
                                                   polyorder=1)
        self.XData = self.XData - rcontact * self.Circut_current

    def bias_correction_from_rfilter(self, Ifilter, Vfilter):  # MODIFIED HERE
        # Ifilter=signal.savgol_filter(Ifilter, window_length=3,polyorder=1)
        f = interp1d(Ifilter, Vfilter)
        self.ZData = self.ZData - f(self.XData)

    def differential_resistance_correction_from_rfilter(
            self, Ifilter, dIfilter, exc=0.01):  # MODIFIED HERE
        differential_resistance = exc * 1e-3 / dIfilter
        differential_resistance = signal.savgol_filter(differential_resistance,
                                                       window_length=11,
                                                       polyorder=1)
        f = interp1d(Ifilter, differential_resistance)
        self.ZData = self.ZData - f(self.XData)

    def obtain_dvdi(self, exc=0.01):
        self.ZData = exc * 1e-3 / self.ZData

    def obtain_dvdi_4Tbluefors(self, circuit_amp, di):
        self.ZData = self.ZData / di * circuit_amp

    def quantum_conductance_to_ohm(self):  # MODIFIED HERE
        self.ZData = 1 / self.ZData * (consts.h / (2 * consts.e**2))

    def resistance_correction(self, R_series):  # MODIFIED HERE
        self.ZData = self.ZData - R_series

    def interp(self, multiplier=1, method='linear'):
        # 对于每一个y值都有着不同x值(e.g. agilent_2 voltage)的数据点而言，要通过对其进行差值使得每一个y值下均有相同的x值，才能用pcolormesh画成色图
        xmax = np.nanmax(self.XData)
        xmin = np.nanmin(self.XData)
        print('\'interp\':xmax=%f' % xmax)
        print('\'interp\':xmin=%f' % xmin)

        x_uni_size = np.shape(np.unique(self.XReference))[0]
        self.xsize_interp = (x_uni_size -
                             1) * multiplier + 1  #插值数据点为原来数据点的multiplier=2倍
        print('\'interp\':XSize after interp=%f' % self.xsize_interp)
        print('\'interp\':XSize before interp=%f' % x_uni_size)

        self.x_interp = np.linspace(xmin, xmax, self.xsize_interp)
        self.yi = self.YData[:, 0]  # 每一个独立的gate(y值)
        self.ZData_interp = np.full((len(self.yi), self.xsize_interp), np.nan)

        for i in range(len(self.yi)):
            interp_indices = ~np.isnan(self.XData[i, :])
            self.ZData_interp[i, :] = interp1d(self.XData[i, interp_indices],
                                               self.ZData[i, interp_indices],
                                               kind=method,
                                               bounds_error=False,
                                               fill_value=np.nan)(
                                                   self.x_interp)

        print('\'interp\':ZData shape before interp=', self.ZData.shape)
        print('\'interp\':ZData shape after interp=', self.ZData_interp.shape)

    def XYCrop(self,
               x_start_value=0,
               x_end_value=0,
               y_start_value=0,
               y_end_value=0):
        """
        获取相应裁剪XY值之后的数据，不输入参数，或者全部输入0代表不剪裁。
        :param x_start_value:
        :param x_end_value:
        :return: self
        """
        # 确定在剪裁范围内的xy开始结束指标，该指标是相对于在201个x数据和121个y数据里而使用的
        # XData_1st = self.XData[0:self.XSize-1]  # 第一个小矩阵快里面的x值,-1同样是因为python是从0开始的,
        # print(XData_1st)
        XData_1st = np.unique(
            self.XData)  #np.unique方法自动从小到大排列，#所以需要还原真实的数据是从大往小了扫还是从小往大了扫
        # print(XData_1st)
        if x_end_value == x_start_value == 0:
            x_start_index = 0
            x_end_index = self.XSize
        if x_end_value != 0:
            if x_end_value <= XData_1st[-1]:
                x_end_index = np.max(np.where(XData_1st <= x_end_value))
            else:
                x_end_index = self.XSize
        if x_start_value != 0:
            if x_start_value >= XData_1st[0]:
                x_start_index = np.max(np.where(XData_1st <= x_start_value))
            else:
                x_start_index = 0
        if x_start_value > x_end_value:
            raise TypeError("x start value must be smaller than x end value")

        # YData = np.linspace(np.min(self.YData),np.max(self.YData),self.YSize,dtype='float64')  # 所有不重复的121个y值
        YData = np.unique(self.YData)  # 所有不重复的121个y值
        # YData=YData[::-1] #反转y值（如果磁场是从大到小）
        print(YData.shape)
        if y_end_value == y_start_value == 0:
            y_start_index = 0
            y_end_index = len(YData)
        if y_start_value == 0:
            y_start_index = 0
        if y_end_value != 0:
            if y_end_value < YData[-1]:  # -1同样是因为python是从0开始的,
                y_end_index = np.max(np.where(YData <= y_end_value))
            else:
                y_end_index = self.YSize
        if y_start_value != 0:
            if y_start_value >= YData[0]:
                y_start_index = np.max(np.where(YData <= y_start_value))
            else:
                y_start_index = 0
        if y_start_value > y_end_value:
            raise TypeError("y start index must be smaller than y end index")
        # 得到经过剪裁后的xdata和ydata。这里是不重复的xdata和ydata，和self.XData，self.YData（和简单的txt文件拷贝下来的有重复的data不是一回事）
        # print(self.XData)
        Cropped_XData = self.XData[x_start_index:x_end_index]
        Cropped_YData = YData[y_start_index:y_end_index]
        Cropped_ZData = np.empty(
            shape=[len(Cropped_YData), len(Cropped_XData)], dtype='float64')
        i = 0
        for y_index in range(y_start_index, y_end_index):
            x_start_index_for_y_index = (
                self.XSize + 0
            ) * y_index + x_start_index  #归纳推理得出 如果每一个小方块中有空格 则变成+1 如果没有变成+0即可
            x_end_index_for_y_index = (self.XSize + 0) * y_index + x_end_index
            ZData_for_y_index = self.ZData[
                x_start_index_for_y_index:x_end_index_for_y_index]
            tmp = OneDimensionArray(ZData_for_y_index)
            # print(tmp.array,y_index)
            # print(tmp.array.shape,y_index)
            Cropped_ZData[i] = tmp.array
            i += 1

        if self.XData[3] - self.XData[4] > 0:  #如果测量过程中x坐标为从大往小了扫，则要反转一下x
            Cropped_XData = Cropped_XData[::-1]
            print(
                'X sweeps from large to small, X coordinate has been flipped.')
        if self.YData[0] - self.YData[-1] > 0:  #磁场若是从大往小了加，则上下反转一下zdata
            Cropped_ZData = np.flipud(Cropped_ZData)
            print(
                'Y ramps from large to small, Y coordinate has been flipped.')

        # 经过Crop之后更新一下CROPPED_XYZDATA值
        self.Cropped_XData = Cropped_XData  # 经过crop后的100个xdata 1d ndarray 从左到右为从小到大的顺序，以免之后peakspacing变成负数
        self.Cropped_YData = Cropped_YData  # 经过crop后的50个ydata 1d ndarray
        self.Cropped_ZData = Cropped_ZData  # 经过crop后的50X100个Zdata 原来的为list，这里给转化一下成为ndarray方便以后其他处理

        return Cropped_XData, Cropped_YData, Cropped_ZData

    def Normalization(self):
        """
        二维数组ZData的行归一化处理，利用一维数组的处理方式
        :return: self
        """
        for i in range(0, len(self.Cropped_ZData)):
            # print(self.Cropped_ZData)
            tmp = OneDimensionArray(self.Cropped_ZData[i])
            tmp.Normalize()
            self.Cropped_ZData[i] = tmp.array

        return self

    def Smooth(self, window_length=11, poly_order=3):
        """
        对二维数组每一行进行多项式平滑处理，利用一维数组的平滑处理方式
        :param window_length:
        :param poly_order:
        :return: self
        """
        for i in range(0, len(self.Cropped_ZData)):
            tmp = OneDimensionArray(self.Cropped_ZData[i])
            tmp.Smooth(window_length, poly_order)
            self.Cropped_ZData[i] = tmp.array

        return self

    def create_kernel(self, x_dev, y_dev, cutoff, distr):
        distributions = {
            'gaussian': lambda r: np.exp(-(r**2) / 2.0),
            'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
            'lorentzian': lambda r: 1.0 / (r**2 + 1.0),
            'thermal': lambda r: np.exp(r) / (1 * (1 + np.exp(r))**2)
        }
        func = distributions[distr]

        hx = np.floor((x_dev * cutoff) / 2.0)
        # print(hx)
        hy = np.floor((y_dev * cutoff) / 2.0)

        x = np.linspace(-hx, hx, int(hx * 2) + 1) / x_dev
        y = np.linspace(-hy, hy, int(hy * 2) + 1) / y_dev

        if x.size == 1: x = np.zeros(1)
        if y.size == 1: y = np.zeros(1)

        xv, yv = np.meshgrid(x, y)
        kernel = func(np.sqrt(xv**2 + yv**2))
        kernel /= np.sum(kernel)

        return kernel

    def lowpass(self, x_width=3, y_height=3, method='gaussian'):
        kernel = self.create_kernel(x_width, y_height, 7, method)
        self.ZData = ndimage.filters.convolve(self.ZData, kernel)
        self.ZData = np.ma.masked_invalid(self.ZData)

    # def lowpass_m(self, x_width=3, y_height=3, method='gaussian'):
    #     kernel = self.create_kernel(x_width, y_height, 4, method)
    #     self.ZData = ndimage.filters.convolve(self.ZData, kernel)
    #     self.ZData = np.ma.masked_invalid(self.ZData)

    def Export(self):
        """
        将Crop过之后的XYZData输出到.dat文件
        :return:
        """
        pass


class OneDimensionArray(object):
    """
    该类为一维数组，包含处理一维数组的各种方法
    属性：
    极大或极小值对应的指标
    """

    def __init__(self, array):

        self.array = np.atleast_1d(array).astype('float64')

        self.MaxIndex = self.__FindMaxIndex()
        self.MinIndex = self.__FindMinIndex()

    def Smooth(self, window_length=11, poly_order=2):
        """
        用于光滑曲线,光滑之后更新一下maxindex和minindex，peakspacing和valleyspacing
        :param window_length:
        :param poly_order:
        :return: self
        """
        self.array = signal.savgol_filter(self.array, window_length,
                                          poly_order)

        self.MaxIndex = self.__FindMaxIndex()
        self.MinIndex = self.__FindMinIndex()

        return self.array

    def __FindMaxIndex(self):
        """
        用于寻找一维数据里的极值，np.greater为极大值，np。greater_equal为包含pleatau的极大值
        :return:ndarray 极值对应的指标
        """
        self.PeakOrVaelly = 'peak'

        return signal.argrelextrema(self.array, np.greater)[0]

    def __FindMinIndex(self):

        self.PeakOrVaelly = 'valley'
        return signal.argrelextrema(self.array, np.less)[0]

    def Normalize(self):
        """
        归一化处理
        :return: self
        """
        self.array = self.array / (np.max(self.array))

        return self

    def FindNearest_Value(self, value):
        """
        用于寻找在self.array里于value最近的那个值
        :param value:
        :return:self.array中最接近value的值
        """
        index = (np.abs(self.array - value)).argmin()

        return self.array[index]

    def FindNearest_index(self, value):
        """
        用于寻找在self.array里于value最近的那个值
        :param value:
        :return:self.array中最接近value的值
        """
        index = (np.abs(self.array - value)).argmin()

        return index

    def Plot(self, a='peak'):
        """
        把原始数据和极值点绘制出来
        :param a: peak or valley
        :return:  none
        """
        x = np.arange(0, len(self.array), 1)
        y = self.array

        if a == 'peak':
            x_exterma = self.MaxIndex
            y_extrema = self.array[x_exterma]
            label = 'local maxium'
        elif a == 'valley':
            x_exterma = self.MinIndex
            y_extrema = self.array[x_exterma]
            label = 'local minimum'
        else:
            raise TypeError("Please designate peak or valley")

        fig, ax = plt.subplots(2, 1, figsize=(16, 8))

        ax[0].set_xlabel('Index', fontsize=20)
        ax[0].set_ylabel('Value', fontsize=20)
        ax[0].scatter(x_exterma,
                      y_extrema,
                      label='Extrema points',
                      marker='x',
                      color='r',
                      s=100)
        ax[0].plot(x, y, label='Original Data', marker='o')
        ax[0].axhline(np.average(y_extrema), c='r', linestyle='-.')
        ax[0].axhline(np.std(y_extrema), c='y', linestyle='-.')
        ax[0].legend(loc="best")
        ax[0].set_title("{} {} extrema points are found.BEFORE FILTER".format(
            len(x_exterma), label))

        self.HeightFilter(0.7)
        if a == 'peak':
            x_exterma = self.MaxIndex
            y_extrema = self.array[x_exterma]
            label = 'local maxium'
        elif a == 'valley':
            x_exterma = self.MinIndex
            y_extrema = self.array[x_exterma]
            label = 'local minimum'
        else:
            raise TypeError("Please designate peak or valley")
        ax[1].set_xlabel('Index', fontsize=20)
        ax[1].set_ylabel('Value', fontsize=20)
        ax[1].scatter(x_exterma,
                      y_extrema,
                      label='Extrema points',
                      marker='x',
                      color='r',
                      s=100)
        ax[1].plot(x, y, label='Original Data', marker='o')
        ax[1].axhline(np.average(y_extrema), c='r', linestyle='-.')
        ax[1].axhline(np.std(y_extrema), c='y', linestyle='-.')
        ax[1].legend(loc="best")
        ax[1].set_title("{} {} extrema points are found.AFTER FILTER".format(
            len(x_exterma), label))

        plt.show()


def find_index_range(array, min_value, max_value):
    """ Find values between and including min_value and max_value in array. """
    minrange = np.where(array >= min_value)
    maxrange = np.where(array <= max_value)
    totalrange = np.intersect1d(minrange, maxrange)
    return totalrange


def qtplot_transform(x, y, z, filename):
    """
    将拟合产生的分离的三个x y z三个data变成qtplot可以读取的形式
    """
    filename = './' + str(filename)
    file = open(filename, mode='w')
    file.write('# Filename: for_qtplot_triangle.dat')
    file.write('\n\n')
    file.write('# Timestamp: ')
    file.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    file.write('\n\n\n\n')

    file.write('#\tColumn 1:')
    file.write('\n\n')
    file.write('#\tend: %.6g' % (x[-1]))
    file.write('\n\n')
    file.write('#\tname: E/Delta0')
    file.write('\n\n')
    file.write('#\tsize: %.6g' % len(x))
    file.write('\n\n')
    file.write('#\tstart: %.6g' % (x[0]))
    file.write('\n\n')
    file.write('#\ttype: coordinate')
    file.write('\n\n\n\n')

    file.write('#\tColumn 2:')
    file.write('\n\n')
    file.write('#\tend: %.6g' % (y[-1]))
    file.write('\n\n')
    file.write('#\tname: Vz(meV)')
    file.write('\n\n')
    file.write('#\tsize: %.6g' % len(y))
    file.write('\n\n')
    file.write('#\tstart: %.6g' % (y[0]))
    file.write('\n\n')
    file.write('#\ttype: coordinate')
    file.write('\n\n\n\n')

    file.write('#\tColumn 3:')
    file.write('\n\n')
    file.write('#\tend: 0')
    file.write('\n\n')
    file.write('#\tname: di/dv in QC')
    file.write('\n\n')
    file.write('#\tsize: 1')
    file.write('\n\n')
    file.write('#\tstart: 0')
    file.write('\n\n')
    file.write('#\ttype: data')
    file.write('\n\n\n\n')

    for i, j in enumerate(y):
        for k, l in enumerate(x):
            file.write('%e\t' % l)
            file.write('%e\t' % j)
            file.write('%e\n' % z[k, i])
