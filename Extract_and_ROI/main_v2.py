import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import os
from skimage import morphology
from roi_extract_lib import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk  # 在tkinter中内嵌matplotlib进行绘图，否则会有问题
from matplotlib.figure import Figure
import csv
from PIL import ImageTk, Image
from matplotlib.backend_bases import key_press_handler
import shutil
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.font_manager import  FontProperties
font=FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=12)

# 全局变量与文件相关----------------------------------------------------------------------------------------------------
# 存储采集图像的文件夹及文件名
FILE_ROOT = 'data/'  # ''D:/Pycharm_CodeSet/data/'
filename = None
# 保存呼吸数据的文件夹及文件名
INS_DATA_ROOT = 'INS_data/'  # ''D:/Pycharm_CodeSet/INS_data/'
ins_filename = None
# 保存呼吸波形的文件夹名
WAVES_ROOT = 'WAVES_img/'  # 'D:/Pycharm_CodeSet/WAVES_img/'
waves_filename = None
# 呼吸波形重建所用傅里叶算子数量
REBUILD_NUM = 9
# 需要处理的图像数目
PROCESS_IMAGE_NUM=None
# 呼吸数据提取完成标志位
EXTRACT_FINISH=-1
# 用户登录成功标志位
LOGIN_SUCCESS=-1

# 主窗口----------------------------------------------------------------------------------------------------------------
window = tk.Tk()
window.option_add('*Font',('Times',11))
window.title('呼吸检测')
window.geometry('730x350')
# window.resizable(False, False)

# 画布提示label控件
label_1=tk.Label(window,text='图像显示区',bg='yellow')
label_1.place(x=200,y=5,anchor='n')
label_2=tk.Label(window,text='波形显示区',bg='yellow')
label_2.place(x=540,y=5,anchor='n')

# 显示当前状态
state_var=tk.StringVar()
text_label=tk.Label(window,text='状态：',bg='yellow')
text_label.place(x=40,y=310,anchor='nw')
state_label=tk.Label(window,textvariable=state_var,bg='yellow')
state_label.place(x=90,y=310,anchor='nw')
state_var.set('等待操作...')
# 呼吸数据提取Label及进度条
l_var=tk.StringVar() # label显示变量
process=tk.Label(window,textvariable=l_var,bg='gray')
process.place(x=40,y=280,anchor='nw')
l_var.set('Processed ' + str(0) + ' / ' + str(100))
p_var=tk.IntVar()   #进度条变量
pb = ttk.Progressbar(window, length=PROCESS_IMAGE_NUM,variable=p_var,value=0)   #进度条
pb.place(x=170,y=280,anchor='nw')

# 函数定义--------------------------------------------------------------------------------------------------------------
# 定义用户登录函数
def user_login_window():
    global LOGIN_SUCCESS
    window.withdraw()
    # window1 = tk.Tk()
    window1=tk.Toplevel(window)
    window1.option_add('*font', (('Times', 11)))
    window1.title('欢迎登录呼吸检测系统')
    window1.geometry('320x160')
    # 用户信息
    tk.Label(window1, text='用户名:').place(x=40, y=40)
    tk.Label(window1, text='密码:').place(x=40, y=80)
    # 用户登录输入框entry
    # 用户名
    var_usr_name = tk.StringVar()
    var_usr_name.set('lzh')
    entry_usr_name = tk.Entry(window1, textvariable=var_usr_name)
    entry_usr_name.place(x=100, y=40)
    # 用户密码
    var_usr_pwd = tk.StringVar()
    entry_usr_pwd = tk.Entry(window1, textvariable=var_usr_pwd, show='*')
    entry_usr_pwd.place(x=100, y=80)

    # 定义用户登录功能
    def usr_login():
        global LOGIN_SUCCESS
        usr_name = var_usr_name.get()
        usr_pwd = var_usr_pwd.get()
        # 这里设置异常捕获，当我们第一次访问用户信息文件时是不存在的，所以这里设置异常捕获。
        # 中间的两行就是我们的匹配，即程序将输入的信息和文件中的信息匹配。
        try:
            with open('usrs_info.pickle', 'rb') as usr_file:
                usrs_info = pickle.load(usr_file)
        except FileNotFoundError:
            # 在没有读取到`usr_file`的时候，程序会创建一个`usr_file`这个文件，并将管理员
            # 的用户和密码写入，即用户名为`admin`密码为`admin`。
            with open('usrs_info.pickle', 'wb') as usr_file:
                usrs_info = {'admin': 'admin'}
                pickle.dump(usrs_info, usr_file)
                usr_file.close()  # 必须先关闭，否则pickle.load()会出现EOFError: Ran out of input
        # 如果用户名和密码与文件中的匹配成功，则会登录成功，并跳出弹窗how are you? 加上你的用户名。
        if usr_name in usrs_info:
            if usr_pwd == usrs_info[usr_name]:
                LOGIN_SUCCESS=1
                tkinter.messagebox.showinfo(title='Welcome', message=usr_name + '登录成功！ ')
                window1.destroy()
                window.deiconify()
            # 如果用户名匹配成功，而密码输入错误，则会弹出'Error, your password is wrong, try again.'
            else:
                tkinter.messagebox.showerror(message='密码错误，请重新输入！')
        else:  # 如果发现用户名不存在
            is_sign_up = tkinter.messagebox.askyesno('Welcome！ ', '您尚未注册，立刻注册？')
            # 提示需不需要注册新用户
            if is_sign_up:
                usr_sign_up()

    # 第9步，定义用户注册功能
    def usr_sign_up():
        def sign_to_Hongwei_Website():
            # 获取注册所输入的信息
            np = new_pwd.get()
            npf = new_pwd_confirm.get()
            nn = new_name.get()
            # 打开我们记录数据的文件，将注册信息读出
            with open('usrs_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)
            # 这里就是判断，如果两次密码输入不一致，则提示Error, Password and confirm password must be the same!
            if np != npf:
                tkinter.messagebox.showerror('Error', '请重新确认密码')
            # 如果用户名已经在我们的数据文件中，则提示Error, The user has already signed up!
            elif nn in exist_usr_info:
                tkinter.messagebox.showerror('Error', '用户名已存在！')
            # 最后如果输入无以上错误，则将注册输入的信息记录到文件当中，并提示注册成功Welcome！,You have successfully signed up!，然后销毁窗口。
            else:
                exist_usr_info[nn] = np
                with open('usrs_info.pickle', 'wb') as usr_file:
                    pickle.dump(exist_usr_info, usr_file)
                tkinter.messagebox.showinfo('Welcome', '注册成功！')
                # 然后销毁窗口。
                window_sign_up.destroy()


        # 定义长在窗口上的窗口
        window_sign_up = tk.Toplevel(window1)
        window_sign_up.geometry('300x200')
        window_sign_up.title('注册窗口')
        new_name = tk.StringVar()  # 将输入的注册名赋值给变量
        new_name.set('lzh')
        tk.Label(window_sign_up, text='用户名: ').place(x=10, y=10)  # 将`User name:`放置在坐标（10,10）。
        entry_new_name = tk.Entry(window_sign_up, textvariable=new_name)  # 创建一个注册名的`entry`，变量为`new_name`
        entry_new_name.place(x=130, y=10)  # `entry`放置在坐标（150,10）.
        new_pwd = tk.StringVar()
        tk.Label(window_sign_up, text='密码: ').place(x=10, y=50)
        entry_usr_pwd = tk.Entry(window_sign_up, textvariable=new_pwd, show='*')
        entry_usr_pwd.place(x=130, y=50)
        new_pwd_confirm = tk.StringVar()
        tk.Label(window_sign_up, text='确认密码: ').place(x=10, y=90)
        entry_usr_pwd_confirm = tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*')
        entry_usr_pwd_confirm.place(x=130, y=90)
        # 下面的 sign_to_Hongwei_Website
        btn_comfirm_sign_up = tk.Button(window_sign_up, text='Sign up', command=sign_to_Hongwei_Website)
        btn_comfirm_sign_up.place(x=180, y=120)

    # 第7步，login and sign up 按钮
    btn_login = tk.Button(window1, text='登录', command=usr_login)
    btn_login.place(x=110, y=110)
    btn_sign_up = tk.Button(window1, text='注册', command=usr_sign_up)
    btn_sign_up.place(x=210, y=110)
    # 第10步，主窗口循环显示
    window1.mainloop()


def extract_roi():
    global filename
    global ins_filename
    global PROCESS_IMAGE_NUM
    global state_var
    global p_var
    global EXTRACT_FINISH
    # 保存胸腔区域数据数组------------------------------------------------------------
    ROI_data = np.zeros((100, 20))
    state_var.set('执行呼吸数据提取...')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        start_img=1
        # end_img=len(os.listdir(filename))
        end_img=PROCESS_IMAGE_NUM
        if end_img is None:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi',message='请先导入图像！')
            state_var.set('等待操作...')
        else:
            ins_filename=INS_DATA_ROOT+sub_file_name.get()+'.csv'
            csvfile = open(ins_filename, 'w', encoding='utf-8')
            writer = csv.writer(csvfile)
            # Label 控件
            # l_var=tk.StringVar() # label显示变量
            # process=tk.Label(window,textvariable=l_var,bg='gray')
            # process.place(x=40,y=260,anchor='nw')
            # Progressbar 控件
            # p_var=tk.IntVar()   #进度条变量
            # pb = ttk.Progressbar(window, length=end_img,variable=p_var)   #进度条
            # pb.place(x=165,y=260,anchor='nw')
            for i in range(start_img,end_img+1):
                p_var.set(int(i*100/end_img))
                l_var.set('Processed '+str(int(i*100/end_img))+' / '+str(100))
                in_name=filename+'/'+str(i)+'.png'    #深度图存储路径
                src = cv.imread(in_name, cv.IMREAD_GRAYSCALE)
                roi_gray = src.copy()
                _, binary = cv.threshold(src, 128, 1, cv.THRESH_BINARY)
                contours, maxAreaIdx = find_contours(binary)
                contour_image = np.zeros_like(src)
                cv.drawContours(contour_image, contours, maxAreaIdx, 255, 1)
                border = contours[maxAreaIdx].reshape(-1, 2)
                proportion = 0.02
                rebuild_contours = get_fourier_descriptor(border, proportion)
                rebuild_img = np.zeros_like(src)
                cv.drawContours(rebuild_img, [rebuild_contours], -1, 255, 1)
                mask = np.zeros_like(src)
                cv.drawContours(mask, [rebuild_contours], -1, 255, cv.FILLED)
                shallow, shallow_line = location_shallow(mask)
                mask = np.where(mask == 255, 1, 0)
                skeleton = morphology.skeletonize(mask) + 0
                dst = skeleton
                shallow[1, 0, 0] = get_shallow_mid(dst, shallow_line)
                Point1, Point2 = getPoints(dst.astype(np.uint8), 2, 7, 2)
                left, right, mainP = get_main_skel(Point1, shallow)
                pca_depth = np.mean(roi_gray[shallow_line:left[1] + 1, shallow[0, 0, 0]:right[0] + 1])
                writer.writerow([pca_depth])

                # pca相关数据保存-----------------------------------------
                mmat = roi_gray[shallow_line:left[1] + 1, shallow[0, 0, 0]:right[0] + 1]
                data_number = mmat.shape[0] * mmat.shape[1]
                shape1 = data_number // 20
                use_data = ((mmat.reshape(-1))[:20 * shape1]).reshape(20, -1)
                ROI_data[i - 1, :] = use_data.mean(axis=1)
                # ROI_data[i - 1, :] = roi_gray[shallow_line:left[1] + 1, shallow[0, 0, 0]:right[0] + 1].reshape(1, -1)[0,:15000]  # (1,-1)


                # 显示方式1：在dst中显示ROI和轮廓，骨架
                # dst = np.where(dst == 1, 255, 0).astype(np.uint8)
                # radius = 3
                # cv.circle(dst, mainP, radius, 255, -1, 8)
                # cv.rectangle(dst, tuple(shallow[0, 0, :]), right, 255, 1)
                # cv.circle(dst, tuple(shallow[0, 0, :]), radius, 255, -1, 8)
                # cv.circle(dst, tuple(shallow[2, 0, :]), radius, 255, -1, 8)
                # cv.circle(dst, tuple(shallow[1, 0, :]), radius, 255, -1, 8)
                # dst = dst + contour_image
                # dst=cv.resize(dst,(320,240))
                # print(i)
                # img=Image.fromarray(dst.astype('uint8'))
                # 显示方式2：在src中显示ROI区域
                cv.rectangle(src, tuple(shallow[0, 0, :]), tuple(right), 255, -1)
                src=cv.resize(src,(320,240))
                img=Image.fromarray(src.astype('uint8'))

                photo=ImageTk.PhotoImage(img)
                canvas1.create_image(160,0,anchor='n',image=photo)
                canvas1.update()
                pb.update()
            csvfile.close()
            # pca相关数据-----------------------------------------
            ROI_data = pd.DataFrame(ROI_data)
            ROI_data.to_csv('pca_data.csv')
            state_var.set('呼吸数据提取完成')
            EXTRACT_FINISH=1


def waves():
    global ins_filename    #呼吸数据文件存储路径
    global waves_filename  #呼吸波形文件存储路径，但是不能使用plt.savefig进行存储，因此未使用此变量
    global state_var
    global EXTRACT_FINISH
    state_var.set('波形显示')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        if EXTRACT_FINISH==-1:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi',message='请先提取呼吸数据！')
            state_var.set('等待操作...')
        else:
            inspiration = []
            fp = open(ins_filename, 'r', encoding='utf-8')
            inspiration.extend(float((line.strip().split())[0]) for line in fp.readlines() if len(line) > 1)
            fp.close()
            waves_filename=WAVES_ROOT+sub_file_name.get()+'.png'
            ax.clear()
            ax.plot(inspiration, 'k--',zorder=1)
            ax.set_title('respiratory wave')
            ax.set_xlabel('frame')
            ax.set_ylabel('respiratory data')
            canvas2.draw()
            print('ok')


def waves_pca():
    global ins_filename    #呼吸数据文件存储路径
    global waves_filename  #呼吸波形文件存储路径，但是不能使用plt.savefig进行存储，因此未使用此变量
    global state_var
    global EXTRACT_FINISH
    state_var.set('波形显示')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        if EXTRACT_FINISH==-1:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi',message='请先提取呼吸数据！')
            state_var.set('等待操作...')
        else:
            CSV_FILE = 'pca_data.csv'
            my_data = np.genfromtxt(CSV_FILE, delimiter=',')[1:, 1:]
            # my_data = my_data.reshape(100, 20, 750).mean(axis=2)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=5)
            X_reduced = pca.fit_transform(my_data)
            ax.clear()
            ax.plot(range(100), X_reduced[:, 0], 'k-')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Value')
            canvas2.draw()


def compute_rr():
    global ins_filename  # 呼吸数据文件存储路径
    global waves_filename  # 呼吸波形文件存储路径，但是不能使用plt.savefig进行存储，因此未使用此变量
    global state_var
    global EXTRACT_FINISH
    state_var.set('显示呼吸频率')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        if EXTRACT_FINISH == -1:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi', message='请先提取呼吸数据！')
            state_var.set('等待操作...')
        else:
            inspiration = []
            fp = open(ins_filename, 'r', encoding='utf-8')
            inspiration.extend(float((line.strip().split())[0]) for line in fp.readlines() if len(line) > 1)
            waves_filename = WAVES_ROOT + sub_file_name.get() + '.png'
            ax.clear()
            ax.plot(inspiration, 'k--', zorder=1, label='raw wave')
            data = np.array(inspiration)
            Y = np.fft.fft(data)
            for i in range(REBUILD_NUM, data.shape[0]):
                Y[i] = 0
            YY = np.fft.ifft(Y)
            YY = np.abs(YY)
            ax.plot(YY, 'k',zorder=2, label='fft wave')
            # 找峰值点并进行绘制
            data = YY
            diff = -1 * np.ones((data.shape[0] - 1, 1))
            for i in range(data.shape[0] - 1):
                diff[i] = data[i + 1] - data[i]
                if diff[i] > 0:
                    diff[i] = 1
                elif diff[i] < 0:
                    diff[i] = -1
            trend = diff
            peak_count = 1
            trough_count = 0
            result = np.zeros((trend.shape[0] - 1, 1))
            for i in range(result.shape[0]):
                result[i] = trend[i + 1] - trend[i]
                if result[i] == -2:
                    peak_count += 1
                    ax.scatter(i + 1, data[i + 1], c='k',s=10, zorder=3)
                elif result[i] == 2:
                    trough_count += 1
            peak_count -= 1
            print(peak_count)
            ax.set_title('respiratory rate')  # 之后加上获取的时间结合peak_count进行计算
            ax.set_xlabel('frame')
            ax.set_ylabel('respiratory data')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.legend()
            canvas2.draw()
            print('ok')


def compute_rr_pca():
    global ins_filename  # 呼吸数据文件存储路径
    global waves_filename  # 呼吸波形文件存储路径，但是不能使用plt.savefig进行存储，因此未使用此变量
    global state_var
    global EXTRACT_FINISH
    state_var.set('显示呼吸频率')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        if EXTRACT_FINISH == -1:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi', message='请先提取呼吸数据！')
            state_var.set('等待操作...')
        else:
            CSV_FILE = 'pca_data.csv'
            my_data = np.genfromtxt(CSV_FILE, delimiter=',')[1:, 1:]
            # my_data = my_data.reshape(100, 20, 750).mean(axis=2)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=5)
            X_reduced = pca.fit_transform(my_data)

            # 低通滤波
            ax.clear()
            for rebuild_num in range(8,9):
                data = X_reduced[:, 0]
                Y = np.fft.fft(data)
                indices = np.argsort(-np.abs(Y))
                for i in range(rebuild_num, data.shape[0]):
                    Y[indices[i]] = 0
                YY = np.fft.ifft(Y)
                # print(YY)
                ax.plot(YY, 'k-', label='after Fourier transform', zorder=2)
                # 找峰值点并进行绘制
                data = np.real(YY)
                diff = np.zeros((data.shape[0], 1))
                for i in range(data.shape[0] - 1):
                    diff[i] = data[i + 1] - data[i]
                    if diff[i] > 0:
                        diff[i] = 1
                    elif diff[i] < 0:
                        diff[i] = -1
                trend = diff
                for i in reversed(range(data.shape[0] - 1)):
                    if diff[i] == 0 and diff[i + 1] >= 0:
                        trend[i] = 1
                    elif diff[i] == 0 and diff[i + 1] < 0:
                        trend[i] = -1
                result = np.zeros((trend.shape[0] - 1, 1))
                peak_count = 1
                trough_count = 1
                for i in range(result.shape[0]):
                    result[i] = trend[i + 1] - trend[i]
                    if result[i] == -2:
                        peak_count += 1
                        # ax.scatter(i+1,data[i+1],c='k',zorder=3,s=15)
                    elif result[i] == 2:
                        trough_count += 1
                peak_count -= 1
                print('ok')
            ax.plot(range(100), X_reduced[:, 0], 'k--', label='before Fourier transform')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Value')
            # ax.set_title('respiratory rate='+'15次/分',fontproperties=font)  # 之后加上获取的时间结合peak_count进行计算
            ax.legend()
            canvas2.draw()

            # inspiration = []
            # fp = open(ins_filename, 'r', encoding='utf-8')
            # inspiration.extend(float((line.strip().split())[0]) for line in fp.readlines() if len(line) > 1)
            # waves_filename = WAVES_ROOT + sub_file_name.get() + '.png'
            # ax.clear()
            # ax.plot(inspiration, 'k--', zorder=1, label='raw wave')
            # data = np.array(inspiration)
            # Y = np.fft.fft(data)
            # for i in range(REBUILD_NUM, data.shape[0]):
            #     Y[i] = 0
            # YY = np.fft.ifft(Y)
            # YY = np.abs(YY)
            # ax.plot(YY, 'k',zorder=2, label='fft wave')
            # # 找峰值点并进行绘制
            # data = YY
            # diff = -1 * np.ones((data.shape[0] - 1, 1))
            # for i in range(data.shape[0] - 1):
            #     diff[i] = data[i + 1] - data[i]
            #     if diff[i] > 0:
            #         diff[i] = 1
            #     elif diff[i] < 0:
            #         diff[i] = -1
            # trend = diff
            # peak_count = 1
            # trough_count = 0
            # result = np.zeros((trend.shape[0] - 1, 1))
            # for i in range(result.shape[0]):
            #     result[i] = trend[i + 1] - trend[i]
            #     if result[i] == -2:
            #         peak_count += 1
            #         ax.scatter(i + 1, data[i + 1], c='k',s=10, zorder=3)
            #     elif result[i] == 2:
            #         trough_count += 1
            # peak_count -= 1
            # print(peak_count)
            # ax.set_title('respiratory rate')  # 之后加上获取的时间结合peak_count进行计算
            # ax.set_xlabel('frame')
            # ax.set_ylabel('respiratory data')
            # ax.set_yticks([])
            # ax.set_xticks([])
            # ax.legend()
            # canvas2.draw()
            # print('ok')


def create_file():
    global state_var
    state_var.set('执行新建文件...')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        subwin4 = tk.Tk()
        subwin4.title('新建文件')
        subwin4.geometry('300x120')
        subwin4.resizable(False, False)
        sublabe4 = tk.Label(subwin4, text='文件名：')
        sublabe4.pack()
        e4 = tk.Entry(subwin4, show=None)
        e4.pack()
        def get_path():
            my_file_name = e4.get()
            print('my_file_path:', my_file_name)
            if my_file_name is None:
                tk.messagebox.showerror(title='Hi',message='文件名为空！')
            elif my_file_name not in os.listdir(FILE_ROOT):
                os.mkdir(FILE_ROOT+my_file_name)
                submenu.add_radiobutton(label=my_file_name, command=import_file, variable=sub_file_name, value=my_file_name)
                subwin4.destroy()
                state_var.set('新建文件完成')
            else:
                tk.messagebox.showinfo(title='Hi',message='此文件已存在，请重命名文件')
                subwin4.destroy()
                state_var.set('等待操作...')

        button4 = tk.Button(subwin4, text='OK', command=get_path)
        button4.pack()


def import_file():  # 导入需要处理的文件并选择需要处理的图片数量
    global filename
    global PROCESS_IMAGE_NUM
    global state_var
    global p_var
    global l_var
    global EXTRACT_FINISH
    EXTRACT_FINISH=-1
    p_var.set(0)
    l_var.set('Processed ' + str(0) + ' / ' + str(100))
    state_var.set('执行导入深度图像文件...')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        filename = FILE_ROOT + sub_file_name.get()
        print(filename)
        number=len(os.listdir(filename))
        print(number)
        if number==0:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi', message='此文件夹为空，请先采集深度图像到此文件夹！')
            state_var.set('等待操作...')
        else:
            subwin3 = tk.Tk()
            subwin3.title('消息')
            subwin3.geometry('320x160')
            subwin3.resizable(False,False)
            text3 = tk.Text(subwin3)
            text3.pack()
            text3.insert('insert', sub_file_name.get()+' 已经导入！\n')
            pro_time=40./100*number
            if pro_time>=60:
                text3.insert('end', '此文件共有'+str(number)+'张图像，预计处理耗时'+str(pro_time//60)+'分钟'+str(pro_time%60)+'秒\n')
            else:
                text3.insert('end', '此文件共有' + str(number) + '张图像，预计处理耗时' + str(pro_time) + '秒\n')
            text3.config(state='disabled')
            label3=tk.Label(subwin3,text='请输入您要处理的图像数目：')
            label3.place(x=160,y=60,anchor='n')
            e1 = tk.Entry(subwin3, show=None)
            e1.place(x=160,y=90,anchor='n')
            def quit_and_get():
                global PROCESS_IMAGE_NUM
                PROCESS_IMAGE_NUM=int(e1.get())
                subwin3.destroy()
                state_var.set('深度图像文件导入完成')
            button3 = tk.Button(subwin3, text='OK', command=quit_and_get)
            button3.place(x=160,y=120,anchor='n')


def stop_extract():
    global state_var
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        state_var.set('暂停')
        tk.messagebox.showinfo(title='您好！', message='恢复请点击OK')
        state_var.set('执行呼吸数据提取...')


def save_file():
    global ins_filename
    global state_var
    global EXTRACT_FINISH
    state_var.set('执行保存呼吸数据文件...')
    if LOGIN_SUCCESS==-1:
        state_var.set('警告')
        tk.messagebox.showwarning(title='Hi',message='请先登录！')
        state_var.set('等待操作...')
    else:
        if EXTRACT_FINISH==-1:
            state_var.set('警告')
            tk.messagebox.showwarning(title='Hi',message='请先提取呼吸数据！')
            state_var.set('等待操作...')
        else:
            subwin=tk.Tk()
            subwin.title('另存为文件')
            subwin.geometry('300x120')
            subwin.resizable(False, False)
            sublabel=tk.Label(subwin,text='另存路径：')
            sublabel.pack()
            e1=tk.Entry(subwin,show=None)
            e1.pack()
            def get_path():
                my_file_path=e1.get()
                print('my_file_path:', my_file_path)
                print(ins_filename)
                shutil.copyfile(ins_filename, my_file_path+'.csv')
                tk.messagebox.showinfo(title='您好！', message='呼吸数据文件已另存到相应路径！')
                subwin.destroy()
                state_var.set('呼吸数据文件另存完成')
            button4 = tk.Button(subwin, text='OK', command=get_path)
            button4.pack()


def help_case():
    global state_var
    state_var.set('查看操作流程帮助')
    subwin1 = tk.Tk()
    subwin1.title('帮助')
    subwin1.geometry('400x160')
    subwin1.resizable(False, False)
    text1 = tk.Text(subwin1)
    text1.pack()
    text1.insert('insert', '首先，您需要登录系统，若您未在系统中，请进行注册；这是您操作本系统的前提！\n\n')
    text1.insert('end', '然后，您可以在文件菜单中选择新建文件，但是必须导入一个深度图像文件，这是您进行呼吸数据提取以及波形显示的前提！\n\n')
    text1.insert('end', '进行呼吸数据提取后，您可以查看呼吸波形及呼吸频率！\n\n')
    text1.config(state='disabled')

    def help_f():
        subwin1.destroy()
        state_var.set('等待操作...')

    button1 = tk.Button(subwin1, text='OK', command=help_f)
    button1.place(x=200, y=110, anchor='n')


def help_admin():
    global state_var
    state_var.set('查看管理操作帮助')
    subwin1 = tk.Tk()
    subwin1.title('帮助')
    subwin1.geometry('400x150')
    subwin1.resizable(False, False)
    text1 = tk.Text(subwin1)
    text1.pack()
    text1.insert('insert', '登录：\n用户登录与注册，用户信息将被保存\n\n')
    text1.insert('end', '退出：\n退出系统\n\n')
    text1.config(state='disabled')
    def help_f():
        subwin1.destroy()
        state_var.set('等待操作...')

    button1 = tk.Button(subwin1, text='OK', command=help_f)
    button1.place(x=200, y=110, anchor='n')


def help_file():
    global state_var
    state_var.set('查看文件操作帮助')
    subwin1 = tk.Tk()
    subwin1.title('帮助')
    subwin1.geometry('400x200')
    subwin1.resizable(False, False)
    text1 = tk.Text(subwin1)
    text1.pack()
    text1.insert('insert', '新建：\n用于新建文件，采集的深度图像将保存到此文件\n\n')
    text1.insert('end', '导入：\n用于选择需要进行呼吸检测的深度图像文件\n\n')
    text1.insert('end', '呼吸数据文件另存为：\n所有的呼吸数据文件均与深度图像文件对应并保存在本地文件夹中，'
                        '使用此操作可以将呼吸数据以excell文件另存到任意指定路径，输入路径格式如下：D:/file_data \n\n')
    text1.config(state='disabled')

    def help_f():
        subwin1.destroy()
        state_var.set('等待操作...')

    button1 = tk.Button(subwin1, text='OK', command=help_f)
    button1.place(x=200, y=160, anchor='n')


def help_detect():
    global state_var
    state_var.set('查看呼吸检测操作帮助')
    subwin2=tk.Tk()
    subwin2.title('帮助')
    subwin2.geometry('500x150')
    subwin2.resizable(False,False)
    text2=tk.Text(subwin2)
    text2.pack()
    text2.insert('insert','开始提取呼吸数据：\n此操作将提取深度图像的胸腔区域，根据此区域计算呼吸数据\n\n')
    text2.insert('end','暂停提取呼吸数据：\n此操作将暂停对当前文件呼吸数据的提取，点击OK恢复\n\n')
    text2.config(state='disabled')
    def help_f1():
        subwin2.destroy()
        state_var.set('等待操作...')
    button2 = tk.Button(subwin2, text='OK', command=help_f1)
    button2.place(x=250,y=110,anchor='n')


def help_show():
    global state_var
    state_var.set('查看显示操作帮助')
    subwin2 = tk.Tk()
    subwin2.title('帮助')
    subwin2.geometry('500x150')
    subwin2.resizable(False, False)
    text2 = tk.Text(subwin2)
    text2.pack()
    text2.insert('insert', '显示呼吸波形：\n此操作将根据呼吸数据绘制人体呼吸波形\n\n')
    text2.insert('end', '计算呼吸频率：\n此操作将对呼吸波形进行低通滤波和峰值检测以计算呼吸频率\n\n')
    text2.config(state='disabled')
    def help_f1():
        subwin2.destroy()
        state_var.set('等待操作...')
    button2 = tk.Button(subwin2, text='OK', command=help_f1)
    button2.place(x=250, y=110, anchor='n')


# 创建菜单和下级菜单----------------------------------------------------------------------------------------------------
menubar = tk.Menu(window)
# 0 用户登录
adminmenu=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='管理（A）',menu=adminmenu)
adminmenu.add_cascade(label='登录',command=user_login_window)
def win_quit():
    window.destroy()
adminmenu.add_cascade(label='退出',command=window.destroy)
# 1 文件菜单
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='文件（F）', menu=filemenu)
submenu = tk.Menu(filemenu)
# 创建import选项的radiobutton以得到选中文件夹的路径
sub_file_name = tk.StringVar()  # 用于存储菜单选中的待处理的文件名
L = len(os.listdir(FILE_ROOT))
for i in os.listdir(FILE_ROOT):
    submenu.add_radiobutton(label=i, command=import_file, variable=sub_file_name, value=i)
filemenu.add_command(label='新建', command=create_file)
filemenu.add_cascade(label='导入', menu=submenu)
filemenu.add_cascade(label='呼吸数据文件另存为',command=save_file)
# 2 呼吸检测菜单
detectmenu=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='呼吸检测（D）',menu=detectmenu)
submenu1=tk.Menu(detectmenu)
detectmenu.add_cascade(label='开始提取呼吸数据',command=extract_roi)
detectmenu.add_cascade(label='暂停提取呼吸数据',command=stop_extract)
# detectmenu.add_cascade(label='显示呼吸波形',command=waves)
# 3 显示菜单
showmenu=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='波形显示（S）',menu=showmenu)
submenu3=tk.Menu(showmenu)
# showmenu.add_cascade(label='显示呼吸波形',command=waves)
# showmenu.add_cascade(label='计算呼吸频率',command=compute_rr)
showmenu.add_cascade(label='显示呼吸波形',command=waves_pca)
showmenu.add_cascade(label='计算呼吸频率',command=compute_rr_pca)
# 4 帮助菜单
helpmenu=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='帮助（H）',menu=helpmenu)
submenu2=tk.Menu(helpmenu)
helpmenu.add_cascade(label='操作流程',command=help_case)
helpmenu.add_cascade(label='管理操作说明',command=help_admin)
helpmenu.add_cascade(label='文件操作说明',command=help_file)
helpmenu.add_cascade(label='呼吸检测操作说明',command=help_detect)
helpmenu.add_cascade(label='显示操作说明',command=help_show)

window.config(menu=menubar)


# canvas控件与matplotlib交互显示呼吸波形--------------------------------------------------------------------------------
fig = Figure(figsize=(3.2, 2.4))
ax = fig.add_subplot(111)
ax.set_xlabel('Frame')
ax.set_ylabel('Value')
canvas2 = FigureCanvasTkAgg(fig, master=window)
x = np.arange(0, 4 * np.pi, 0.1)  # 默认显示正弦波图像
y = np.sin(x)
# ax.plot(x, y, 'k')
# ax.set_yticks([])
# ax.set_xticks([])
canvas2.draw()
canvas2.get_tk_widget().place(x=540, y=32, anchor='n')
toolbar = NavigationToolbar2Tk(canvas2, window)
toolbar.place(x=380, y=276, anchor='nw')
toolbar.update()

def on_key_press(event):
    print('you press {}'.format(event.key))
    key_press_handler(event, canvas2, toolbar)

canvas2.mpl_connect('key_press_event', on_key_press)
# canvas1显示ROI提取图像------------------------------------------------------------------------------------------------
canvas1 = tk.Canvas(window, height=240, width=320,bg='white')
logo_img = Image.open('logo_img/logo1.jpg')  # # canvas显示jpg格式图片需要使用PIL进行转换，默认显示logo图像
logo_file = ImageTk.PhotoImage(logo_img)
# canvas1.create_image(162, 2, anchor='n', image=logo_file)
canvas1.place(x=200, y=30, anchor='n')

window.mainloop()
# ----------------------------------------------------------------------------------------------------------------------
