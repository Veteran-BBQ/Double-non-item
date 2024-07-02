import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sko.PSO import PSO
import matplotlib.pyplot as plt

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def train(X0):
    X1 = X0.cumsum(axis=0)
    Z1 = (np.array([-0.5 * (X1[:, -1][k - 1] + X1[:, -1][k])
                    for k in range(1, len(X1[:, -1]))]))
    Z = Z1.reshape(len(X1[:, -1]) - 1, 1)
    A = (X0[:, -1][1:]).reshape(len(Z), 1)
    B = np.hstack((Z, X1[1:, :-1]))
    u = np.linalg.inv(np.matmul(B.T, B)).dot(B.T).dot(A)
    a = u[0][0]
    b = u[1:]
    return a, b


def predict(k, X0, a, b):
    X1 = X0.cumsum(axis=0)
    f = lambda k, X1: ((X0[0, -1] - (1 / a) * (X1[k, ::]).dot(b))
                       * np.exp(-a * k) + (1 / a) * (X1[k, ::]).dot(b))
    X1_hat = [float(f(k, X1)) for k in range(k)]
    X0_hat = np.diff(X1_hat)
    X0_hat = np.hstack((X1_hat[0], X0_hat))
    return X0_hat


def evaluate(X0_hat, X0):
    S1 = np.std(X0, ddof=1)
    S2 = np.std(X0 - X0_hat, ddof=1)
    C = S2 / S1
    Pe = np.mean(X0 - X0_hat)
    temp = np.abs((X0 - X0_hat - Pe)) < 0.6745 * S1
    p = np.count_nonzero(temp) / len(X0)
    return np.sum(abs(X0 - X0_hat)), p


def cost(b):
    global X_train, X
    a1, _ = train(X_train)
    Y_pred = predict(len(X), X[:, :-1], a1, b)
    Y_train_pred = Y_pred[:len(X_train)]
    costs, _ = evaluate(Y_train_pred, X_train[:, -1])
    return costs
def plot_predict_results(Y_predict):
    predict_window = tk.Tk()
    predict_window.title("预测数据结果")
    predict_window.geometry("800x600")

    predict_canvas = FigureCanvasTkAgg(plt.figure(figsize=(8, 6)), master=predict_window)
    predict_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    ax = predict_canvas.figure.add_subplot(111)
    ax.plot(np.arange(len(Y_predict)), Y_predict, '-o')
    ax.set_title('预测数据')
    ax.grid(True)
    ax.legend(['预测负荷'])

    predict_canvas.draw()

    predict_window.mainloop()

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("灰色模型PSO优化.py")
        self.geometry(f"{1100}x{650}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)



        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=15)  ##边界度数
        self.sidebar_frame.grid(row=0, column=0,padx=(10, 10), pady=(10, 5), rowspan=4, sticky="nsew")  ##rowspan=4：框架会跨越4行
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="灰色模型+PSO优化",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(10, 0))
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="小白模式：使用预设参数，训练集占\n比80%。",
                                                 font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label.grid(row=1, column=0, padx=5, pady=(15, 0))
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="专业模式：可自行调整训练集占比，\n修改PSO超参数。",
                                                 font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label.grid(row=2, column=0, padx=5, pady=(10, 0))
        # 侧边栏按钮
        # self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.load_data)
        # self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.tabview = customtkinter.CTkTabview(self.sidebar_frame, width=250)
        self.tabview.grid(row=4, column=0, padx=(10, 10), pady=(120, 0), sticky="new")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.tabview.add("小白模式")
        self.tabview.add("专业模式")
        self.tabview.tab("小白模式").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("专业模式").grid_columnconfigure(0, weight=1)
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("小白模式"), text="加载数据",
                                                           command=self.load_data)
        self.string_input_button.grid(row=0, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("小白模式"), text="训练模型",
                                                           command=self.train_model)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("小白模式"), text="PSO优化",
                                                           command=self.run_pso)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("小白模式"), text="绘制结果",
                                                           command=self.plot_results)
        self.string_input_button.grid(row=3, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("小白模式"), text="导入预测数据",
                                                           command=self.load_predict_data)
        self.string_input_button.grid(row=4, column=0, padx=20, pady=(0, 10))


        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("专业模式"), text="加载数据",
                                                           command=self.load_data)
        self.string_input_button.grid(row=0, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("专业模式"), text="模型参数修改",
                                                           command=self.open_ratio_window)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("专业模式"), text="PSO参数修改",
                                                           command=self.open_pso_window)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("专业模式"), text="绘制结果",
                                                           command=self.plot_results)
        self.string_input_button.grid(row=3, column=0, padx=20, pady=(0, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("专业模式"), text="导入预测数据",
                                                           command=self.load_predict_data)
        self.string_input_button.grid(row=4, column=0, padx=20, pady=(0, 10))

        #预览窗口
        style = ttk.Style()
        style.configure("Treeview", background="gray", foreground="black", fieldbackground="black")
        # style.configure("Treeview.Heading", background="gray", foreground="black")

        self.tree = ttk.Treeview(self, show="headings", style="Treeview")

        self.tree.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew", rowspan=3)

        self.tree_scroll_y = customtkinter.CTkScrollbar(self, command=self.tree.yview)
        self.tree_scroll_y.grid(row=0, column=2, padx=(0, 5), pady=(10, 0), sticky="ns", rowspan=3)
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set)

        self.tree_scroll_x = customtkinter.CTkScrollbar(self, command=self.tree.xview, orientation="horizontal")
        self.tree_scroll_x.grid(row=2, column=1, padx=(0, 20), pady=(30, 0), sticky="esw", columnspan=2)
        self.tree.configure(xscrollcommand=self.tree_scroll_x.set)
        #提示框
        self.textbox = customtkinter.CTkTextbox(self,height=100, width=100, corner_radius=15)
        self.textbox.grid(row=4, column=0, padx=(5, 5), pady=(0, 5), columnspan=4,sticky="news")
        self.grid_rowconfigure(4, weight=1)
        # 创建图像显示区域
        self.figure = Figure(figsize=(5, 4), dpi=100)
        # self.figure.patch.set_facecolor('#2E2E2E')
        self.figure.patch.set_facecolor('white') # 设置背景颜色
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().configure(background='white', highlightcolor='white', highlightbackground='white')
        self.canvas.get_tk_widget().grid(row=3, column=1, padx=(10,0 ), pady=(10, 10), sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self.zoom)
        self.canvas.mpl_connect("scroll_event", self.hover)


        # 添加鼠标滚轮事件
        # self.figure = Figure(figsize=(5, 4), dpi=100)
        # self.canvas = FigureCanvasTkAgg(self.figure, self)
        # self.canvas.get_tk_widget().grid(row=3, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")

    def zoom(self, event):
        ax = self.figure.gca()
        if event.button == 'up':
            scale_factor = 1.1
        elif event.button == 'down':
            scale_factor = 0.9
        else:
            scale_factor = 1

        xdata, ydata = event.xdata, event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.canvas.draw()

    # def load_data(self):
    #     global X, X_train, X_test
    #     file_path = filedialog.askopenfilename()
    #     self.data = pd.read_csv(file_path)
    #     self.tree.delete(*self.tree.get_children())
    #     self.tree["columns"] = list(self.data.columns)
    #     for col in self.data.columns:
    #         self.tree.heading(col, text=col)
    #         self.tree.column(col, anchor="center")
    #
    #     # 插入行数据
    #     for index, row in self.data.iterrows():
    #         self.tree.insert("", "end", values=list(row))
    #
    #     # self.textbox2.insert(tk.END, str(self.data.head()))  # 显示前5行数据
    #     X = self.data.values
    #     X_train = X[:200, :]
    #     X_test = X[200:, :]
    #     self.textbox.insert(tk.END, "数据加载完成\n")
    #     self.textbox.see(tk.END)
    def load_data(self):
        global X, X_train, X_test
        file_path = filedialog.askopenfilename()
        self.data = pd.read_csv(file_path)

        # 获取数据信息
        num_variables = len(self.data.columns)
        num_records = len(self.data)

        # 输出导入数据的信息到textbox
        info_str = f"导入数据成功\n变量数: {num_variables}\n数据总数: {num_records}\n"
        self.textbox.insert(tk.END, info_str)
        self.textbox.see(tk.END)

        # 清空并更新tree
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.data.columns)
        for col in self.data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        # 插入行数据
        for index, row in self.data.iterrows():
            self.tree.insert("", "end", values=list(row))

        # 更新全局变量
        X = self.data.values
        X_train = X[:200, :]
        X_test = X[200:, :]

    def train_model(self):
        global a, b
        a, b = train(X_train)
        self.textbox.insert(tk.END, f"模型训练完成\n参数a: {a}\n参数b: {b}\n")
        self.textbox.see(tk.END)
    def run_pso(self):
        global pso
        pso = PSO(func=cost, n_dim=4, pop=50, max_iter=150, lb=[-10, -10, -10, -10], ub=[10, 10, 100, 10],
                  w=0.9, c1=0.5, c2=0.5)
        pso.run()
        self.gbest_x = pso.gbest_x
        self.gbest_y = pso.gbest_y
        self.textbox.insert(tk.END, f"PSO优化完成\n最佳位置: {self.gbest_x}\n最佳成本: {self.gbest_y}\n")
        self.textbox.see(tk.END)
    def plot_results(self):
        Y_pred = predict(len(X), X[:, :-1], a, self.gbest_x)
        Y_train_pred = Y_pred[:len(X_train)]
        Y_test_pred = Y_pred[len(X_train):]

        if 'X_predict' in globals():
            Y_predict = predict(len(X_predict), X_predict[:, :-1], a, self.gbest_x)
            self.textbox.insert(tk.END, "预测数据计算完成\n")
            self.textbox.see(tk.END)
        else:
            Y_predict = []

        _, p_train = evaluate(Y_train_pred, X_train[:, -1])
        _, p_test = evaluate(Y_test_pred, X_test[:, -1])

        self.textbox.insert(tk.END, f"训练集小误差概率: {p_train}\n")
        self.textbox.insert(tk.END, f"测试集小误差概率: {p_test}\n")
        self.textbox.see(tk.END)

        self.figure.clear()
        self.figure.set_size_inches(10, 4)
        ax = self.figure.add_subplot(111)  # 使用同一个 subplot

        ax.grid()
        ax.plot(np.arange(len(Y_train_pred)), X_train[:, -1], '->')
        ax.plot(np.arange(len(Y_train_pred)), Y_train_pred, '-o')
        ax.plot(np.arange(len(Y_test_pred)) + len(X_train), X_test[:, -1], '->')
        ax.plot(np.arange(len(Y_test_pred)) + len(X_train), Y_test_pred, '-o')

        if len(Y_predict) > 0:
            ax.plot(np.arange(len(Y_predict)) + len(X_train) + len(X_test), Y_predict, '-o')

        ax.legend(
            ['实际负荷（训练集）', '灰色模型预测（训练集）', '实际负荷（测试集）', '灰色模型预测（测试集）', '预测负荷'])
        ax.set_title('灰色模型预测结果')

        self.canvas.draw()

        if len(Y_predict) > 0:
            plot_predict_results(Y_predict)
            # ax_predict = self.figure.add_subplot(133)
            # ax_predict.grid()
            # ax_predict.plot(np.arange(len(Y_predict)), Y_predict, '-o')
            # ax_predict.legend(['预测负荷'])
            # ax_predict.set_title('预测数据')

        self.canvas.draw()
    def hover(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                x = self.scatter.get_offsets()[index, 0]
                y = self.scatter.get_offsets()[index, 1]
                self.annot.xy = (x, y)
                text = f'x: {x:.2f}, y: {y:.2f}'
                self.annot.set_text(text)
                self.annot.set_visible(True)
                self.figure.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.figure.canvas.draw_idle()

    def load_predict_data(self):
        global X_predict
        file_path = filedialog.askopenfilename()
        self.predict_data = pd.read_csv(file_path)
        X_predict = self.predict_data.values
        self.textbox.insert(tk.END, "预测数据加载完成\n")
        self.textbox.see(tk.END)

    def open_ratio_window(self):
        self.ratio_window = customtkinter.CTkToplevel(self)
        self.ratio_window.title("修改训练集占比")
        self.ratio_window.geometry("300x150")
        self.ratio_window.transient(self)  # 设置为self的子窗口
        self.ratio_window.grab_set()
        self.ratio_label = customtkinter.CTkLabel(self.ratio_window, text="训练集占比:")
        self.ratio_label.pack(pady=(20, 5))

        self.ratio_entry = customtkinter.CTkEntry(self.ratio_window)
        self.ratio_entry.pack(pady=(0, 20))

        self.confirm_button = customtkinter.CTkButton(self.ratio_window, text="确认", command=self.update_train_ratio)
        self.confirm_button.pack(pady=(0, 10))

    def update_train_ratio(self):
        global X, X_train, X_test
        try:
            ratio = float(self.ratio_entry.get())
            if 0 < ratio < 1:
                split_index = int(len(X) * ratio)
                X_train = X[:split_index, :]
                X_test = X[split_index:, :]
                self.textbox.insert(tk.END, f"训练集占比更新为 {ratio}\n")
                self.textbox.insert(tk.END, f"训练集大小: {len(X_train)}\n测试集大小: {len(X_test)}\n")
                self.train_model()
                self.ratio_window.destroy()
            else:
                self.textbox.insert(tk.END, "请输入介于 0 和 1 之间的数值\n")
        except ValueError:
            self.textbox.insert(tk.END, "请输入有效的数值\n")
        self.textbox.see(tk.END)

    def open_pso_window(self):
        self.pso_window = customtkinter.CTkToplevel(self)
        self.pso_window.title("修改PSO超参数")
        self.pso_window.geometry("300x600")
        self.pso_window.transient(self)  # 设置为self的子窗口
        self.pso_window.grab_set()
        self.pso_window.configure(fg_color=self._fg_color)

        self.pso_label = customtkinter.CTkLabel(self.pso_window, text="PSO 超参数")
        self.pso_label.pack(pady=(10, 5))

        self.pop_label = customtkinter.CTkLabel(self.pso_window, text="种群数量 (pop):")
        self.pop_label.pack(pady=(5, 5))
        self.pop_entry = customtkinter.CTkEntry(self.pso_window)
        self.pop_entry.pack(pady=(0, 5))

        self.iter_label = customtkinter.CTkLabel(self.pso_window, text="最大迭代次数 (max_iter):")
        self.iter_label.pack(pady=(5, 5))
        self.iter_entry = customtkinter.CTkEntry(self.pso_window)
        self.iter_entry.pack(pady=(0, 5))

        self.lb_label = customtkinter.CTkLabel(self.pso_window, text="下边界 (lb):")
        self.lb_label.pack(pady=(5, 5))
        self.lb_entry = customtkinter.CTkEntry(self.pso_window)
        self.lb_entry.pack(pady=(0, 5))

        self.ub_label = customtkinter.CTkLabel(self.pso_window, text="上边界 (ub):")
        self.ub_label.pack(pady=(5, 5))
        self.ub_entry = customtkinter.CTkEntry(self.pso_window)
        self.ub_entry.pack(pady=(0, 5))

        self.w_label = customtkinter.CTkLabel(self.pso_window, text="惯性权重 (w):")
        self.w_label.pack(pady=(5, 5))
        self.w_entry = customtkinter.CTkEntry(self.pso_window)
        self.w_entry.pack(pady=(0, 5))

        self.c1_label = customtkinter.CTkLabel(self.pso_window, text="个体学习因子 (c1):")
        self.c1_label.pack(pady=(5, 5))
        self.c1_entry = customtkinter.CTkEntry(self.pso_window)
        self.c1_entry.pack(pady=(0, 5))

        self.c2_label = customtkinter.CTkLabel(self.pso_window, text="社会学习因子 (c2):")
        self.c2_label.pack(pady=(5, 5))
        self.c2_entry = customtkinter.CTkEntry(self.pso_window)
        self.c2_entry.pack(pady=(0, 5))

        self.confirm_button = customtkinter.CTkButton(self.pso_window, text="确认", command=self.update_pso_params)
        self.confirm_button.pack(pady=(15, 10))

    def update_pso_params(self):
        try:
            pop = int(self.pop_entry.get())
            max_iter = int(self.iter_entry.get())
            lb = list(map(float, self.lb_entry.get().strip('[]').split(',')))
            ub = list(map(float, self.ub_entry.get().strip('[]').split(',')))
            w = float(self.w_entry.get())
            c1 = float(self.c1_entry.get())
            c2 = float(self.c2_entry.get())

            self.pso_params = {'pop': pop, 'max_iter': max_iter, 'lb': lb, 'ub': ub, 'w': w, 'c1': c1, 'c2': c2}
            self.textbox.insert(tk.END, f"PSO参数更新为:\n{self.pso_params}\n")

            # 调用 run_pso 方法进行 PSO 优化
            self.run_pso2()

            self.pso_window.destroy()
        except ValueError:
            self.textbox.insert(tk.END, "请输入有效的数值\n")
        self.textbox.see(tk.END)

    def run_pso2(self):
        global pso
        params = self.pso_params
        pso = PSO(func=cost, n_dim=4, pop=params['pop'], max_iter=params['max_iter'],
                  lb=params['lb'], ub=params['ub'], w=params['w'], c1=params['c1'], c2=params['c2'])
        pso.run()
        self.gbest_x = pso.gbest_x
        self.gbest_y = pso.gbest_y
        self.textbox.insert(tk.END, f"PSO优化完成\n最佳位置: {self.gbest_x}\n最佳成本: {self.gbest_y}\n")
        self.textbox.see(tk.END)


if __name__ == "__main__":
    app = App()
    app.mainloop()