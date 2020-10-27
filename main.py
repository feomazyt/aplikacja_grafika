import struct
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.colorchooser import askcolor
import PIL
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import array
import itertools


class Grafika(object):
    DEFAULT_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    ACTUAL_MODE = None

    def __init__(self):
        self.root = Tk()

        # TopBar
        self.topBar = Frame(self.root, width=700, height=30)
        self.topBar.pack(expand=False, fill='both', side='top', anchor='nw')

        # MainCanvas
        self.mainArea = Canvas(self.root, bg='white', width=700, height=700)
        self.mainArea.pack(expand=True, fill='both', side='left')
        self.mainArea.old_coords = None

        # menu
        self.menuBar = Menu(self.root)
        self.fileMenu = Menu(self.menuBar, tearoff=0)
        self.fileMenu.add_command(label="New", command=self.placeholder)
        self.fileMenu.add_command(label="Open", command=self.load_image)
        self.fileMenu.add_command(label="Save", command=self.save_image)
        self.menuBar.add_cascade(label="Plik", menu=self.fileMenu)

        # SideBar
        self.sidebar = Frame(self.root, width=200, bg='#CCC', height=600, relief='sunken', borderwidth=2)
        self.sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

        # Size
        self.size_button = Scale(self.topBar, from_=1, to=20, orient='horizontal')
        self.size_button.pack(side='right')

        # Pointer
        self.pointer_button = Button(self.sidebar, text='wskaźnik', command=self.pointer_click)
        self.pointer_button.pack(fill=X)

        # Clear
        self.clear_button = Button(self.sidebar, text='Wyczyść', command=self.clear)
        self.clear_button.pack(fill=X)

        # Pen
        self.pen_button = Button(self.sidebar, text='Ołówek', command=self.pen_click)
        self.pen_button.pack(fill=X)

        # Eraser
        self.eraser_button = Button(self.sidebar, text='Gumka', command=self.eraser_click)
        self.eraser_button.pack(fill=X)

        # Line
        self.line_button = Button(self.sidebar, text='Linia', command=self.draw_click)
        self.line_button.pack(fill=X)

        # Circle
        self.circle_button = Button(self.sidebar, text='Koło', command=self.circle_click)
        self.circle_button.pack(fill=X)

        # Rectangle
        self.rectangle_button = Button(self.sidebar, text='Prostokąt', command=self.rectangle_click)
        self.rectangle_button.pack(fill=X)

        # Resize+
        self.resize_plus_button = Button(self.sidebar, text='Zwiększ', command=self.resize_more)
        self.resize_plus_button.pack(fill=X)

        # Resize-
        self.resize_minus_button = Button(self.sidebar, text='Zmniejsz', command=self.resize_less)
        self.resize_minus_button.pack(fill=X)

        # preview original picture
        self.preview_original_button = Button(self.sidebar, text='Zobacz bazowy', command=self.preview_original)
        self.preview_original_button.pack(fill=X)

        # actual picture
        self.actual_img_button = Button(self.sidebar, text='aktualny obraz', command=self.actual_img)
        self.actual_img_button.pack(fill=X)

        # reset image
        self.reset_img_button = Button(self.sidebar, text='reset obrazu', command=self.reset_img)
        self.reset_img_button.pack(fill=X)

        # wygladzajacy
        self.avg_button = Button(self.sidebar, text='Wygladzający', command=lambda:self.convolution(
            kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
        self.avg_button.pack(fill=X)

        # median
        self.median_button = Button(self.sidebar, text='Median', command=self.median)
        self.median_button.pack(fill=X)

        # sobel
        self.sobel_button = Button(self.sidebar, text='Sobel poz', command=lambda : self.convolution(
            kernel=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])))
        self.sobel_button.pack(fill=X)

        self.sobel_button_2 = Button(self.sidebar, text='Sobel pion', command=lambda: self.convolution(
            kernel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])))
        self.sobel_button_2.pack(fill=X)

        # sharp
        self.sharp_button = Button(self.sidebar, text='Wyostrzający', command=lambda: self.convolution(
            kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])))
        self.sharp_button.pack(fill=X)

        # Gaussian blur
        self.Gaussian_button = Button(self.sidebar, text='Rozmycie Gauss', command=lambda: self.convolution(
            kernel=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])))
        self.Gaussian_button.pack(fill=X)

        # convolution
        self.convolution_button = Button(self.sidebar, text='Splot', command=self.ask_mask)
        self.convolution_button.pack(fill=X)

        # Adding
        self.adding_button = Button(self.sidebar, text='Dodawanie', command=self.placeholder)
        self.adding_button.pack(fill=X)

        # Sub
        self.sub_button = Button(self.sidebar, text='Odejmowanie', command=self.placeholder)
        self.sub_button.pack(fill=X)

        # Multi
        self.multi_button = Button(self.sidebar, text='mnożenie', command=self.placeholder)
        self.multi_button.pack(fill=X)

        # Divide
        self.divide_button = Button(self.sidebar, text='Dzielenie', command=self.placeholder)
        self.divide_button.pack(fill=X)

        # Bright
        self.bright_button = Button(self.sidebar, text='Rozjaśnianie', command=self.placeholder)
        self.bright_button.pack(fill=X)

        # Gray
        self.gray_button = Button(self.sidebar, text='skala szarości', command=self.gray)
        self.gray_button.pack(fill=X)

        self.RGB_label_Frame = Frame(self.root, width=20, height=100)
        self.RGB_label_Frame.pack()
        self.RGB_Frame = Frame(self.root, width=100, height=100)
        self.RGB_Frame.pack()
        self.CMYK_label_Frame = Frame(self.root, width=20, height=100)
        self.CMYK_label_Frame.pack()
        self.CMYK_Frame = Frame(self.root, width=100, height=100)
        self.CMYK_Frame.pack()
        self.preview_frame = Frame(self.root, width=100, height=100)
        self.preview_frame.pack()

        self.R_label = Label(self.RGB_label_Frame, text='RED').pack(side='left')
        self.G_label = Label(self.RGB_label_Frame, text='GREEN').pack(side='left')
        self.B_label = Label(self.RGB_label_Frame, text='BLUE').pack(side='left')

        self.R_value = DoubleVar()
        self.R_level = Scale(self.RGB_Frame, variable=self.R_value, to=255, command=self.convert_rgb_cmyk)
        self.R_level.pack(side='left')
        self.G_value = DoubleVar()
        self.G_level = Scale(self.RGB_Frame, variable=self.G_value, to=255, command=self.convert_rgb_cmyk)
        self.G_level.pack(side='left')
        self.B_value = DoubleVar()
        self.B_level = Scale(self.RGB_Frame, variable=self.B_value, to=255, command=self.convert_rgb_cmyk)
        self.B_level.pack(side='left')

        self.C_label = Label(self.CMYK_label_Frame, text='CYAN').pack(side='left')
        self.M_label = Label(self.CMYK_label_Frame, text='MAGENTA').pack(side='left')
        self.Y_label = Label(self.CMYK_label_Frame, text='YELLOW').pack(side='left')
        self.K_label = Label(self.CMYK_label_Frame, text='BLACK').pack(side='left')

        self.C_value = DoubleVar()
        self.C_level = Scale(self.CMYK_Frame, variable=self.C_value, to=100, command=self.convert_cmyk_rgb)
        self.C_level.pack(side='left')
        self.M_value = DoubleVar()
        self.M_level = Scale(self.CMYK_Frame, variable=self.M_value, to=100, command=self.convert_cmyk_rgb)
        self.M_level.pack(side='left')
        self.Y_value = DoubleVar()
        self.Y_level = Scale(self.CMYK_Frame, variable=self.Y_value, to=100, command=self.convert_cmyk_rgb)
        self.Y_level.pack(side='left')
        self.K_value = DoubleVar()
        self.K_level = Scale(self.CMYK_Frame, variable=self.K_value, to=100, command=self.convert_cmyk_rgb)
        self.K_level.pack(side='left')

        self.color_prev = Canvas(self.preview_frame, bg='white', width=100, height=100)
        self.color_prev.pack()

        self.setup()
        self.root.config(menu=self.menuBar)
        self.root.mainloop()

    def setup(self):
        self.edit_mode = False
        self.line_id = None
        self.edit_item_data = None
        self.drag_data_x = None
        self.drag_data_y = None
        self.old_x = None
        self.old_y = None
        self.line_width = self.size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.edit_line_1 = None
        self.edit_line_2 = None

    def gray(self):
        test = self.cv_im.ravel()
        shape = self.cv_im.shape
        tab = []
        for x in range(0, len(test), 3):
            r = int(test[x])
            g = int(test[x+1])
            b = int(test[x+2])
            tab.append(int((r+b+g)/3))
        self.cv_im = np.reshape(tab, (-1, shape[1]))
        pil_im = Image.fromarray(self.cv_im)
        self.image = ImageTk.PhotoImage(pil_im)
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def ask_mask(self):
        self.mask = Tk()
        mask_label = Label(self.mask, text="proszę wpisać dane maski oddzielone znakami białymi").pack()
        self.mask_entry = Text(self.mask)
        self.mask_entry.pack()
        ok_button = Button(self.mask, text="OK", command=self.create_mask)
        ok_button.pack()
        self.mask.mainloop()

    def create_mask(self):
        mask_data = self.mask_entry.get("1.0", END)
        self.mask.destroy()
        mask_data = list(map(int, mask_data.split()))
        if len(mask_data) % 2 == 0 or len(mask_data) <= 1:
            return
        size = int(math.sqrt(len(mask_data)))
        mask_data = np.reshape(mask_data, (-1, size))
        self.convolution(mask_data, size)

    def preview_original(self):
        self.mainArea.create_image(0, 0, image=self.base_img, anchor=NW)

    def actual_img(self):
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def reset_img(self):
        self.cv_im = np.array(self.pilImage)
        self.image = self.base_img
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def convert_rgb_cmyk(self, arg):
        try:
            R = self.R_level.get() / 255
        except:
            pass
        try:
            G = self.G_level.get() / 255
        except:
            pass
        try:
            B = self.B_level.get() / 255
        except:
            pass

        K = 1 - max(R, G, B)
        C = (1 - R - K) / (1 - K)
        M = (1 - G - K) / (1 - K)
        Y = (1 - B - K) / (1 - K)

        self.K_value.set(round(100 * K))
        self.C_value.set(round(100 * C))
        self.M_value.set(round(100 * M))
        self.Y_value.set(round(100 * Y))

        img = Image.new('RGB', (100, 100), (self.R_level.get(), self.G_level.get(), self.B_level.get()))
        self.imgp = ImageTk.PhotoImage(img, master=self.color_prev)
        self.color_prev.create_image(0, 0, image=self.imgp, anchor=NW)

    def convert_cmyk_rgb(self, arg):
        C = self.C_level.get() / 100
        M = self.M_level.get() / 100
        Y = self.Y_level.get() / 100
        K = self.K_level.get() / 100

        R = round(255 * (1 - C) * (1 - K))
        G = round(255 * (1 - M) * (1 - K))
        B = round(255 * (1 - Y) * (1 - K))

        self.R_value.set(R)
        self.G_value.set(G)
        self.B_value.set(B)

        img = Image.new('CMYK', (100, 100), (round(C * 255), round(M * 255), round(Y * 255), round(K * 255)))
        self.imgp = ImageTk.PhotoImage(img, master=self.color_prev)
        self.color_prev.create_image(0, 0, image=self.imgp, anchor=NW)

    def median(self):
        width, height = self.pilImage.width, self.pilImage.height
        temp = np.zeros((height + 2, width + 2, 3))
        for x in range(1, height + 1):
            for y in range(1, width + 1):
                for z in range(3):
                    temp[x][y][z] = self.cv_im[x - 1][y - 1][z]
        # temp = self.im.copy()
        for x in range(1, height + 1):
            for y in range(1, width + 1):
                for z in range(3):
                    x1 = temp[x - 1][y - 1][z]
                    x2 = temp[x - 1][y][z]
                    x3 = temp[x - 1][y + 1][z]
                    x4 = temp[x][y - 1][z]
                    x5 = temp[x][y][z]
                    x6 = temp[x][y + 1][z]
                    x7 = temp[x + 1][y - 1][z]
                    x8 = temp[x + 1][y][z]
                    x9 = temp[x + 1][y + 1][z]

                    tab = [x1, x2, x3, x4, x5,
                           x6, x7, x8, x9]
                    tab.sort()
                    median = tab[4]
                    self.cv_im[x - 1][y - 1][z] = median

        pil_im = Image.fromarray(self.cv_im)
        self.image = ImageTk.PhotoImage(pil_im)
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def convolution(self, kernel, size=3):

        kernel = np.flipud(np.fliplr(kernel))
        ksum = kernel.sum()
        if ksum <= 0:
            ksum = 1
        output = np.zeros_like(self.cv_im)

        margin = math.floor(size/2)

        if len(self.cv_im.shape) == 3:
            image_padded = np.zeros((self.cv_im.shape[0] + (margin*2), self.cv_im.shape[1] + (margin*2), 3))
            image_padded[margin:-margin, margin:-margin] = self.cv_im

            for x in range(self.cv_im.shape[1]):
                for y in range(self.cv_im.shape[0]):
                    for z in range(3):
                        # element-wise multiplication of the kernel and the image
                        sum = (kernel * image_padded[y: y + size, x: x + size, z]).sum()
                        sum = sum/ksum
                        if sum > 255:
                            sum = 255
                        elif sum < 0:
                            sum = 0
                        output[y, x, z] = sum
        else:
            image_padded = np.zeros((self.cv_im.shape[0] + (margin * 2), self.cv_im.shape[1] + (margin * 2)))
            image_padded[margin:-margin, margin:-margin] = self.cv_im

            for x in range(self.cv_im.shape[1]):
                for y in range(self.cv_im.shape[0]):
                    # element-wise multiplication of the kernel and the image
                    sum = (kernel * image_padded[y: y + size, x: x + size]).sum()
                    sum = sum / ksum
                    if sum > 255:
                        sum = 255
                    elif sum < 0:
                        sum = 0
                    output[y, x] = sum

        self.cv_im = output
        pil_im = Image.fromarray(self.cv_im)
        self.image = ImageTk.PhotoImage(pil_im)
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def resize_more(self):
        self.pilImage = self.pilImage.resize((self.pilImage.width * 2, self.pilImage.height * 2), Image.AFFINE)
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def resize_less(self):
        self.pilImage = self.pilImage.resize((round(self.pilImage.width / 2), round(self.pilImage.height / 2)),
                                             Image.AFFINE)
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.mainArea.create_image(0, 0, image=self.image, anchor=NW)

    def load_image(self):
        try:
            path = filedialog.askopenfilename(initialdir="/", title="Wybierz plik", filetypes=(
                ('JPEG', ('*.jpg', '*.jpeg', '*.jpe')),
                ('PPM', '*.ppm')))

            if path.endswith('.ppm') and 'p3' in path:
                # mode
                # width
                # height
                # max color

                seconds = time.time()
                print('start')

                ppm = open(path, 'r')
                data = ppm.read()
                if len(data) > 1000000:
                    data1 = data[:179]
                    data2 = data[179:]
                    data1 = re.sub(r'(?m) *#.*\n?', '', data1)
                    encoding, width, height, color, *values = data1.split()
                    data = values + data2.split()
                    data = list(map(int, data))
                else:
                    data = re.sub(r'(?m) *#.*\n?', '', data)
                    encoding, width, height, color, *values = data.split()
                    data = list(map(int, values))

                def grouper(n, iterable):
                    args = [iter(iterable)] * n
                    return zip(*args)

                if int(color) == 255:

                    tab = list(grouper(3, data))

                else:
                    def scale(x):
                        return round(x * (1 / int(color) * 255))

                    data = list(map(scale, data))
                    tab = list(grouper(3, data))

                self.pilImage = Image.new('RGB', (int(width), int(height)))
                self.pilImage.putdata(tab)
                self.cv_im = np.array(self.pilImage)
                self.base_img = ImageTk.PhotoImage(self.pilImage)
                self.image = ImageTk.PhotoImage(self.pilImage)
                self.mainArea.create_image(0, 0, image=self.image, anchor=NW)
                print('stop', (time.time() - seconds))

            elif path.endswith('.ppm') and 'p6' in path:
                # mode
                # width
                # height
                # max color

                f = open(path, 'rb')
                self.pmmdata = {
                    'mode': None,
                    'width': None,
                    'height': None,
                    'colormax': None,
                }

                def fill_pmmdata_byte(f):
                    array = None
                    for i, line in enumerate(f):
                        line = line.decode('charmap')
                        if line[0] == '#':
                            continue
                        line = line.split()
                        for x in line:
                            if x == '#' or x[0] == '#':
                                break
                            if self.pmmdata['mode'] is None or self.pmmdata['width'] is None or \
                                    self.pmmdata['height'] is None or self.pmmdata['colormax'] is None:

                                if self.pmmdata['mode'] is None:
                                    self.pmmdata['mode'] = x
                                elif self.pmmdata['width'] is None:
                                    self.pmmdata['width'] = int(x)
                                elif self.pmmdata['height'] is None:
                                    self.pmmdata['height'] = int(x)
                                elif self.pmmdata['colormax'] is None:
                                    self.pmmdata['colormax'] = int(x)
                            if array is None and self.pmmdata['height'] is not None and self.pmmdata[
                                'width'] is not None \
                                    and self.pmmdata['colormax'] is not None:
                                array = np.zeros((self.pmmdata['height'], self.pmmdata['width'], 3), np.uint8)
                                return

                fill_pmmdata_byte(f)
                width = self.pmmdata['width']
                height = self.pmmdata['height']
                seconds = time.time()
                print('start')

                def grouper(n, iterable):
                    args = [iter(iterable)] * n
                    return zip(*args)

                tab = []
                for i in range(0, height * width, width):
                    test = struct.unpack(str(width * 3) + 'B', f.read(width * 3))
                    res = list(grouper(3, test))
                    tab += res

                self.pilImage = Image.new('RGB', (width, height))
                self.pilImage.putdata(tab)
                self.cv_im = np.array(self.pilImage)
                self.base_img = ImageTk.PhotoImage(self.pilImage)
                self.image = ImageTk.PhotoImage(self.pilImage)
                self.mainArea.create_image(0, 0, image=self.image, anchor=NW)
                print('stop', (time.time() - seconds))

            else:
                self.pilImage = Image.open(path)
                self.cv_im = np.array(self.pilImage)
                self.base_img = ImageTk.PhotoImage(self.pilImage)
                self.image = ImageTk.PhotoImage(self.pilImage)
                self.mainArea.create_image(0, 0, image=self.image, anchor=NW)
        except:
            messagebox.showerror("Nie wczytano", "Wystąpił błąd!")

    def save_image(self):
        self.path = filedialog.asksaveasfilename(title="Zapisz plik", defaultextension=".jpg")
        self.dialog = Tk()
        frame = Frame(self.dialog, width=80, height=30)
        frame.pack()
        level_label = Label(frame, width=50, text='Proszę podać stopień kompresji')
        level_label.pack()
        self.compress_button = Scale(frame, from_=0, to=100, orient='horizontal')
        self.compress_button.pack()
        ok_button = Button(frame, text='OK', command=self.save_image2)
        ok_button.pack()

    def save_image2(self):
        quality = self.compress_button.get()
        self.dialog.destroy()
        print(quality)
        try:
            self.pilImage.save(self.path, format='JPEG', subsampling=0, quality=quality)
            messagebox.showinfo("plik zapisany", "plik został zapisany")
        except:
            messagebox.showerror("Nie zapisano", "Wystąpił błąd")

    def TopBar_config(self, mode):
        self.ACTUAL_MODE = mode
        # self.topBar.pack_forget()
        for widget in self.topBar.winfo_children():
            widget.destroy()

        # Size
        self.size_button = Scale(self.topBar, from_=1, to=10, orient='horizontal')
        self.size_button.pack(side='right')

        if self.ACTUAL_MODE == 'line':
            if self.edit_mode == True:
                # Delete
                self.delete_button = Button(self.topBar, text='usuń', command=self.delete_object)
                self.delete_button.pack(side='right')
            # Line Start
            self.labels_start = Frame(self.topBar)
            self.labels_start.pack(side='left')
            self.start_x_lbl = Label(self.labels_start, width=10, text='start x')
            self.start_x_lbl.pack()
            self.start_y_lbl = Label(self.labels_start, width=10, text='start y')
            self.start_y_lbl.pack()

            self.entry_start = Frame(self.topBar)
            self.entry_start.pack(side='left')
            self.start_x = Entry(self.entry_start, width=10)
            self.start_x.pack()
            self.start_y = Entry(self.entry_start, width=10)
            self.start_y.pack()

            # Line End
            self.labels_end = Frame(self.topBar)
            self.labels_end.pack(side='left')
            self.end_x_lbl = Label(self.labels_end, width=10, text='koniec x')
            self.end_x_lbl.pack()
            self.end_y_lbl = Label(self.labels_end, width=10, text='koniec y')
            self.end_y_lbl.pack(side='left', fill=X)

            self.entry_end = Frame(self.topBar)
            self.entry_end.pack(side='left')
            self.end_x = Entry(self.entry_end, width=10)
            self.end_x.pack()
            self.end_y = Entry(self.entry_end, width=10)
            self.end_y.pack()

            # Line
            self.line_coords_button = Button(self.topBar, text='Ok', command=self.create_line_click)
            self.line_coords_button.pack(side='left')

        elif self.ACTUAL_MODE == 'circle':
            if self.edit_mode == True:
                # Delete
                self.delete_button = Button(self.topBar, text='usuń', command=self.delete_object)
                self.delete_button.pack(side='right')
            # circle center
            self.labels_center = Frame(self.topBar)
            self.labels_center.pack(side='left')
            self.center_x_lbl = Label(self.labels_center, width=10, text='środek x')
            self.center_x_lbl.pack()
            self.center_y_lbl = Label(self.labels_center, width=10, text='środek y')
            self.center_y_lbl.pack()

            self.entry_center = Frame(self.topBar)
            self.entry_center.pack(side='left')
            self.center_x = Entry(self.entry_center, width=10)
            self.center_x.pack()
            self.center_y = Entry(self.entry_center, width=10)
            self.center_y.pack()

            # Line End
            self.labels_radius = Frame(self.topBar)
            self.labels_radius.pack(side='left')
            self.radius_x_lbl = Label(self.labels_radius, width=10, text='promień x')
            self.radius_x_lbl.pack()
            self.radius_y_lbl = Label(self.labels_radius, width=10, text='promień y')
            self.radius_y_lbl.pack(side='left', fill=X)

            self.entry_radius = Frame(self.topBar)
            self.entry_radius.pack(side='left')
            self.radius_x = Entry(self.entry_radius, width=10)
            self.radius_x.pack()
            self.radius_y = Entry(self.entry_radius, width=10)
            self.radius_y.pack()

            # Line
            self.line_coords_button = Button(self.topBar, text='Ok', command=self.create_line_click)
            self.line_coords_button.pack(side='left')

        elif self.ACTUAL_MODE == 'rectangle':
            if self.edit_mode == True:
                # Delete
                self.delete_button = Button(self.topBar, text='usuń', command=self.delete_object)
                self.delete_button.pack(side='right')
            # rectangle center
            self.labels_corner_1 = Frame(self.topBar)
            self.labels_corner_1.pack(side='left')
            self.corner_1_x_lbl = Label(self.labels_corner_1, text='x1')
            self.corner_1_x_lbl.pack()
            self.corner_1_y_lbl = Label(self.labels_corner_1, text='y1')
            self.corner_1_y_lbl.pack()

            self.entry_corner_1 = Frame(self.topBar)
            self.entry_corner_1.pack(side='left')
            self.corner_1_x = Entry(self.entry_corner_1, width=10)
            self.corner_1_x.pack()
            self.corner_1_y = Entry(self.entry_corner_1, width=10)
            self.corner_1_y.pack()

            # Line End
            self.labels_corner_2 = Frame(self.topBar)
            self.labels_corner_2.pack(side='left')
            self.corner_2_x_lbl = Label(self.labels_corner_2, text='x2')
            self.corner_2_x_lbl.pack()
            self.corner_2_y_lbl = Label(self.labels_corner_2, text='y2')
            self.corner_2_y_lbl.pack(side='left', fill=X)

            self.entry_corner_2 = Frame(self.topBar)
            self.entry_corner_2.pack(side='left')
            self.corner_2_x = Entry(self.entry_corner_2, width=10)
            self.corner_2_x.pack()
            self.corner_2_y = Entry(self.entry_corner_2, width=10)
            self.corner_2_y.pack()

            # Line
            self.line_coords_button = Button(self.topBar, text='Ok', command=self.create_line_click)
            self.line_coords_button.pack(side='left')

    def delete_object(self):
        self.mainArea.delete(self.edit_line_1)
        self.mainArea.delete(self.edit_line_2)
        self.mainArea.delete(self.edit_item_data['id'])
        self.TopBar_config(None)
        self.edit_mode = False

    def on_line_click(self, event):
        self.mainArea.delete(self.edit_line_1)
        self.mainArea.delete(self.edit_line_2)
        self.edit_mode = True
        self.TopBar_config('line')
        canvas_item_id = event.widget.find_withtag('current')[0]
        x2, y2, x1, y1 = self.mainArea.coords(canvas_item_id)
        self.edit_item_data = {'id': canvas_item_id,
                               'x1': x2,
                               'y1': y2,
                               'x2': x1,
                               'y2': y1
                               }
        self.start_x.delete(0, END)
        self.start_x.insert(0, x1)
        self.start_y.delete(0, END)
        self.start_y.insert(0, y1)
        self.end_x.delete(0, END)
        self.end_x.insert(0, x2)
        self.end_y.delete(0, END)
        self.end_y.insert(0, y2)

        self.mainArea.tag_bind(canvas_item_id, "<ButtonPress-1>", self.line_move)
        self.mainArea.tag_bind(canvas_item_id, "<ButtonRelease-1>", self.line_move)
        self.mainArea.tag_bind(canvas_item_id, "<B1-Motion>", self.line_move)
        self.edit_line_1 = self.mainArea.create_rectangle(x1 - 3, y1 + 3, x1 + 3, y1 - 3, fill='black')
        self.edit_line_2 = self.mainArea.create_rectangle(x2 - 3, y2 + 3, x2 + 3, y2 - 3, fill='black')
        self.mainArea.tag_bind(self.edit_line_1,
                               '<B1-Motion>',
                               lambda event,
                                      arg='start':
                               self.edit_line(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='start':
                               self.edit_line(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='start':
                               self.edit_line(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<B1-Motion>',
                               lambda event,
                                      arg='end':
                               self.edit_line(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='end':
                               self.edit_line(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='end':
                               self.edit_line(event, arg))

    def on_circle_click(self, event):
        self.mainArea.delete(self.edit_line_1)
        self.mainArea.delete(self.edit_line_2)
        self.edit_mode = True
        self.TopBar_config('circle')
        canvas_item_id = event.widget.find_withtag('current')[0]
        x1, y1, x2, y2 = self.mainArea.coords(canvas_item_id)
        r1 = abs(x2 - x1) / 2
        r2 = abs(y2 - y1) / 2
        x = (x1 - r1) if x1 > x2 else (x2 - r1)
        y = (y1 - r2) if y1 > y2 else (y2 - r2)
        self.edit_item_data = {'id': canvas_item_id,
                               'x': x,
                               'y': y,
                               'r1': r1,
                               'r2': r2,
                               }
        self.center_x.delete(0, END)
        self.center_x.insert(0, x)
        self.center_y.delete(0, END)
        self.center_y.insert(0, y)
        self.radius_x.delete(0, END)
        self.radius_x.insert(0, r1)
        self.radius_y.delete(0, END)
        self.radius_y.insert(0, r2)
        self.mainArea.tag_bind(canvas_item_id, "<ButtonPress-1>", self.circle_move)
        self.mainArea.tag_bind(canvas_item_id, "<ButtonRelease-1>", self.circle_move)
        self.mainArea.tag_bind(canvas_item_id, "<B1-Motion>", self.circle_move)
        self.edit_line_1 = self.mainArea.create_rectangle(x - 3, y - r2 + 3, x + 3, y - r2 - 3, fill='black')
        self.edit_line_2 = self.mainArea.create_rectangle(x + r1 - 3, y + 3, x + r1 + 3, y - 3, fill='black')
        self.mainArea.tag_bind(self.edit_line_1,
                               '<B1-Motion>',
                               lambda event,
                                      arg='top':
                               self.edit_circle(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='top':
                               self.edit_circle(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='top':
                               self.edit_circle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<B1-Motion>',
                               lambda event,
                                      arg='right':
                               self.edit_circle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='right':
                               self.edit_circle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='right':
                               self.edit_circle(event, arg))

    def on_rectangle_click(self, event):
        self.mainArea.delete(self.edit_line_1)
        self.mainArea.delete(self.edit_line_2)
        self.edit_mode = True
        self.TopBar_config('rectangle')
        canvas_item_id = event.widget.find_withtag('current')[0]
        x1, y2, x2, y1 = self.mainArea.coords(canvas_item_id)
        self.edit_item_data = {'id': canvas_item_id,
                               'x1': x1,
                               'y1': y1,
                               'x2': x2,
                               'y2': y2
                               }

        self.corner_1_x.delete(0, END)
        self.corner_1_x.insert(0, x1)
        self.corner_1_y.delete(0, END)
        self.corner_1_y.insert(0, y1)
        self.corner_2_x.delete(0, END)
        self.corner_2_x.insert(0, x2)
        self.corner_2_y.delete(0, END)
        self.corner_2_y.insert(0, y2)
        self.mainArea.tag_bind(canvas_item_id, "<ButtonPress-1>", self.rectangle_move)
        self.mainArea.tag_bind(canvas_item_id, "<ButtonRelease-1>", self.rectangle_move)
        self.mainArea.tag_bind(canvas_item_id, "<B1-Motion>", self.rectangle_move)
        self.edit_line_1 = self.mainArea.create_rectangle(x1 - 3, y1 + 3, x1 + 3, y1 - 3, fill='black')
        self.edit_line_2 = self.mainArea.create_rectangle(x2 - 3, y2 + 3, x2 + 3, y2 - 3, fill='black')
        self.mainArea.tag_bind(self.edit_line_1,
                               '<B1-Motion>',
                               lambda event,
                                      arg='corner_1':
                               self.edit_rectangle(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='corner_1':
                               self.edit_rectangle(event, arg))
        self.mainArea.tag_bind(self.edit_line_1,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='corner_1':
                               self.edit_rectangle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<B1-Motion>',
                               lambda event,
                                      arg='corner_2':
                               self.edit_rectangle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonPress-1>',
                               lambda event,
                                      arg='corner_2':
                               self.edit_rectangle(event, arg))
        self.mainArea.tag_bind(self.edit_line_2,
                               '<ButtonRelease-1>',
                               lambda event,
                                      arg='corner_2':
                               self.edit_rectangle(event, arg))

    def line_move(self, event):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y
        elif str(event.type) == 'ButtonRelease':
            x2, y2, x1, y1 = self.mainArea.coords(self.edit_item_data['id'])
            self.edit_item_data['x1'] = x2
            self.edit_item_data['y1'] = y2
            self.edit_item_data['x2'] = x1
            self.edit_item_data['y2'] = y1
        elif str(event.type) == 'Motion':
            x2, y2, x1, y1 = self.mainArea.coords(self.edit_item_data['id'])
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            self.mainArea.move(self.edit_item_data['id'], delta_x, delta_y)
            self.mainArea.move(self.edit_line_1, delta_x, delta_y)
            self.mainArea.move(self.edit_line_2, delta_x, delta_y)
            self.start_x.delete(0, END)
            self.start_x.insert(0, int(x1))
            self.start_y.delete(0, END)
            self.start_y.insert(0, int(y1))
            self.end_x.delete(0, END)
            self.end_x.insert(0, int(x2))
            self.end_y.delete(0, END)
            self.end_y.insert(0, int(y2))
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def circle_move(self, event):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y
        elif str(event.type) == 'ButtonRelease':
            x1, y1, x2, y2 = self.mainArea.coords(self.edit_item_data['id'])
            r1 = abs(x2 - x1) / 2
            r2 = abs(y2 - y1) / 2
            x = (x1 - r1) if x1 > x2 else (x2 - r1)
            y = (y1 - r2) if y1 > y2 else (y2 - r2)

            self.edit_item_data['x'] = x
            self.edit_item_data['y'] = y
            self.edit_item_data['r1'] = r1
            self.edit_item_data['r2'] = r2
        elif str(event.type) == 'Motion':
            x1, y1, x2, y2 = self.mainArea.coords(self.edit_item_data['id'])
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            self.mainArea.move(self.edit_item_data['id'], delta_x, delta_y)
            self.mainArea.move(self.edit_line_1, delta_x, delta_y)
            self.mainArea.move(self.edit_line_2, delta_x, delta_y)
            x1, y1, x2, y2 = self.mainArea.coords(self.edit_item_data['id'])
            r1 = abs(x2 - x1) / 2
            r2 = abs(y2 - y1) / 2
            x = (x1 - r1) if x1 > x2 else (x2 - r1)
            y = (y1 - r2) if y1 > y2 else (y2 - r2)
            self.center_x.delete(0, END)
            self.center_x.insert(0, x)
            self.center_y.delete(0, END)
            self.center_y.insert(0, y)
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def rectangle_move(self, event):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y
        elif str(event.type) == 'ButtonRelease':

            self.edit_item_data['x1'] = self.corner_1_x.get()
            self.edit_item_data['y1'] = self.corner_1_y.get()
            self.edit_item_data['x2'] = self.corner_2_x.get()
            self.edit_item_data['y2'] = self.corner_2_y.get()
        elif str(event.type) == 'Motion':
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            self.mainArea.move(self.edit_item_data['id'], delta_x, delta_y)
            self.mainArea.move(self.edit_line_1, delta_x, delta_y)
            self.mainArea.move(self.edit_line_2, delta_x, delta_y)
            # x1, y2, x2, y1 = self.mainArea.coords(self.edit_item_data['id'])
            x1 = float(self.corner_1_x.get()) + delta_x
            y1 = float(self.corner_1_y.get()) + delta_y
            x2 = float(self.corner_2_x.get()) + delta_x
            y2 = float(self.corner_2_y.get()) + delta_y

            self.corner_1_x.delete(0, END)
            self.corner_1_x.insert(0, x1)
            self.corner_1_y.delete(0, END)
            self.corner_1_y.insert(0, y1)
            self.corner_2_x.delete(0, END)
            self.corner_2_x.insert(0, x2)
            self.corner_2_y.delete(0, END)
            self.corner_2_y.insert(0, y2)
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def edit_line(self, event, arg):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y

        elif str(event.type) == 'ButtonRelease':
            if arg == 'start':
                self.edit_item_data['x2'] = event.x
                self.edit_item_data['y2'] = event.y
            else:
                self.edit_item_data['x1'] = event.x
                self.edit_item_data['y1'] = event.y

        elif str(event.type) == 'Motion':
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            if arg == 'start':
                self.mainArea.coords(self.edit_item_data['id'], self.edit_item_data['x1'], self.edit_item_data['y1'],
                                     event.x, event.y)
                self.mainArea.move(self.edit_line_1, delta_x, delta_y)
                self.start_x.delete(0, END)
                self.start_x.insert(0, event.x)
                self.start_y.delete(0, END)
                self.start_y.insert(0, event.y)
            else:
                self.mainArea.coords(self.edit_item_data['id'], event.x, event.y, self.edit_item_data['x2'],
                                     self.edit_item_data['y2'])
                self.mainArea.move(self.edit_line_2, delta_x, delta_y)
                self.end_x.delete(0, END)
                self.end_x.insert(0, event.x)
                self.end_y.delete(0, END)
                self.end_y.insert(0, event.y)
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def edit_circle(self, event, arg):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y

        elif str(event.type) == 'ButtonRelease':
            if arg == 'top':
                self.edit_item_data['r2'] = abs(self.edit_item_data['y'] - event.y)
            else:
                self.edit_item_data['r1'] = abs(self.edit_item_data['x'] - event.x)

        elif str(event.type) == 'Motion':
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            if arg == 'top':
                r = abs(self.edit_item_data['y'] - event.y)
                self.mainArea.coords(self.edit_item_data['id'],
                                     self.edit_item_data['x'] - self.edit_item_data['r1'],
                                     self.edit_item_data['y'] - r,
                                     self.edit_item_data['x'] + self.edit_item_data['r1'],
                                     self.edit_item_data['y'] + r)
                self.mainArea.move(self.edit_line_1, 0, delta_y)
                self.radius_y.delete(0, END)
                self.radius_y.insert(0, r)
                self.radius_y.delete(0, END)
                self.radius_y.insert(0, r)
            else:
                r = abs(self.edit_item_data['x'] - event.x)
                self.mainArea.coords(self.edit_item_data['id'],
                                     self.edit_item_data['x'] - r,
                                     self.edit_item_data['y'] - self.edit_item_data['r2'],
                                     self.edit_item_data['x'] + r,
                                     self.edit_item_data['y'] + self.edit_item_data['r2'])
                self.mainArea.move(self.edit_line_2, delta_x, 0)
                self.radius_x.delete(0, END)
                self.radius_x.insert(0, r)
                self.radius_x.delete(0, END)
                self.radius_x.insert(0, r)
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def edit_rectangle(self, event, arg):
        if str(event.type) == 'ButtonPress':
            self.drag_data_x = event.x
            self.drag_data_y = event.y

        elif str(event.type) == 'ButtonRelease':
            x1, y2, x2, y1 = self.mainArea.coords(self.edit_item_data['id'])

            self.edit_item_data['x1'] = self.corner_1_x.get()
            self.edit_item_data['y1'] = self.corner_1_y.get()
            self.edit_item_data['x2'] = self.corner_2_x.get()
            self.edit_item_data['y2'] = self.corner_2_y.get()

        elif str(event.type) == 'Motion':
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            if arg == 'corner_2':
                self.mainArea.coords(self.edit_item_data['id'],
                                     self.edit_item_data['x1'],
                                     self.edit_item_data['y1'],
                                     event.x,
                                     event.y)
                self.mainArea.move(self.edit_line_2, delta_x, delta_y)
                self.corner_2_x.delete(0, END)
                self.corner_2_x.insert(0, event.x)
                self.corner_2_y.delete(0, END)
                self.corner_2_y.insert(0, event.y)
            else:
                self.mainArea.coords(self.edit_item_data['id'],
                                     event.x,
                                     event.y,
                                     self.edit_item_data['x2'],
                                     self.edit_item_data['y2'])
                self.mainArea.move(self.edit_line_1, delta_x, delta_y)
                self.corner_1_x.delete(0, END)
                self.corner_1_x.insert(0, event.x)
                self.corner_1_y.delete(0, END)
                self.corner_1_y.insert(0, event.y)
            self.drag_data_x = event.x
            self.drag_data_y = event.y

    def clear(self):
        self.ACTUAL_MODE = None
        self.mainArea.delete('all')

    def create_line_click(self):
        if self.edit_mode:
            if self.ACTUAL_MODE == 'line':
                self.mainArea.coords(self.edit_item_data['id'],
                                     self.end_x.get(), self.end_y.get(),
                                     self.start_x.get(), self.start_y.get())
                self.mainArea.coords(self.edit_item_data['id'],
                                     self.start_x.get(), self.start_y.get(),
                                     self.end_x.get(), self.end_y.get())
                self.mainArea.coords(self.edit_line_2,
                                     float(self.end_x.get()) - 3,
                                     float(self.end_y.get()) + 3,
                                     float(self.end_x.get()) + 3,
                                     float(self.end_y.get()) - 3)

                self.mainArea.coords(self.edit_line_1,
                                     float(self.start_x.get()) - 3,
                                     float(self.start_y.get()) + 3,
                                     float(self.start_x.get()) + 3,
                                     float(self.start_y.get()) - 3)
                self.mainArea.itemconfig(self.edit_item_data['id'],
                                         width=self.size_button.get())

            if self.ACTUAL_MODE == 'circle':
                self.mainArea.coords(self.edit_item_data['id'],
                                     float(self.center_x.get()) - float(self.radius_x.get()),
                                     float(self.center_y.get()) - float(self.radius_y.get()),
                                     float(self.center_x.get()) + float(self.radius_x.get()),
                                     float(self.center_y.get()) + float(self.radius_y.get()))
                self.mainArea.coords(self.edit_line_1,
                                     float(self.center_x.get()) - 3,
                                     float(self.center_y.get()) - float(self.radius_y.get()) + 3,
                                     float(self.center_x.get()) + 3,
                                     float(self.center_y.get()) - float(self.radius_y.get()) - 3)
                self.mainArea.coords(self.edit_line_2,
                                     float(self.center_x.get()) + float(self.radius_x.get()) - 3,
                                     float(self.center_y.get()) + 3,
                                     float(self.center_x.get()) + float(self.radius_x.get()) + 3,
                                     float(self.center_y.get()) - 3)
                self.mainArea.itemconfig(self.edit_item_data['id'],
                                         width=self.size_button.get())
                self.edit_item_data['x'] = float(self.center_x.get())
                self.edit_item_data['y'] = float(self.center_y.get())
                self.edit_item_data['r1'] = float(self.radius_x.get())
                self.edit_item_data['r2'] = float(self.radius_y.get())

            if self.ACTUAL_MODE == 'rectangle':
                self.mainArea.coords(self.edit_item_data['id'],
                                     float(self.corner_1_x.get()),
                                     float(self.corner_1_y.get()),
                                     float(self.corner_2_x.get()),
                                     float(self.corner_2_y.get()))
                self.mainArea.coords(self.edit_line_1,
                                     float(self.corner_1_x.get()) - 3,
                                     float(self.corner_1_y.get()) + 3,
                                     float(self.corner_1_x.get()) + 3,
                                     float(self.corner_1_y.get()) - 3)
                self.mainArea.coords(self.edit_line_2,
                                     float(self.corner_2_x.get()) - 3,
                                     float(self.corner_2_y.get()) + 3,
                                     float(self.corner_2_x.get()) + 3,
                                     float(self.corner_2_y.get()) - 3)
                self.mainArea.itemconfig(self.edit_item_data['id'],
                                         width=self.size_button.get())

        else:
            if self.ACTUAL_MODE == 'line':
                line_id = self.mainArea.create_line(self.end_x.get(), self.end_y.get(), self.start_x.get(),
                                                    self.start_y.get(),
                                                    width=self.size_button.get(), fill=self.color,
                                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
                self.mainArea.tag_bind(line_id, '<Double-1>', self.on_line_click)
            if self.ACTUAL_MODE == 'circle':
                circle_id = self.mainArea.create_oval(float(self.center_x.get()) - float(self.radius_x.get()),
                                                      float(self.center_y.get()) - float(self.radius_y.get()),
                                                      float(self.center_x.get()) + float(self.radius_x.get()),
                                                      float(self.center_y.get()) + float(self.radius_y.get()),
                                                      width=self.size_button.get())
                self.mainArea.tag_bind(circle_id, '<Double-1>', self.on_circle_click)
            if self.ACTUAL_MODE == 'rectangle':
                rectangle_id = self.mainArea.create_rectangle(float(self.corner_1_x.get()),
                                                              float(self.corner_1_y.get()),
                                                              float(self.corner_2_x.get()),
                                                              float(self.corner_2_y.get()),
                                                              width=self.size_button.get())
                self.mainArea.tag_bind(rectangle_id, '<Double-1>', self.on_rectangle_click)

    def unbind_events(self):
        self.mainArea.unbind('<B1-Motion>')
        self.mainArea.unbind('<ButtonPress-1>')
        self.mainArea.unbind('<ButtonRelease-1>')
        self.edit_mode = False

    def pointer_click(self):
        self.unbind_events()
        self.activate_button(self.pointer_button, 'pointer')

    def pen_click(self):
        self.unbind_events()
        self.mainArea.bind('<B1-Motion>', self.paint)
        self.mainArea.bind('<ButtonRelease-1>', self.reset)
        self.activate_button(self.pen_button, 'pen')

    def eraser_click(self):
        self.unbind_events()
        self.mainArea.bind('<B1-Motion>', self.paint)
        self.mainArea.bind('<ButtonRelease-1>', self.reset)
        self.activate_button(self.eraser_button, 'eraser', eraser_md=True)

    def draw_click(self):
        self.unbind_events()
        self.activate_button(self.line_button, 'line')
        self.mainArea.bind('<ButtonPress-1>', self.draw_line)
        self.mainArea.bind('<ButtonRelease-1>', self.draw_line)
        self.mainArea.bind('<B1-Motion>', self.draw_line)

    def circle_click(self):
        self.unbind_events()
        self.activate_button(self.circle_button, 'circle')
        self.mainArea.bind('<ButtonPress-1>', self.draw_circle)
        self.mainArea.bind('<ButtonRelease-1>', self.draw_circle)
        self.mainArea.bind('<B1-Motion>', self.draw_circle)

    def rectangle_click(self):
        self.unbind_events()
        self.activate_button(self.rectangle_button, 'rectangle')
        self.mainArea.bind('<ButtonPress-1>', self.draw_rectangle)
        self.mainArea.bind('<ButtonRelease-1>', self.draw_rectangle)
        self.mainArea.bind('<B1-Motion>', self.draw_rectangle)

    def activate_button(self, btn, mode, eraser_md=False):
        self.mainArea.delete(self.edit_line_1)
        self.mainArea.delete(self.edit_line_2)
        self.TopBar_config(mode)
        self.active_button.config(relief=RAISED)
        btn.config(relief=SUNKEN)
        self.active_button = btn
        self.eraser_on = eraser_md

    def draw_rectangle(self, event):
        if str(event.type) == 'ButtonPress':
            self.mainArea.old_coords = event.x, event.y
            self.corner_1_x.delete(0, END)
            self.corner_1_x.insert(0, event.x)
            self.corner_1_y.delete(0, END)
            self.corner_1_y.insert(0, event.y)
        elif str(event.type) == 'ButtonRelease':
            x1, y1 = self.mainArea.old_coords
            rectangle_id = self.mainArea.create_rectangle(x1, y1, event.x, event.y, width=self.size_button.get())
            self.mainArea.tag_bind(rectangle_id, '<Double-1>', self.on_rectangle_click)
            self.mainArea.delete(self.rectangle_id)
        elif str(event.type) == 'Motion':
            try:
                self.mainArea.delete(self.rectangle_id)
            except:
                pass
            x1, y1 = self.mainArea.old_coords
            self.corner_2_x.delete(0, END)
            self.corner_2_x.insert(0, event.x)
            self.corner_2_y.delete(0, END)
            self.corner_2_y.insert(0, event.y)
            self.rectangle_id = self.mainArea.create_rectangle(x1, y1, event.x, event.y,
                                                               width=self.size_button.get())

    def draw_circle(self, event):
        if str(event.type) == 'ButtonPress':
            self.mainArea.old_coords = event.x, event.y
            self.drag_data_x = event.x
            self.drag_data_y = event.y
            self.center_x.delete(0, END)
            self.center_x.insert(0, event.x)
            self.center_y.delete(0, END)
            self.center_y.insert(0, event.y)
        elif str(event.type) == 'ButtonRelease':
            try:
                x, y = self.mainArea.old_coords
                delta_x = event.x - self.drag_data_x
                delta_y = event.y - self.drag_data_y
                oval_id = self.mainArea.create_oval(x - delta_x, y - delta_y, x + delta_x, y + delta_y,
                                                    width=self.size_button.get())
                self.mainArea.tag_bind(oval_id, '<Double-1>', self.on_circle_click)
                self.mainArea.delete(self.oval_id)
            except:
                pass
        elif str(event.type) == 'Motion':
            try:
                self.mainArea.delete(self.oval_id)
            except:
                pass

            x, y = self.mainArea.old_coords
            delta_x = event.x - self.drag_data_x
            delta_y = event.y - self.drag_data_y
            self.oval_id = self.mainArea.create_oval(x - delta_x, y - delta_y, x + delta_x, y + delta_y,
                                                     width=self.size_button.get())

            self.radius_x.delete(0, END)
            self.radius_x.insert(0, abs(delta_x))
            self.radius_y.delete(0, END)
            self.radius_y.insert(0, abs(delta_y))

    def draw_line(self, event):
        if str(event.type) == 'ButtonPress':
            self.start_x.delete(0, END)
            self.start_x.insert(0, event.x)
            self.start_y.delete(0, END)
            self.start_y.insert(0, event.y)
            self.mainArea.old_coords = event.x, event.y

        elif str(event.type) == 'ButtonRelease':
            x, y = event.x, event.y
            x1, y1 = self.mainArea.old_coords
            line_id = self.mainArea.create_line(x, y, x1, y1,
                                                width=self.size_button.get(), fill=self.color,
                                                capstyle=ROUND, smooth=TRUE, splinesteps=36)

            self.mainArea.tag_bind(line_id, '<Double-1>', self.on_line_click)
            self.mainArea.delete(self.line_id)

        elif str(event.type) == 'Motion':
            try:
                self.mainArea.delete(self.line_id)
            except:
                pass
            x1, y1 = self.mainArea.old_coords
            self.end_x.delete(0, END)
            self.end_x.insert(0, event.x)
            self.end_y.delete(0, END)
            self.end_y.insert(0, event.y)
            self.line_id = self.mainArea.create_line(event.x, event.y, x1, y1,
                                                     width=self.size_button.get(), fill=self.color,
                                                     capstyle=ROUND, smooth=TRUE, splinesteps=36)

    def paint(self, event):
        self.line_width = self.size_button.get()
        if self.eraser_on:
            color = 'white'
        else:
            color = self.color

        if self.old_x and self.old_y:
            self.mainArea.create_line(self.old_x, self.old_y, event.x, event.y,
                                      width=self.line_width, fill=color,
                                      capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def placeholder(self):
        x = 0
        siema = "siema"


if __name__ == '__main__':
    Grafika()
