import tkinter as tk
from tkinter import*
from PIL import Image,ImageTk,ImageEnhance
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog as fd
import scipy.misc

#############################################################  tk窗口  ############################################
window=tk.Tk() #window窗口 Tk是tk內的一個objecct
window.title('Homework2')    #給窗口命名
window.geometry('750x580')   #訂定窗口大小
#############################################################  open選項  ############################################
lbl1 = tk.Label(window)
lbl2 = tk.Label(window)
def show_img():
    global filename
    global img
    global rload
    global load
    filename = fd.askopenfilename()
    load = Image.open(filename)
    rload=load.resize((300,300),Image.BILINEAR)
    render = ImageTk.PhotoImage(rload)
    img=np.array(Image.open(filename).convert('L'))
    lbl1.config(image=render)           #show在預設的label上
    lbl1.Image = render
    lbl2.config(image=render)
    lbl2.Image = render

button_open = tk.Button(window,text='open',width=7,height=2,command=show_img)     #open按鈕
button_open.place(x=10,y=20)
lbl1.place(x=100,y=20)	        #兩張圖的位置
lbl2.place(x=430,y=20)
###############################################################  灰度分切  ############################################

def Gray_level():
    global load
    #輸入 x1, x2(x1 < x2)
    x1 = Ef.get()
    x2 = Et.get()
    x1=int(x1)
    x2=int(x2)
    img = np.array(load.convert('L'))
    #檢查 x1 < x2
    if x2 < x1:
        x1, x2 = x2, x1
    h,w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            #在範圍外
            if np.all(img[i, j] < x1):
                #選擇黑色還是原樣
                if color.get() == 1:
                    img[i, j] = 0
            #在範圍內
            elif np.all(img[i, j] <= x2 ):
                #convert the pixel in the range to 200 or 100 
                if color.get() == 1:
                    img[i, j] = 200
                else:
                    img[i, j] = 100
            #outside the range
            else:
                #選擇黑色還是原樣
                if color.get() == 1:
                    img[i, j] = 0
            img[i,j] = img[i, j]
    render = ImageTk.PhotoImage(Image.fromarray(img, mode = 'L').resize((300,300),Image.BILINEAR))    #把圖片放上label
    lbl2.config(image=render)
    lbl2.Image = render


label_gfrom=tk.Label(window,text='Gray-Level Slicing    from',width=30,height=2).place(x=10,y=350)    #文字
Ef=Entry(window)         #可輸入的兩個值（範圍） place不能直接點在後面 會出錯
Ef.place(x=220,y=360)
Et=Entry(window)
Et.place(x=440,y=360)
label_gto=tk.Label(window,text='to',width=2,height=2).place(x=400,y=350)            #文字
button_ok = tk.Button(window,text='ok',width=4,height=1,command=Gray_level)     #確認輸入完畢的按鈕
button_ok.place(x=610,y=360)

#選擇要黑色還是保持原樣
color = tk.IntVar()
glb = tk.Radiobutton(window, text = "Black", variable = color, value = 1)
glb.place(x = 250, y = 330)
glw = tk.Radiobutton(window, text = "origin", variable = color, value = 0)
glw.place(x = 330, y = 330)
gl_lbl = tk.Label(window, compound = tk.CENTER, text = 'Gray-level (0~255)')
gl_lbl.place(x = 450, y = 330)



###############################################################  平滑  ############################################

def smooth(v):
    global filename
    src=cv.imread(filename)
    sigma=int(v)        #取scale上的值
    dst=cv.GaussianBlur(src,(9,9),sigma)           #高斯模糊
    render = ImageTk.PhotoImage(Image.fromarray(dst).resize((300,300),Image.BILINEAR))    #把圖片放上label
    lbl2.config(image=render)
    lbl2.Image = render

label_blur=tk.Label(window,text='Smooth',width=7,height=2).place(x=380,y=400)
blur_scale = tk.Scale(window, from_=1, to=10, orient=tk.HORIZONTAL,length=200, showvalue=0, tickinterval=0, resolution=1, command=smooth)
blur_scale.place(x=470,y=410)

###############################################################  銳化  ############################################
def sharpen(v):
    global load
    sharpness=float(v)
    dst=ImageEnhance.Sharpness(load).enhance(sharpness)
    render = ImageTk.PhotoImage((dst).resize((300,300),Image.BILINEAR))    #把圖片放上label
    lbl2.config(image=render)
    lbl2.Image = render


label_sharpen=tk.Label(window,text='Sharpen',width=7,height=2).place(x=50,y=400)
sharpen_scale = tk.Scale(window, from_=1, to=15, orient=tk.HORIZONTAL,length=200, showvalue=0, tickinterval=0, resolution=0.1, command=sharpen)
sharpen_scale.place(x=140,y=410)
###############################################################  傅立葉轉換  ############################################

def fourier():
    global img
    f=np.fft.fft2(img)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    render = ImageTk.PhotoImage(Image.fromarray(magnitude_spectrum).resize((300,300),Image.BILINEAR))    #把圖片放上label
    lbl2.config(image=render)
    lbl2.Image = render

button_blur = tk.Button(window,text='Fourier Transformed',width=14,height=2,command=fourier)
button_blur.place(x=10,y=450)

###############################################################  Amplitude and Phase  ############################################

def Amp():
    global img
    #讀取
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) 
    # 逆變換-取絕對值就是振幅
    f1shift = np.fft.ifftshift(np.abs(fshift)) 
    img_back = np.fft.ifft2(f1shift)
    #取出來是複數無法顯示
    img_back = np.abs(img_back)
    #調整大小和顯示
    render = ImageTk.PhotoImage(Image.fromarray(img_back).resize((300,300),Image.BILINEAR))
    lbl2.config(image=render)
    lbl2.Image = render

def Pha():
    global img
    #讀取
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) 
    # 逆變換-取相位 
    f2shift = np.fft.ifftshift(np.angle(fshift)) 
    img_back = np.fft.ifft2(f2shift) 
    #取出來的是複數 無法顯示 
    img_back = np.abs(img_back)
    #調整大小和顯示
    render = ImageTk.PhotoImage(Image.fromarray(img_back).resize((300,300),Image.BILINEAR))
    lbl2.config(image=render)
    lbl2.Image = render


button_Amp = tk.Button(window,text='Amplitude Image(i_2DFFT)',width=25,height=2,command=Amp)
button_Amp.place(x=130,y=450)
button_Pha = tk.Button(window,text='Phase Image(i_2DFFT)',width=25,height=2,command=Pha)
button_Pha.place(x=340,y=450)

def A_P():
    global filename
    img=cv.imread(filename,0)
    f=np.fft.fft2(img)
    fshift=np.fft.fftshift(f)
    img=20*np.log(np.abs(fshift))
    plt.subplot(121)
    plt.imshow(img,'gray')
    plt.title('Amplitude')

    img2=np.angle(fshift)
    plt.subplot(122)
    plt.imshow(img2,'gray')
    plt.title('Phase')
    plt.show()

button_A_P = tk.Button(window,text='Amplitude and Phase Image',width=25,height=2,command=A_P)
button_A_P.place(x=550,y=450)
###############################################################  位元平面  ############################################


def bitPlan(r):
    s=np.empty(r.shape, dtype=np.uint8)
    for j in range(r.shape[0]):
        for i in range(r.shape[1]):
            bits=bin(r[j][i])[2:].rjust(8,'0')
            fill=int(bits[-flat - 1])
            s[j][i]=255 if fill else 0
    return s

def bitPlanShow():
    global filename
    img=Image.open(filename)
    img=img.convert('L')
    img_mat=scipy.misc.fromimage(img)
    img_conveted_mat=bitPlan(img_mat)
    im_conveted=Image.fromarray(img_conveted_mat)
    render = ImageTk.PhotoImage(im_conveted.resize((300,300),Image.BILINEAR))
    lbl2.config(image=render)
    lbl2.Image = render
flat=0
var=tk.IntVar()
def choseBitPlan():
    global flat    
    if var.get()==0:
        flat=0
        bitPlanShow()
    elif var.get()==1:
        flat=1
        bitPlanShow()
    elif var.get()==2:
        flat=2
        bitPlanShow()
    elif var.get()==3:
        flat=3
        bitPlanShow()
    elif var.get()==4:
        flat=4
        bitPlanShow()
    elif var.get()==5:
        flat=5
        bitPlanShow()
    elif var.get()==6:
        flat=6
        bitPlanShow()
    else :
        flat=7
        bitPlanShow()

r1=Radiobutton(window,text='1 st bit plan',variable=var,value=0,command=choseBitPlan)     #設置選擇按鈕顯示圖片1
r1.place(x=190,y=510)

r2=Radiobutton(window,text='2 nd bit plan',variable=var,value=1,command=choseBitPlan)     #設置選擇按鈕顯示圖片2
r2.place(x=190,y=540)

r3=Radiobutton(window,text='3 rd bit plan',variable=var,value=2,command=choseBitPlan)     #設置選擇按鈕顯示圖片3
r3.place(x=340,y=510)

r4=Radiobutton(window,text='4 th bit plan',variable=var,value=3,command=choseBitPlan)     #設置選擇按鈕顯示圖片4
r4.place(x=340,y=540)

r5=Radiobutton(window,text='5 th bit plan',variable=var,value=4,command=choseBitPlan)     #設置選擇按鈕顯示圖片5
r5.place(x=490,y=510)

r6=Radiobutton(window,text='6 th bit plan',variable=var,value=5,command=choseBitPlan)     #設置選擇按鈕顯示圖片6
r6.place(x=490,y=540)

r7=Radiobutton(window,text='7 th bit plan',variable=var,value=6,command=choseBitPlan)     #設置選擇按鈕顯示圖片7
r7.place(x=640,y=510)

r8=Radiobutton(window,text='8 th bit plan',variable=var,value=7,command=choseBitPlan)     #設置選擇按鈕顯示圖片8
r8.place(x=640,y=540)

label_bitPlan=tk.Label(window,text='Bit Plan Image',width=20,height=2).place(x=10,y=520)
########################################################################################################################
window.mainloop()   #loop不斷的刷新

