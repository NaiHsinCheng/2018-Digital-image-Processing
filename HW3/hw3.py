import tkinter as tk
from tkinter import*
from PIL import Image,ImageTk,ImageEnhance
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog as fd
import scipy.misc

###########################################  tk窗口  ############################################
window=tk.Tk() #window窗口 Tk是tk內的一個objecct
window.title('Homework3')    #給窗口命名
window.geometry('750x580')   #訂定窗口大小
##########################################  open選項  ###########################################
lbl1 = tk.Label(window)
lbl2 = tk.Label(window)
def show_img():
    global filename
    global img
    global rload
    global load
    filename = fd.askopenfilename()
    load = Image.open(filename)
    if load==Image.open('Fig0460a.tif'):
        rload=load.resize((230,333),Image.BILINEAR)
    else:
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


####################################  RGB三色分量圖  ############################################



def CIShow():
    global filename
    img=cv.imread(filename)
    (B,G,R)=cv.split(img)
    zeros=np.zeros(img.shape[:2],dtype="uint8")
    RR=cv.merge([R,zeros,zeros])
    GG=cv.merge([zeros,G,zeros])
    BB=cv.merge([zeros,zeros,B])
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(2,2,1);plt.title("Original")
    plt.imshow(img)
    plt.subplot(2,2,2);plt.title("Blue")
    plt.imshow(BB)
    plt.subplot(2,2,3);plt.title("Green")
    plt.imshow(GG)
    plt.subplot(2,2,4);plt.title("Red")
    plt.imshow(RR)
    plt.show()

r1=Radiobutton(window,text='RGB component image',command=CIShow)     #設置選擇按鈕顯示圖片1
r1.place(x=100,y=380)


################################  把RGB轉成HSI 顯示H，S，I的灰階圖  ################################

def rgbTOhsi():    #rgb轉成hsi
    global filename
    img=cv.imread(filename)
    rows = int(img.shape[0]) 
    cols = int(img.shape[1]) 
    b, g, r = cv.split(img) 
    b = b / 255.0 
    g = g / 255.0 
    r = r / 255.0 
    H, S, I = cv.split(img) 
    for i in range(rows): 
        for j in range(cols): 
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j])) 
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j])) 
            theta = float(np.arccos(num/den)) 
            if den == 0: 
                H = 0 
            elif b[i, j] <= g[i, j]: 
                H = theta 
            else: 
                H = 2*3.14169265 - theta 
            min_RGB = min(min(b[i, j], g[i, j]), r[i, j]) 
            sum = b[i, j]+g[i, j]+r[i, j] 
            if sum == 0: 
                S = 0 
            else: 
                S = 1 - 3*min_RGB/sum 
            H = H/(2*3.14159265) 
            I = sum/3.0 
            img[i, j, 0] = H*255 
            img[i, j, 1] = S*255 
            img[i, j, 2] = I*255    #此時的img是hsi影像
    H_img=img[:,:,0]    #h s i 分別顯示
    S_img=img[:,:,1]
    I_img=img[:,:,2]
    H_img=Image.fromarray(H_img)
    S_img=Image.fromarray(S_img)
    I_img=Image.fromarray(I_img)
    if HSI==0:
       render = ImageTk.PhotoImage(H_img.resize((300,300),Image.BILINEAR))
    elif HSI==1:
       render = ImageTk.PhotoImage(S_img.resize((300,300),Image.BILINEAR))
    elif HSI==2:
       render = ImageTk.PhotoImage(I_img.resize((300,300),Image.BILINEAR))
    lbl2.config(image=render)
    lbl2.Image = render

HSI=0
v=tk.IntVar()
def choseHSI():    #選擇要顯示h s i哪個
    global HSI   
    if v.get()==0:
        HSI=0
        rgbTOhsi()
    elif v.get()==1:
        HSI=1
        rgbTOhsi()
    elif v.get()==2:
        HSI=2
        rgbTOhsi()

rb1=Radiobutton(window,text='Hue',variable=v,value=0,command=choseHSI)     #設置選擇按鈕顯示圖片1
rb1.place(x=100,y=410)

rb2=Radiobutton(window,text='Saturation',variable=v,value=1,command=choseHSI)     #設置選擇按鈕顯示圖片2
rb2.place(x=290,y=410)

rb3=Radiobutton(window,text='Intensity',variable=v,value=2,command=choseHSI)     #設置選擇按鈕顯示圖片3
rb3.place(x=500,y=410)
        

###################################  RGB互補色  ###################################

def complement():
    global filename
    img=cv.imread(filename)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
             img[i,j,0]=255-img[i,j,0]
             img[i,j,1]=255-img[i,j,1]
             img[i,j,2]=255-img[i,j,2]
    cc=Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    render = ImageTk.PhotoImage(cc.resize((300,300),Image.BILINEAR))
    lbl2.config(image=render)
    lbl2.Image = render

rb4=Radiobutton(window,text='colorcomplements',command=complement)     #設置選擇按鈕顯示圖片1
rb4.place(x=100,y=440)

############################################  5x5平均過濾內核 / 拉普拉斯銳化 #############################


def blur():
    global filename
    img=cv.imread(filename)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    kernel=np.ones((5,5),np.float32)/25
    dst=cv.filter2D(img,-1,kernel)
    sub=cv.subtract(img,dst)
    plt.figure()
    plt.subplot(1,3,1);plt.title("original")
    plt.imshow(img)
    plt.subplot(1,3,2);plt.title("result")
    plt.imshow(dst)
    plt.subplot(1,3,3);plt.title("difference")
    plt.imshow(sub)
    plt.show()

def sharp():
    global filename
    img = cv.imread(filename)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    B,G,R=cv.split(img)   #rgb分開做
    #拉普拉斯算子
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32)
    B = cv.filter2D(B,-1,kernel)
    G = cv.filter2D(G,-1,kernel)
    R = cv.filter2D(R,-1,kernel)
    dst=cv.merge([B,G,R])
    sub=cv.subtract(img,dst)
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(132)
    plt.imshow(sub)
    plt.title('result')
    plt.subplot(133)
    plt.imshow(dst)
    plt.title('difference')
    plt.show()

rb5=Radiobutton(window,text='5x5 average kernel',command=blur)     #設置選擇按鈕顯示圖片1
rb5.place(x=100,y=470)
rb6=Radiobutton(window,text='Laplacian sharping',command=sharp)     #設置選擇按鈕顯示圖片2
rb6.place(x=300,y=470)

############################################  分割羽毛  #############################
def HSI2RGB(hsi_img):     #hsi 轉 rgb
    row = np.shape(hsi_img)[0] 
    col = np.shape(hsi_img)[1]
    rgb_img = hsi_img.copy()
    H,S,I = cv.split(hsi_img)
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I 
    for i in range(row): 
        h = H[i]*2*np.pi 
        a1 = h >=0 
        a2 = h < 2*np.pi/3 
        a = a1 & a2
        tmp = np.cos(np.pi / 3 - h) 
        b = I[i] * (1 - S[i]) 
        r = I[i]*(1+S[i]*np.cos(h)/tmp) 
        g = 3*I[i]-r-b 
        B[i][a] = b[a] 
        R[i][a] = r[a] 
        G[i][a] = g[a]
        a1 = h >= 2*np.pi/3 
        a2 = h < 4*np.pi/3 
        a = a1 & a2
        tmp = np.cos(np.pi - h) 
        r = I[i] * (1 - S[i]) 
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp) 
        b = 3 * I[i] - r - g 
        R[i][a] = r[a] 
        G[i][a] = g[a] 
        B[i][a] = b[a]
        a1 = h >= 4 * np.pi / 3 
        a2 = h < 2 * np.pi 
        a = a1 & a2
        tmp = np.cos(5 * np.pi / 3 - h) 
        g = I[i] * (1-S[i]) 
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp) 
        r = 3 * I[i] - g - b 
        B[i][a] = b[a] 
        G[i][a] = g[a] 
        R[i][a] = r[a] 
    rgb_img[:,:,0] = B*255 
    rgb_img[:,:,1] = G*255 
    rgb_img[:,:,2] = R*255 
    return rgb_img

def feather():   #h的灰階圖跟s的灰階圖做運算 再把它轉成rgb 留下羽毛
    global filename
    img=cv.imread(filename)
    rows = int(img.shape[0]) 
    cols = int(img.shape[1]) 
    b, g, r = cv.split(img) 
    b = b / 255.0 
    g = g / 255.0 
    r = r / 255.0 
    H, S, I = cv.split(img) 
    for i in range(rows): 
        for j in range(cols): 
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j])) 
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j])) 
            theta = float(np.arccos(num/den)) 
            if den == 0: 
                H = 0 
            elif b[i, j] <= g[i, j]: 
                H = theta 
            else: 
                H = 2*3.14169265 - theta 
            min_RGB = min(min(b[i, j], g[i, j]), r[i, j]) 
            sum = b[i, j]+g[i, j]+r[i, j] 
            if sum == 0: 
                S = 0 
            else: 
                S = 1 - 3*min_RGB/sum 
            H = H/(2*3.14159265) 
            I = sum/3.0 
            img[i, j, 0] = H*255 
            img[i, j, 1] = S*255 
            img[i, j, 2] = I*255 
    H_img=img[:,:,0] 
    S_img=img[:,:,1]
    I_img=img[:,:,2]
    h,w = H_img.shape[:2]
    for i in range(h):
        for j in range(w):
            if np.all((H_img[i, j] < 200 or H_img[i, j] > 225 )and ( S_img[i, j] <150 or S_img[i, j] > 155)):
                    I_img[i, j] = 0
    img[:,:,0]=H_img
    img[:,:,1]=S_img
    img[:,:,2]=I_img
    rgbimg=HSI2RGB(img)
    render = ImageTk.PhotoImage(Image.fromarray(rgbimg).resize((300,300),Image.BILINEAR))    #把圖片放上label
    lbl2.config(image=render)
    lbl2.Image = render

rb1=Radiobutton(window,text='feather segment',command=feather)     #設置選擇按鈕顯示圖片1
rb1.place(x=100,y=500)

############################################  高斯高通濾波  #############################

def homo(rh, rl, c ,D0):          #homomorphic filter函數
    global filename
    img = cv.imread(filename)
    img = np.float32(img)
    img = img/255
    rows,cols,dim=img.shape
    imgYCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y,cr,cb = cv.split(imgYCrCb)
    y_log = np.log(y+0.01)
    y_fft = np.fft.fft2(y_log)
    y_fft_shift = np.fft.fftshift(y_fft)
    G = np.ones((rows,cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j]=((rh-rl)*(1-np.exp(-c*(((i-rows/2)**2+(j-cols/2)**2)**0.5)/(D0**2))))+rl  #公式
    result_filter = G * y_fft_shift
    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    result = np.exp(result_interm)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(122)
    plt.imshow(result,'gray')
    plt.title('result')
    plt.show()

def OK():            #完成參數輸入 將值放入homomorphic filter函數
    h = float(rh.get())
    l = float(rl.get())
    cc = float(c.get())
    D = float(D0.get())
    homo(h, l, cc ,D)

def homodone():      #預設的homomorphic filter
    rh, rl, c ,D0 = 3.0,0.4,5,20
    homo(rh, rl, c ,D0)

rb1=Radiobutton(window,text='homomorphic filter (rh=3.0, rl=0.4, c=5 ,D0=20)',command=homodone)   
rb1.place(x=100,y=530)

label_hf=tk.Label(window,text='homomorphic filter',width=20,height=2).place(x=20,y=565)    #文字
rh=Entry(window)         #可輸入的兩個值（範圍） place不能直接點在後面 會出錯
rh.place(x=220,y=560)
rl=Entry(window)
rl.place(x=440,y=560)
c=Entry(window)
c.place(x=220,y=590)
D0=Entry(window)
D0.place(x=440,y=590)
label_rh=tk.Label(window,text='rh',width=2,height=2).place(x=200,y=550)            #文字rh
label_rl=tk.Label(window,text='rl',width=2,height=2).place(x=420,y=550)            #文字rl
label_c=tk.Label(window,text='c',width=2,height=2).place(x=200,y=580)            #文字c
label_d0=tk.Label(window,text='D0',width=2,height=2).place(x=420,y=580)            #文字D0
button_ok = tk.Button(window,text='ok',width=4,height=1,command=OK)     #確認輸入完畢的按鈕
button_ok.place(x=610,y=575)

########################################################################################################################
window.mainloop()   #loop不斷的刷新

