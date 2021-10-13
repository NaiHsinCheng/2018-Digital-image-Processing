import tkinter as tk
from tkinter import*
from PIL import Image,ImageTk
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#############################################################  tk窗口  ############################################
window=tk.Tk() #window窗口 Tk是tk內的一個objecct
window.title('Homework1')    #給窗口命名
window.geometry('750x580')   #訂定窗口大小
#############################################################  open選項  ############################################

label_open=tk.Label(window,text='open',width=7,height=2).place(x=16,y=15)    #設置open標籤

img33=Image.open('LennaGray256.tif')
img3=ImageTk.PhotoImage(img33.resize((300,300),Image.BILINEAR))        #讀縮小後的圖片1
img3=ImageTk.PhotoImage(img33)     #讀圖片3

img11=Image.open('woman.tif')
img1=ImageTk.PhotoImage(img11.resize((300,300),Image.BILINEAR))        #讀縮小後的圖片1

img22=Image.open('LennaGray256.jpg')
img2=ImageTk.PhotoImage(img22.resize((300,300),Image.BILINEAR))        #讀圖片2

result=img33
j=ImageTk.PhotoImage(result.resize((300,300),Image.BILINEAR))

var=tk.IntVar()

imgLab=tk.Label(window,width=300,height=300).place(x=500,y=15)

def open_picture():    #顯示1
    global result,imgLab,j
    if var.get()==1:
        imglabel=Label(window,image=img1).place(x=150,y=15)
        result=img11
        j=ImageTk.PhotoImage(result.resize((300,300),Image.BILINEAR))
        imgLab=tk.Label(window,image=j,width=300,height=300)
        imgLab.place(x=500,y=15)
    elif var.get()==2:
        imglabel=Label(window,image=img2).place(x=150,y=15)
        result=img22
        j=ImageTk.PhotoImage(result.resize((300,300),Image.BILINEAR))
        imgLab=tk.Label(window,image=j,width=300,height=300)
        imgLab.place(x=500,y=15)
    else :
        imglabel=Label(window,image=img3).place(x=150,y=15)
        result=img33
        j=ImageTk.PhotoImage(result.resize((300,300),Image.BILINEAR))
        imgLab=tk.Label(window,image=j,width=300,height=300)
        imgLab.place(x=500,y=15)

r1=Radiobutton(window,text='woman.tif',variable=var,value=1,command=open_picture)     #設置選擇按鈕顯示圖片1
r1.place(x=1,y=50)

r2=Radiobutton(window,text='LennaGray256.jpg',variable=var,value=2,command=open_picture)     #設置選擇按鈕顯示圖片2
r2.place(x=1,y=80)

r3=Radiobutton(window,text='LennaGray256.tif',variable=var,value=3,command=open_picture)     #設置選擇按鈕顯示圖片3
r3.place(x=1,y=110)

###############################################################  儲存影像  ############################################

def img_save():
        global result
        result.save("result.tif")
button_save=tk.Button(window,text='save',width=7,height=2,command=img_save)    #設置save按鈕
button_save.place(x=10,y=150)

###############################################################  調整大小(?)  ############################################

label_zoom=tk.Label(window,text='zoom',width=10,height=2).place(x=20,y=475)    #設置zoom標籤


def chan(v):
    global result,j
    size=int(v)
    result=np.array(result)
    im=cv.resize(result,(size,size),interpolation=cv.INTER_LINEAR)
    j=ImageTk.PhotoImage(Image.fromarray(im))
    imgLab.config(image=j)
    
        

s_zoom=tk.Scale(window,from_=1,to=1000,orient=tk.HORIZONTAL,length=500,showvalue=1,tickinterval=0,resolution=1,command=chan)
s_zoom.place(x=180,y=466)   #zoom拉軸 從5到23 orient是調方>向 長度的單位是pixel 顯示出它取道的值 resolution顯示到兩位小數 tkinterval:標籤>的變>量長度




###############################################################  調整亮度對比  ############################################

I=cv.imread('woman.tif',0)
O=I
number=tk.IntVar()
def cb(self):    #對比亮度
    global I,O,result,j
    if var.get()==1:
        I=cv.imread('woman.tif',0)
    elif var.get()==2:
        I=cv.imread('LennaGray256.jpg',0)
    else :
        I=cv.imread('LennaGray256.tif',0)
    if number.get()==1:
        O=I*float(scb_a.get())+float(scb_b.get())	#線性
    elif number.get()==2:
        O=np.exp(I*float(scb_a.get())+float(scb_b.get()))	#指數
    else :
        O=np.log(I*float(scb_a.get())+float(scb_b.get()))	#對數
    O[0>255]=255
    result=Image.fromarray((np.round(O)).astype(np.uint8))			#顯示圖片
    j=ImageTk.PhotoImage(image=result.resize((300,300),Image.BILINEAR))
    imgLab.config(image=j)

def chs():	#更改對數b的範圍
        scb_b.config(from_=1.01)
        scb_a.config(from_=0.01)
        scb_a.config(to=1000000000000000000000000000000000000000)
        scb_a.config(tickinterval=10000000000000)
def chss():	#更改指數範圍
        scb_b.config(from_=0.01)
        scb_a.config(from_=0.001)
        scb_a.config(to=1)
        scb_a.config(tickinterval=0.2)
def chsss():	#更改線性範圍
        scb_b.config(from_=0.01)
        
me1=Radiobutton(window,text='Linearly',variable=number,value=1,command=chsss)     #設置選擇按鈕挑方法1
me1.place(x=10,y=350)
me2=Radiobutton(window,text='Exponentially',variable=number,value=2,command=chss)     #設置選擇按鈕挑方法2
me2.place(x=10,y=380)
me3=Radiobutton(window,text='Logarithmically',variable=number,value=3,command=chs)     #設置選擇按鈕挑方法3
me3.place(x=10,y=410)


label_a=tk.Label(window,text='a',width=5,height=2).place(x=130,y=360)	#控制a的拉軸
scb_a=tk.Scale(window,from_=0.01,to=2,orient=tk.HORIZONTAL,length=230,showvalue=1,tickinterval=0.5,resolution=0.01,command=cb)
scb_a.place(x=180,y=346)

label_b=tk.Label(window,text='b',width=5,height=2).place(x=430,y=360)	#控制b的拉軸
scb_b=tk.Scale(window,from_=0.1,to=10,orient=tk.HORIZONTAL,length=500,showvalue=1,tickinterval=1,resolution=0.1,command=cb)
scb_b.place(x=470,y=346)



###############################################################  histogram  ############################################

def img_histogram():
        if var.get()==1:
                imghistogram=cv.imread("woman.tif",0)		#直接讀為灰圖
        elif var.get()==2:
                imghistogram=cv.imread("LennaGray256.jpg",0)
        else :
                imghistogram=cv.imread("LennaGray256.tif",0)
        hist_cv=cv.calcHist([imghistogram],[0],None,[256],[0,256])	#(圖像，通道-灰度圖，掩膜-無，灰度，像素範圍)
        plt.figure()
        plt.plot(hist_cv)	#展示出來
        plt.figure()
        after=cv.equalizeHist(imghistogram)
        hist_cv_after=cv.calcHist([after],[0],None,[256],[0,256])	#均等畫後的
        plt.plot(hist_cv_after)	
        plt.show()			#顯示

button_histogram=tk.Button(window,text='histogram',width=10,height=2,command=img_histogram).place(x=10,y=550)    #設置histogram按鈕

###############################################################  直方圖等化  ############################################


def hist_eq():
        global result,j
        if var.get()==1:
                image=cv.imread("woman.tif",0)  #直接讀為灰圖
        elif var.get()==2:
                image=cv.imread("LennaGray256.jpg",0)
        else :
                image=cv.imread("LennaGray256.tif",0)
 
        lut = np.zeros(256, dtype = image.dtype )#創建空的查表

        hist,bins = np.histogram(image.flatten(),256,[0,256]) 
        cdf = hist.cumsum() 		#計算累計直方圖
        cdf_m = np.ma.masked_equal(cdf,0) 	#去除直方圖中的0
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#lut[i] = int(255.0 *p[i])
        cdf = np.ma.filled(cdf_m,0).astype('uint8') 	#將掩膜處理調的元素補為0
        #計算
        eq = cdf[image]
        cv.imshow("After",eq)
        cv.waitKey(0)
	
button_histeq=tk.Button(window,text='histogram equalization',width=25,height=2,command=hist_eq).place(x=200,y=550)    #設置histogram按鈕

########################################################################################################################
window.mainloop()   #loop不斷的刷新
