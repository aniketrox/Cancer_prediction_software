#====================================Importing Required Libraries=======================================#

from tkinter import *
import os
from PIL import Image,ImageTk

from tkinter import filedialog
from tkinter.filedialog import askopenfile

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#=======================================================================================================#

#============================Device Setting======================================#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#================================================================================#

def btn_clicked():
    print("Button Clicked")

#===================Main Frame======================#
root = Tk()
root.title('CPS-Optimizer')
root.geometry('1200x700')
root.resizable(0, 0)
#===================================================#

frame1 = Frame(root,bg="#6a6a6b",width=1200,height=450)
frame1.pack()

canvas1 = Canvas(frame1, width= 430, height= 70, bg="#454545",bd=0,highlightthickness=0, relief='ridge')
canvas1.create_text(215, 35, text="Cancer Prediction Software", fill="white", font=('Courier 20 bold'))
canvas1.place(relx=0.5, rely=0.12, anchor=CENTER)

frame21 = Frame(frame1,bg="#a7a7a8",width=1150,height=150)
frame21.place(relx=0.5, rely=0.4, anchor=CENTER)

#==================================processing and ending=====================================#
img = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/process.png")
img_resized=img.resize((130,130))
photo = ImageTk.PhotoImage(img_resized)
lukemia_logo = Label(frame21,image = photo,bg="#a7a7a8")
lukemia_logo.image = photo
lukemia_logo.place(relx=0.05, rely=0.1)


brain_logo = Label(frame21,image = photo,bg="#a7a7a8")
brain_logo.image = photo
brain_logo.place(relx=0.32, rely=0.1)

breast_logo = Label(frame21,image = photo,bg="#a7a7a8")
breast_logo.image = photo
breast_logo.place(relx=0.58, rely=0.1)

kidney_logo = Label(frame21,image = photo,bg="#a7a7a8")
kidney_logo.image = photo
kidney_logo.place(relx=0.83, rely=0.1)
#============================================================================================#


img3 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/7.png")
photo3 = ImageTk.PhotoImage(img3)
e3_button=Button(frame1, bg="#6a6a6b",cursor="hand2",fg="black",borderwidth=0.1, text="",image=photo3,compound=LEFT,font="Courier 19 bold",relief=SUNKEN)
e3_button.place(relx=0.58, rely=0.74)

img4 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/8.png")
photo4 = ImageTk.PhotoImage(img4)
e4_button=Button(frame1, bg="#6a6a6b",cursor="hand2",fg="black",borderwidth=0.1, text="",image=photo4,compound=LEFT,relief=SUNKEN)
e4_button.place(relx=0.83, rely=0.74)


#=================================Blood Cancer==================================#
l_logo = 0
l_path = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/lukemia'
l_valid_data_dir = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/lukemia/training_data/fold_1'
l_train_data_dir = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/lukemia/training_data/fold_0'
l_classes = ('all','hem')

l_transformer = torchvision.transforms.Compose([torchvision.transforms. Resize((224,224)),torchvision.transforms.RandomHorizontalFlip(p=0.5), torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.RandomRotation(10), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.7, 0.2]),])
l_trainset = ImageFolder(l_train_data_dir, transform=l_transformer)
l_testset = ImageFolder(l_valid_data_dir, transform=l_transformer)
l_batch_size=16
l_trainloader = torch.utils.data.DataLoader(l_trainset, batch_size=l_batch_size, shuffle=True, num_workers=2)
l_testloader = torch.utils.data.DataLoader(l_testset, batch_size=l_batch_size,shuffle=False,num_workers=2)

class lNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=53,kernel_size=4,stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2= nn.Conv2d(53, 106, kernel_size=4, stride=2, padding=3)
        self.conv3 = nn.Conv2d(106, 212, kernel_size=4, stride=2, padding=3)
        self.fc1 = nn.Linear(3392, 1908)
        self.fc2= nn. Linear(1908, 2)

    def forward(self, x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x= self.pool(F.relu(self.conv3(x)))
        x= torch.flatten(x, 1)
        x= F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


lnet = lNet()
lnet = lnet.to(device)

def train_lukemia_engine():
    global l_path,l_valid_data_dir,l_train_data_dir,lNet,l_classes
    
    


    criterion = nn.CrossEntropyLoss()
    lr_dynamic = 0.01
    momentum_dynamic = 0.9
    optimizer = optim.SGD(lnet.parameters(), lr=lr_dynamic, momentum=momentum_dynamic)

    Epoch_list =[]
    Loss_list =[]


    for epoch in range(50):
        if epoch >= 10:
            if sum(Loss_list[-5:]) > sum(Loss_list[-10:-5]):
                break
            running_loss=0.0
            for i, data in enumerate(l_trainloader, 0):
                inputs, labels =data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = lnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    if i==99:
                        Epoch_list.append((epoch) + 0.25)
                        Loss_list.append(running_loss)

                    if i==199:
                        Epoch_list.append((epoch) + 0.5)
                        Loss_list.append(running_loss)

                    if i==299:
                        Epoch_list.append((epoch) + 0.75)
                        Loss_list.append(running_loss)

                    if i==399:
                        Epoch_list.append((epoch) + 1)
                        Loss_list.append(running_loss)

                    running_loss = 0.0
                    lr_dynamic = 0.96*lr_dynamic
                    optimizer = optim.SGD(lnet.parameters(),lr=lr_dynamic,momentum=momentum_dynamic)

    print("finished")
    l_logo = 1
    if l_logo ==1:
        img = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/completed.png")
        img_resized=img.resize((110,110))
        photo = ImageTk.PhotoImage(img_resized)
        lukemia_logo = Label(frame21,image = photo,bg="#a7a7a8")
        lukemia_logo.image = photo
        lukemia_logo.place(relx=0.05, rely=0.1)


    l_path_net = l_path + r'/model.pth'
    torch.save(lnet.state_dict(),l_path_net)


#===============================================================================#
img1 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/1.png")
photo1 = ImageTk.PhotoImage(img1)
e1_button=Button(frame1, bg="#6a6a6b",cursor="hand2",fg="black",borderwidth=0.1,image=photo1,text="",compound=LEFT,relief=SUNKEN,command=train_lukemia_engine)
e1_button.place(relx=0.05, rely=0.74)

#====================================Lukemia Window============================================#
def Engine_1():
    new_window = Toplevel(root)
    new_window.title('Engine One')
    new_window.geometry('1000x620')
    new_window.resizable(0, 0)

    frame1 = Frame(new_window,bg="#6a6a6b",width=1000,height=70)
    frame1.pack()

    canvas1 = Canvas(frame1, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    canvas1.create_text(215, 35, text="Lukemia Prediction", fill="white", font=('Courier 20 bold'))
    canvas1.place(relx=0.5, rely=0.39, anchor=CENTER)

    frame21 = Frame(new_window,bg="#6a6a6b",width=250,height=250)
    frame21.place(relx=0.032, rely=0.15)

    
    frame22 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame22.place(relx=0.32, rely=0.15)

    frame221 = Frame(frame22,bg="white",width=100,height=100)
    frame221.place(relx=0.1, rely=0.04)


    def upload_photo():
        f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png'),("all files","*.*")]
        filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File",filetypes=f_types)
        img = Image.open(filename)
        img_resized=img.resize((100,100))
        photo = ImageTk.PhotoImage(img_resized)
        label = Label(frame221,image = photo)
        label.image = photo
        label.place(relx=0, rely=0)


    upload_button=Button(frame22, bg="#a7a7a8",cursor="hand2",fg="black",borderwidth=0.1, text="upload",command=upload_photo,compound=LEFT,font="Courier 13",relief=SUNKEN)
    upload_button.place(relx=0.6, rely=0.1)

    frame222= Label(frame22,text="Name", font=("candara",13),bg="light gray")
    frame222.place(relx=0.5, rely=0.35, anchor=CENTER,width=250,height=35)

    frame223= Label(frame22,text="Age", font=("candara",13),bg="light gray")
    frame223.place(relx=0.5, rely=0.45, anchor=CENTER,width=250,height=35)

    frame224= Label(frame22,text="Sex", font=("candara",13),bg="light gray")
    frame224.place(relx=0.5, rely=0.55, anchor=CENTER,width=250,height=35)

    text = Canvas(frame22, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    text.create_text(215, 35, text="Your blood cells are:", fill="black", font=('Courier 14'))
    text.place(relx=0.5, rely=0.65, anchor=CENTER)

    affected = StringVar()
    label22= Label(frame22,text="", textvariable=affected,font=("candara",13,"bold"),bg="light gray")
    label22.place(relx=0.5, rely=0.75, anchor=CENTER,width=250,height=35)

    def upload_file():
        f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png'),("all files","*.*")]
        filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File",filetypes=f_types)
        img = Image.open(filename)
        img_resized=img.resize((250,250))
        photo = ImageTk.PhotoImage(img_resized)
        label = Label(frame21,image = photo)
        label.image = photo
        label.place(relx=0, rely=0)

        img = l_transformer(img)
        img = img.float()
        img = img.unsqueeze(0)

        l_path_net = l_path + r'/model.pth'
        net = lNet()
        net.load_state_dict(torch.load(l_path_net))
        outputs = net(img)
        _,predicted = torch.max(outputs.data,1)
        sm = torch.nn.Softmax()
        # print(classes[predicted.tolist()[0]].upper())
        affected.set(l_classes[predicted.tolist()[0]].upper())



    filex_button=Button(new_window, bg="#a7a7a8",cursor="hand2",fg="black",command=upload_file,borderwidth=0.1, text="select",compound=LEFT,font="Courier 19",relief=SUNKEN)
    filex_button.place(relx=0.09, rely=0.6)


    
    frame23 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame23.place(relx=0.66, rely=0.15)

    frame231= Label(frame23,text="Generate Report", font=("candara",15),bg="light gray")
    frame231.place(relx=0.5, rely=0.15, anchor=CENTER,width=250,height=35)

    ntext = Canvas(frame23, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    ntext.create_text(215, 35, text="Under Maintainance!!!", fill="black", font=('Courier 14'))
    ntext.place(relx=0.5, rely=0.5, anchor=CENTER)

    new_window.mainloop()
#===============================================================================#

#====================================brain tumour==========================================#
b_logo = 0
b_path = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/brain_tumour'
# b_valid_data_dir = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/lukemia/training_data/fold_1'
b_train_data_dir = r'C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/brain_tumour/data'
b_classes = ('Astrocitoma T1',
 'Astrocitoma T1C+',
 'Astrocitoma T2',
 'Carcinoma T1',
 'Carcinoma T1C+',
 'Carcinoma T2',
 'Ependimoma T1',
 'Ependimoma T1C+',
 'Ependimoma T2',
 'Ganglioglioma T1',
 'Ganglioglioma T1C+',
 'Ganglioglioma T2',
 'Germinoma T1',
 'Germinoma T1C+',
 'Germinoma T2',
 'Glioblastoma T1',
 'Glioblastoma T1C+',
 'Glioblastoma T2',
 'Granuloma T1',
 'Granuloma T1C+',
 'Granuloma T2',
 'Meduloblastoma T1',
 'Meduloblastoma T1C+',
 'Meduloblastoma T2',
 'Meningioma T1',
 'Meningioma T1C+',
 'Meningioma T2',
 'Neurocitoma T1',
 'Neurocitoma T1C+',
 'Neurocitoma T2',
 'Oligodendroglioma T1',
 'Oligodendroglioma T1C+',
 'Oligodendroglioma T2',
 'Papiloma T1',
 'Papiloma T1C+',
 'Papiloma T2',
 'Schwannoma T1',
 'Schwannoma T1C+',
 'Schwannoma T2',
 'Tuberculoma T1',
 'Tuberculoma T1C+',
 'Tuberculoma T2',
 '_NORMAL T1',
 '_NORMAL T2')

b_transformer = torchvision.transforms.Compose([torchvision.transforms. Resize((224,224)),torchvision.transforms.RandomHorizontalFlip(p=0.5), torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.RandomRotation(10), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.7, 0.2]),])
b_trainset = ImageFolder(b_train_data_dir, transform=l_transformer)
b_batch_size=16
b_trainloader = torch.utils.data.DataLoader(b_trainset, batch_size=l_batch_size, shuffle=True, num_workers=2)

class bNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=53,kernel_size=4,stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2= nn.Conv2d(53, 106, kernel_size=4, stride=2, padding=3)
        self.conv3 = nn.Conv2d(106, 212, kernel_size=4, stride=2, padding=3)
        self.fc1 = nn.Linear(3392, 1908)
        self.fc2= nn. Linear(1908, 44)

    def forward(self, x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x= self.pool(F.relu(self.conv3(x)))
        x= torch.flatten(x, 1)
        x= F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


bnet = bNet()
bnet = bnet.to(device)

def train_btumor_engine():
    global b_path,b_train_data_dir,bNet,b_classes
    
    


    criterion = nn.CrossEntropyLoss()
    lr_dynamic = 0.01
    momentum_dynamic = 0.9
    optimizer = optim.SGD(bnet.parameters(), lr=lr_dynamic, momentum=momentum_dynamic)

    Epoch_list =[]
    Loss_list =[]


    for epoch in range(50):
        if epoch >= 10:
            if sum(Loss_list[-5:]) > sum(Loss_list[-10:-5]):
                break
            running_loss=0.0
            for i, data in enumerate(b_trainloader, 0):
                inputs, labels =data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = bnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    if i==99:
                        Epoch_list.append((epoch) + 0.25)
                        Loss_list.append(running_loss)

                    if i==199:
                        Epoch_list.append((epoch) + 0.5)
                        Loss_list.append(running_loss)

                    if i==299:
                        Epoch_list.append((epoch) + 0.75)
                        Loss_list.append(running_loss)

                    if i==399:
                        Epoch_list.append((epoch) + 1)
                        Loss_list.append(running_loss)

                    running_loss = 0.0
                    lr_dynamic = 0.96*lr_dynamic
                    optimizer = optim.SGD(bnet.parameters(),lr=lr_dynamic,momentum=momentum_dynamic)

    print("finished")
    b_logo = 1
    if b_logo ==1:
        img = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/completed.png")
        img_resized=img.resize((110,110))
        photo = ImageTk.PhotoImage(img_resized)
        brain_logo = Label(frame21,image = photo,bg="#a7a7a8")
        brain_logo.image = photo
        brain_logo.place(relx=0.32, rely=0.1)


    b_path_net = l_path + r'/model.pth'
    torch.save(lnet.state_dict(),b_path_net)
#==========================================================================================#
img2 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/6.png")
photo2 = ImageTk.PhotoImage(img2)
e2_button=Button(frame1, bg="#6a6a6b",cursor="hand2",fg="black",command=train_btumor_engine,borderwidth=0.1,image=photo2,text="",compound=LEFT,relief=SUNKEN)
e2_button.place(relx=0.32, rely=0.74)
#========================================brain tumour window==============================================#
def Engine_2():
    new_window = Toplevel(root)
    new_window.title('Engine Two')
    new_window.geometry('1000x620')
    new_window.resizable(0, 0)

    frame1 = Frame(new_window,bg="#6a6a6b",width=1000,height=70)
    frame1.pack()

    canvas1 = Canvas(frame1, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    canvas1.create_text(215, 35, text="Brain Tumor Classification", fill="white", font=('Courier 20 bold'))
    canvas1.place(relx=0.5, rely=0.39, anchor=CENTER)

    frame21 = Frame(new_window,bg="#6a6a6b",width=250,height=250)
    frame21.place(relx=0.032, rely=0.15)

    
    frame22 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame22.place(relx=0.32, rely=0.15)

    frame221 = Frame(frame22,bg="white",width=100,height=100)
    frame221.place(relx=0.1, rely=0.04)


    def upload_photo():
        f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png'),("all files","*.*")]
        filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File",filetypes=f_types)
        img = Image.open(filename)
        img_resized=img.resize((100,100))
        photo = ImageTk.PhotoImage(img_resized)
        label = Label(frame221,image = photo)
        label.image = photo
        label.place(relx=0, rely=0)


    upload_button=Button(frame22, bg="#a7a7a8",cursor="hand2",fg="black",borderwidth=0.1, text="upload",command=upload_photo,compound=LEFT,font="Courier 13",relief=SUNKEN)
    upload_button.place(relx=0.6, rely=0.1)

    frame222= Label(frame22,text="Name", font=("candara",13),bg="light gray")
    frame222.place(relx=0.5, rely=0.35, anchor=CENTER,width=250,height=35)

    frame223= Label(frame22,text="Age", font=("candara",13),bg="light gray")
    frame223.place(relx=0.5, rely=0.45, anchor=CENTER,width=250,height=35)

    frame224= Label(frame22,text="Sex", font=("candara",13),bg="light gray")
    frame224.place(relx=0.5, rely=0.55, anchor=CENTER,width=250,height=35)

    text = Canvas(frame22, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    text.create_text(215, 35, text="The tumor is type of:", fill="black", font=('Courier 14'))
    text.place(relx=0.5, rely=0.65, anchor=CENTER)

    affected = StringVar()
    label22= Label(frame22,text="", textvariable=affected,font=("candara",13,"bold"),bg="light gray")
    label22.place(relx=0.5, rely=0.75, anchor=CENTER,width=250,height=35)

    def upload_file():
        f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png'),("all files","*.*")]
        filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File",filetypes=f_types)
        img = Image.open(filename)
        img_resized=img.resize((250,250))
        photo = ImageTk.PhotoImage(img_resized)
        label = Label(frame21,image = photo)
        label.image = photo
        label.place(relx=0, rely=0)

        img = b_transformer(img)
        img = img.float()
        img = img.unsqueeze(0)

        b_path_net = b_path + r'/model.pth'
        bnet = bNet()
        bnet.load_state_dict(torch.load(b_path_net))
        outputs = bnet(img)
        _,predicted = torch.max(outputs.data,1)
        sm = torch.nn.Softmax()
        # print(classes[predicted.tolist()[0]].upper())
        affected.set(b_classes[predicted.tolist()[0]].upper())



    filex_button=Button(new_window, bg="#a7a7a8",cursor="hand2",fg="black",command=upload_file,borderwidth=0.1, text="select",compound=LEFT,font="Courier 19",relief=SUNKEN)
    filex_button.place(relx=0.09, rely=0.6)


    
    frame23 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame23.place(relx=0.66, rely=0.15)

    frame231= Label(frame23,text="Generate Report", font=("candara",15),bg="light gray")
    frame231.place(relx=0.5, rely=0.15, anchor=CENTER,width=250,height=35)

    ntext = Canvas(frame23, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    ntext.create_text(215, 35, text="Under Maintainance!!!", fill="black", font=('Courier 14'))
    ntext.place(relx=0.5, rely=0.5, anchor=CENTER)

    new_window.mainloop()
#==================================================================================================#
def Engine_3():
    new_window = Toplevel(root)
    new_window.title('Engine Three')
    new_window.geometry('1000x620')
    new_window.resizable(0, 0)

    frame1 = Frame(new_window,bg="#6a6a6b",width=1000,height=70)
    frame1.pack()

    canvas1 = Canvas(frame1, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    canvas1.create_text(215, 35, text="Cancer Prediction Software", fill="white", font=('Courier 20 bold'))
    canvas1.place(relx=0.5, rely=0.39, anchor=CENTER)

    frame21 = Frame(new_window,bg="#6a6a6b",width=250,height=250)
    frame21.place(relx=0.032, rely=0.15)

    frame21 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame21.place(relx=0.32, rely=0.15)

    frame21 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame21.place(relx=0.66, rely=0.15)


    new_window.mainloop()

def Engine_4():
    new_window = Toplevel(root)
    new_window.title('Engine Four')
    new_window.geometry('1000x620')
    new_window.resizable(0, 0)

    frame1 = Frame(new_window,bg="#6a6a6b",width=1000,height=70)
    frame1.pack()

    canvas1 = Canvas(frame1, width= 430, height= 70, bg="#6a6a6b",bd=0,highlightthickness=0, relief='ridge')
    canvas1.create_text(215, 35, text="Cancer Prediction Software", fill="white", font=('Courier 20 bold'))
    canvas1.place(relx=0.5, rely=0.39, anchor=CENTER)

    frame21 = Frame(new_window,bg="#6a6a6b",width=250,height=250)
    frame21.place(relx=0.032, rely=0.15)

    frame21 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame21.place(relx=0.32, rely=0.15)

    frame21 = Frame(new_window,bg="#6a6a6b",width=300,height=500)
    frame21.place(relx=0.66, rely=0.15)


    new_window.mainloop()


imge1 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/2.png")
photoe1 = ImageTk.PhotoImage(imge1)
e1predict_button=Button(root,cursor="hand2",fg="black",command=Engine_1,borderwidth=0.1, text="",compound=LEFT,image=photoe1,relief=SUNKEN)
e1predict_button.place(relx=0.05, rely=0.8)

imge2 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/3.png")
photoe2 = ImageTk.PhotoImage(imge2)
e2predict_button=Button(root,cursor="hand2",fg="black",command=Engine_2,borderwidth=0.1, text="",compound=LEFT,image=photoe2,relief=SUNKEN)
e2predict_button.place(relx=0.32, rely=0.8)

imge3 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/4.png")
photoe3 = ImageTk.PhotoImage(imge3)
e3predict_button=Button(root,cursor="hand2",fg="black",command=Engine_1,borderwidth=0.1, text="",compound=LEFT,image=photoe3,relief=SUNKEN)
e3predict_button.place(relx=0.58, rely=0.8)

imge4 = Image.open("C:/transfer/new f drive/Godmode/@pgec project/cancer prediction software/Proxlight_Designer_Export/5.png")
photoe4 = ImageTk.PhotoImage(imge4)
e4predict_button=Button(root,cursor="hand2",fg="black",command=Engine_1,borderwidth=0.1, text="",compound=LEFT,image=photoe4,relief=SUNKEN)
e4predict_button.place(relx=0.83, rely=0.8)
root.mainloop()


