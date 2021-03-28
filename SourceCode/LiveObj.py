from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
import csv
import threading
import _thread
from multiprocessing import Process
import math

from QTable import RL_Qtable
from Dataset import LocationCalculate


ShrinkV=3
bufferLen=4
UserView=200
lenU = 120          #korean 240
Tracking= True

def FromObjectToUpdatTile(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,tileNo):
    L = len(Namelist)
    if L>10:
        K=50
    else:
        K=100
    for i in range(0, L):
        X=CorUPleft[i][0]
        Y=CorUPleft[i][1]
        XL=CorDownRig[i][0]
        YL=CorDownRig[i][1]
        if XL-X<150 or YL-Y<150:
            X = X - K
            XL = XL + K
            YL=YL+150
            Y=Y-150
            if X<0:
                X=0
            if Y<=0:
                Y=0
            if XL>width:
                XL=width
            if YL>height:
                YL=height
        UpdateTileStatuesBasedOnObject(TileStatus, width, height, tileNo, X, Y, XL, YL)

def FromObjectToUpdatTileOnTracking(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,tileNo, UX, UY, UXL, UYL):
    L = len(Namelist)
    if L>10:
        K=50
    else:
        K=100
    for i in range(0, L):
        X=CorUPleft[i][0]
        Y=CorUPleft[i][1]
        XL=CorDownRig[i][0]
        YL=CorDownRig[i][1]
        if ((X<=UXL and X>=UX )or (XL<=UXL and XL>=UX )) and ((Y<=UYL and Y>=UY )or (YL<=UYL and YL>=UY )):
            if XL-X<200 or YL-Y<200:
                X=X-K
                XL=XL+K
                YL=YL+150
                Y=Y-150
                if X<0:
                    X=0
                if Y<=0:
                    Y=0
                if XL>width:
                    XL=width
                if YL>height:
                    YL=height
            UpdateTileStatuesBasedOnObject(TileStatus, width, height, tileNo, X, Y, XL, YL)


def CheckIFUserViewCovered (TileStatus,w,h,tileNo,X,Y,XL,YL):
    wd = w / tileNo
    hd = h / tileNo
    IL = math.floor(X / wd)
    IH = math.ceil(XL / wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH>10:
        JH=10
    if IH>10:
        IH=10
    for i in range(IL, IH ):
        for j in range(JL, JH ):
            if TileStatus[i + j * tileNo] ==0:
                return False
    return True

def UpdateTileStatues(TileStatus):
    i=len(TileStatus)
    for k in range(0,i):
        if TileStatus[k]>0:
            TileStatus[k]=TileStatus[k]-1

def DrawTiles(TileStatus,w,h,tileNo,img):
    wd=w/tileNo
    hd=h/tileNo
    color=(255,0,255)
    color2 = (0, 255, 0)
    for i in range(0,tileNo*tileNo):
        Y=i//tileNo
        X=i%tileNo
        a=int(X*wd)
        b=int(Y*hd)
        C1=[a,b]
        a1=int(X * wd+wd)
        b1=int(Y * hd+hd)
        C2 =[a,b]
        if TileStatus[i] == 0:
            cv2.rectangle(img, (a,b), (a1,b1), color, 1)
        else:
            cv2.rectangle(img, (a,b), (a1,b1), color2, 1)
# the object size should be larger than the tile
#X,Y is the coordinate of the upperleft XL,YL is the down right
def UpdateTileStatuesBasedOnObject(TileStatus,w,h,tileNo,X,Y,XL,YL):
    wd = w / tileNo
    hd = h / tileNo
    IL=math.floor(X/wd)
    IH=math.ceil(XL/wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH >= 10:
        JH = 9
    if IH >= 10:
        IH = 9
    for i in range(IL, IH + 1):
        for j in range(JL, JH + 1):
            #if j>6:
                #print("==================")
                #print(YL)
            if i+j*tileNo>99:
                print(i,j,i+j*tileNo)
            TileStatus[i+j*tileNo]=ShrinkV


def ReadAllUserData(FileHead,NO):
    AllUserData=[]
    for i in range(1,NO+1):
        FileName=FileHead+str(i)+'.csv'
        if i==1:
            flag=1
        else:
            flag=0
        OneUserData,TimeStamp = ReadOneuserData(FileName,flag)
        AllUserData.append(OneUserData)
    return AllUserData,TimeStamp

def ReadOneuserData(FileName,flagTime):
    TimeStamp = []
    #flagTime = 1
    Userdata = []
    with open(FileName) as csvfile:
        csv_reader = csv.reader(csvfile)  
        birth_header = next(csv_reader) 
        for row in csv_reader:  
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    Userdata = [[float(x) for x in row] for row in Userdata]  
    Userdata = np.array(Userdata)  
    return Userdata,TimeStamp

def GetUserDataPerframe(FileName,flagTime,frameRate,ResolutionW,ResolutionH,LenU):
    flagTime=1
    OneUserData,TimeStamp=ReadOneuserData(FileName,flagTime)
    str = TimeStamp[0].split(':')
    PreTime = math.ceil(float(str[2]))
    CurTime = math.floor(float(str[2]))
    count=0
    i=0
    #frameRate=30
    UserDperFrame = []
    FrameCount=frameRate+1
    while count<LenU:
        if FrameCount >= frameRate:
            FrameCount=0
            UserIn1s = []
            Ind = 0
            while PreTime > CurTime:
                str = TimeStamp[i].split(':')
                # print(str[2])
                # Ind=Ind+1
                # print([Ind,j])
                CurTime = math.floor(float(str[2]))
                if CurTime == 0 and PreTime == 60:
                    break

                x = OneUserData[i][1]
                y = OneUserData[i][2]
                z = OneUserData[i][3]
                w = OneUserData[i][4]
                H, W = LocationCalculate(x, y, z, w)
                IH = math.floor(H * ResolutionH)
                IW = math.floor(W * ResolutionW)
                #UserAll.append([IW, IH])
                    # print(IW,IH)
                i=i+1
                UserIn1s.append([IW, IH])
                Ind=Ind+1
            #print(Ind)
            FrameCount = 0
            PreTime = CurTime + 1
            #UL=len(UserIn1s)
            UL=Ind
        
            if UL < frameRate:
                for k in range(0, UL):
                    UserDperFrame.append(UserIn1s[k])
                for k in range(UL - 1, frameRate):
                    UserDperFrame.append(UserIn1s[UL - 1])
            else:
                step=math.floor(UL/frameRate)
                for k in range(0, frameRate):
                    UserDperFrame.append(UserIn1s[k*step])
                # UserIn1s.append(UserAllR)
        else:
            count=count+1
            FrameCount = frameRate + 1
    return UserDperFrame



def multiThread_feature(img,i):
    color = (255, 0, 0)
    cv2.rectangle(img, (400, 300), (1000, 600), color, 1)
    print ("good")
    print(i)

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img, Namelist, CorUPleft, CorDownRig):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    Namelist.append(label)
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    CorUPleft.append(c1)
    CorDownRig.append(c2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # cv2.rectangle(img, c1, c2,color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


def UpdateTrack(Namelist, CorUPleft, CorDownRig, FNamelist, FCorUPleft, FCorDownRig, FConfi):
    Count = 0
    #print("tital list")
    #print(len(FNamelist))
    #print("new list")
    #print(len(Namelist))
    L = len(FNamelist)
    NewList = []
    for j in range(0, len(Namelist)):
        NewList.append(0)
    for i in range(0, L):
        D = []
        for j in range(0, len(Namelist)):
            if Namelist[j] == FNamelist[i] and NewList[j] == 0:
                UL = (CorUPleft[j][0] - FCorUPleft[i][0]) ** 2 + (CorUPleft[j][1] - FCorUPleft[i][1]) ** 2
                # RD=(CorDownRig[0] - FCorDownRig[i][0]) ** 2 + (CorDownRig[1] - FCorDownRig[i][1]) ** 2
                D.append(UL.item())
                # print(UL.item())
            else:
                D.append(10000)
        K = D.copy()
        K.sort()
        #print(D)
        if K[0] < 400:
            I = D.index(K[0])
            FConfi[i] = 1
            FCorUPleft[i] = CorUPleft[I]
            # FCorUPleft[i][1] = CorUPleft[I][1]
            FCorDownRig[i] = CorDownRig[I]
            # FCorDownRig[i][1] = CorDownRig[I][1]
            Count = Count + 1
            NewList[I] = 1
            #print(I)

        else:
            FConfi[i] = 0
    Cad = 0
    for j in range(0, len(Namelist)):
        if NewList[j] == 0:
            FNamelist.append(Namelist[j])
            FCorUPleft.append(CorUPleft[j])
            # FCorUPleft[i][1] = CorUPleft[I][1]
            FCorDownRig.append(CorDownRig[j])
            FConfi.append(1)
            Cad = Cad + 1
    #print("Total mathed")
    #print(Count)
    #print("Total added")
    #print(Cad, NewList.count(0))


def DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im):
    for i in range(0, len(FNamelist)):
        if FConfi[i] == 1:
            color = (i * 5, 255 - i * 5, i * 7)
            cv2.rectangle(orig_im, FCorUPleft[i], FCorDownRig[i], color, 1)
            t_size = cv2.getTextSize(FNamelist[i], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = FCorUPleft[i][0] + t_size[0] + 3, FCorUPleft[i][1] + t_size[1] + 4
            # cv2.rectangle(img, FCorUPleft[i], c2, color, -1)
            cv2.putText(orig_im, str(i), (FCorUPleft[i][0], FCorUPleft[i][1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,1, [225, 255, 255], 1)


if __name__ == '__main__':

    for UserIndex in range(1,49):
        args = arg_parse()
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        start = 0

        CUDA = torch.cuda.is_available()

        num_classes = 80

        CUDA = torch.cuda.is_available()

        bbox_attrs = 5 + num_classes

        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        if CUDA:
            model.cuda()

        model(get_test_input(inp_dim, CUDA), CUDA)

        model.eval()

        videofile = args.video
    # Read file operation
        #=====================================================================
        VideoName = "1-7-Cooking BattleB"          #1-2-FrontB  1-1-Conan Gore FlyB  1-9-RhinosB
        videofile = VideoName + ".mp4"  # 2-4-FemaleBasketballB 2-6-AnittaB 1-6-FallujaB 2-3-RioVRB  2-5-FightingB  2-8-reloadedB
        CSVUserFileHeader="video_5_D1_"    #videoname: A-B-Name.mp4 if A=1 add D1; B is the number for video, for user csv it is B-1
                                        #video_0_ or video_0_D1_
        fileName = "video_5_D1_"+str(UserIndex)+".csv"
        NOUsr=UserIndex                         #1-48
        CSVFinalUserFile=CSVUserFileHeader+str(NOUsr)+".csv"
        if Tracking == True:
            CSVFinalUserFileToSave="AAResult_"+CSVUserFileHeader+str(NOUsr)+"_"+str(bufferLen)+"s_tracking.csv"
        else:
            CSVFinalUserFileToSave = "AAResult_" + CSVUserFileHeader + str(NOUsr) + "_" + str(bufferLen) + "s_Basic.csv"
        outAccuBanDRes = open(CSVFinalUserFileToSave, 'a', newline='')
        csv_writeABF = csv.writer(outAccuBanDRes, dialect='excel')
        ACCbadRestul=[VideoName+str(NOUsr),"UserFeedback","ObjectDetection","Total Tiles Used","total tiles"]
        csv_writeABF.writerow(ACCbadRestul)
        SfileName = "Asave" + VideoName + ".csv"
        SvideoName = "Asave" + VideoName + ".avi"
        outF = open(SfileName, 'a', newline='')
        csv_writeF = csv.writer(outF, dialect='excel')
        cap = cv2.VideoCapture(videofile)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        assert cap.isOpened(), 'Cannot capture source'

        '''initialization '''
        ret, frame = cap.read()
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        height, width = frame.shape[:2]

        out = cv2.VideoWriter(SvideoName, fourcc, 30.0, (width, height))

        # Some parameters here:
        Framerate = int(cap.get(5))
        TileNO=10           # the final number should be TileNO*TileNO
        TileStatus=[3]*TileNO*TileNO


        UserDataPF = GetUserDataPerframe(fileName, 1, Framerate, width, height, lenU)
        UserLenData=len(UserDataPF)
        ''' ===========================test video and user data================================'''
        TestUserDataInVIdeo=False
        if TestUserDataInVIdeo == True:
            lenU = 120
            LenNew = lenU * 30
            fileName = "video_0_2.csv"
            f = 1
            i=0
            UserDat = GetUserDataPerframe(fileName, Framerate, 30, 1280, 720, lenU)
            while cap.isOpened():
                ret, frame = cap.read()

                color = (255,255,0)
                C1=(UserDat[i][0]-100,UserDat[i][1]-100)
                C2 = (UserDat[i][0]+100, UserDat[i][1]+100)
                #C1=200,300
                #C2=300,400
                #c1 = 600, 200  # 500,200
                #c2 = 900, 500  # 800,600
                color = (255, 255, 0)
                #cv2.circle(frame, (300, 400), 10, color=(0, 255, 255))
                #cv2.rectangle(frame, c1, c2, color, -1)
                cv2.rectangle(frame,C1 , C2,color, 1)
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                i=i+1
        '''===========test end=================='''

        '''list the objects'''
        FNamelist = []
        FCorUPleft = []
        FCorDownRig = []
        FConfi = []
        IsFirst = True
        '''
        Test multi thread
        '''
        KKK=0
        _thread.start_new_thread(multiThread_feature,(IsFirst,KKK))
        KKK2=0
        _thread.start_new_thread(multiThread_feature,(IsFirst,KKK2))
        frames = 0
        start = time.time()
        CountRunTime=0
        Accur=0
        CountInOnebuffer=0
        while cap.isOpened():
            Accur = 1
            ret, frame = cap.read()
            if ret:

                img, orig_im, dim = prep_image(frame, inp_dim)

                im_dim = torch.FloatTensor(dim).repeat(1, 2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

                if type(output) == int:
                    print("第一次")
                    frames += 1
                    print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                    cv2.imshow("frame", orig_im)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    continue

                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))

                Namelist = []
                CorUPleft = []
                CorDownRig = []
                Confi = []

                '''
                feature extract and match
                '''
                '''
                # exchange img1 and img2: makesure img2 is always the previous one
                img2 = img1.copy()
                # convert the new input image to Gray scale
                img1 = cv2.cvtColor(orig_im, cv2.COLOR_RGB2GRAY)
                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                #            matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                #            matches = sorted(matches, key=lambda x: x.distance)
                # print(matches[0:10])
                '''
                # check all the objects that are detected
                list(map(lambda x: write(x, orig_im, Namelist, CorUPleft, CorDownRig), output))
                #get user feedback and check the update：
                UsrCurX=0
                UsrCurY=0
                UserFeedBackResult=10
                if CountRunTime>bufferLen*Framerate+Framerate:
                    UserFeedBackResult=5
                    UsrCurX=UserDataPF[CountRunTime-bufferLen*Framerate][0]
                    UsrCurY = UserDataPF[CountRunTime - bufferLen * Framerate][1]
                    Vx=0
                    Vy=0
                    for ind in range(CountRunTime-(bufferLen*Framerate+Framerate),CountRunTime-bufferLen*Framerate):
                        Vx=UserDataPF[ind+1][0]-UserDataPF[ind][0]+Vx
                        Vy = UserDataPF[ind + 1][1] - UserDataPF[ind][1] + Vy
                    Vx=Vx/Framerate
                    Vy = Vy / Framerate
                    UsrCurX=int(UsrCurX+Vx)
                    UsrCurY=int(UsrCurY+Vy)
                    X = int(UsrCurX - UserView)
                    Y = int(UsrCurY - UserView)
                    XL = int(UsrCurX + UserView)
                    YL = int(UsrCurY + UserView)
                    if X <= 0:
                        X = 0
                    if Y < 0:
                        Y = 0
                    if XL > width:
                        XL = width
                    if YL > height:
                        YL = height
                    UsRX=X
                    UsRL=Y
                    UpdateTileStatuesBasedOnObject(TileStatus, width, height, TileNO, X, Y, XL, YL)

                #Update Tile based on the objects
                print(CountRunTime,lenU*Framerate,UserLenData)
                if Tracking ==True and CountRunTime>bufferLen*Framerate+Framerate:
                    FromObjectToUpdatTileOnTracking(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,TileNO, X, Y, XL, YL)
                else:
                    FromObjectToUpdatTile(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,TileNO)
                #Check accuracy
                X=UserDataPF[CountRunTime][0]-UserView/2
                Y=UserDataPF[CountRunTime][1]-UserView/2
                XL = UserDataPF[CountRunTime][0] + UserView / 2
                YL = UserDataPF[CountRunTime][1] + UserView / 2
                if X<=0:
                    X=0
                if Y<0:
                    Y=0
                if XL>width:
                    XL=width
                if YL>height:
                    YL=height
                C1=(int(X),int(Y))
                C2=(int(XL),int(YL))

                #Update based on the Q-table
                RL_Qtable(TileStatus, width, height, TileNO, X, Y, XL, YL)

                #Check the performance
                Res=CheckIFUserViewCovered(TileStatus, width, height, TileNO, X, Y, XL, YL)
                if Res==False:
                    Accur = 0
                    print("wrong")
                CountRunTime=CountRunTime+1
                if CountRunTime%10 ==0:
                    UpdateTileStatues(TileStatus)
                #AccuracyBased on user feedback

                if UserFeedBackResult==5:
                    if (UsRX<=X and UsRX+20>=X) and (UsRL<=Y and UsRL+20>Y):
                        UserFeedBackResult=1
                    else:
                        UserFeedBackResult=0

                #show result in video:
                DrawTiles(TileStatus, width, height, TileNO, orig_im)
                color=(0,0,255)
                cv2.rectangle(orig_im, C1, C2, color, 1)
                ACCbadRestul = [time.time() - start, UserFeedBackResult, Accur, TileNO*TileNO-TileStatus.count(0),
                                TileNO*TileNO]
                csv_writeABF.writerow(ACCbadRestul)
                # print(Namelist)
                '''
                Below add tracking part
                '''
                '''
                list(Namelist)
                list(CorUPleft)
                list(CorDownRig)
                list(FNamelist)
                list(FCorUPleft)
                list(FCorDownRig)
                list(FConfi)
                '''

                '''
    
                # 
                if IsFirst == False:
                    UpdateTrack(Namelist, CorUPleft, CorDownRig, FNamelist, FCorUPleft, FCorDownRig, FConfi)
                    DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im)
                else:
                    FNamelist = Namelist.copy()
                    FCorUPleft = CorUPleft.copy()
                    FCorDownRig = CorDownRig.copy()
                    for i in range(0, len(FNamelist)):
                        FConfi.append(1)
                    DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im)
    
                IsFirst = False
                # print("2")
                stu3 = []
                for i in range(0, len(FNamelist)):
                    if FConfi[i] == 1:
                        stu3.append(str(i))
                        stu3.append(FCorUPleft[i][0].item())
                        stu3.append(FCorUPleft[i][1].item())
                        stu3.append(FCorDownRig[i][0].item())
                        stu3.append(FCorDownRig[i][1].item())
                csv_writeF.writerow(stu3)
    
                '''
                cv2.imshow("frame", orig_im)
                out.write(orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                if CountRunTime>lenU*Framerate-1:
                    break


            else:
                break

        cap.release()
        out.release()


