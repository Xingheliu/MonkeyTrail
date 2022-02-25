import csv
from moviepy.editor import VideoFileClip
from pylab import *
from pro.tool import video2frames
import os
path = "D:\\study_2021\\data\\yolo_train_data\\my_work_train\\"#"D:\\workdata\\monkeymovie\\2020video_3"
# path_video = "D:\\workdata\\2021\\Voc\\video\\cut_ch01_20190602061150.mp4"#"D:\\workdata\\monkeymovie\\2020video_3\\ch08_20200919000000.mp4"
# path_out='D:\\workdata\\monkeymovie\\20201012\\train_movie\\ch03_right_cut_drink_2.avi'#D:\\study\\bishe\\ppt_video\\video\\jianji
# path_out1='D:\\workdata\\Amonkey_20200905\\ch01_action_a_m02_cut.avi'



# count_csv_path = "E:\\movie_3_2019_hou\\interest_box"
# with open(count_csv_path, 'a', newline='') as f:
#     f_csv = csv.writer(f)
#     headers = [interest_box_min, interest_box_max]
#     f_csv.writerow(headers)
# clip1 = VideoFileClip(path).subclip(1458,1502).crop(x1=int(interest_box_min[0]),y1=int(interest_box_min[1]),x2=int(interest_box_max[0]),y2=int(interest_box_max[1]))#28-39
# clip1.write_videofile(path_out,codec='libx264', audio=False)
import cv2
def batch_volumex(path):
    # 函数功能：在指定路径下，将该文件夹的视频声音调为x倍
    origin_path = os.getcwd()
    os.chdir(path)

    in_box = {}
    for fname in os.listdir(path):
        if len(fname.split('_'))<2:
            continue
        im_pathOut = './temp'
        video2frames(path+fname, im_pathOut, extract_time_points=(2, 3, 4))
        special_frame = array(cv2.imread('./temp/frame_000001.bmp'))
        RoI = cv2.selectROI(windowName="roi", img=special_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        init_box = [int(RoI[0]),int(RoI[1]),int(RoI[0] + RoI[2]),int(RoI[1] + RoI[3])]
        in_box[fname]= init_box
    print(in_box)
    for fname in os.listdir(path):
        if len(fname.split('_'))<2:
            continue
        print(fname)
        init_box = in_box[fname]
        clip1 = VideoFileClip(fname).crop(x1=int(init_box[0]), y1=int(init_box[1]),
                                                             x2=int(init_box[2]),
                                                             y2=int(init_box[3]))  # 28-39
        clip1.write_videofile(path+"cut\\cut_"+fname, codec='libx264', audio=False)
    os.chdir(origin_path)
#
batch_volumex(path)