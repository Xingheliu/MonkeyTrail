import cv2
import csv
import os
from shapely.geometry import Polygon as Po
# from pro.ssd_video_set import SSD_M
import glob
from pro.yolo_video_set import YOLO_M
import pandas as pd
from moviepy.editor import VideoFileClip
from pro.tool import video2frames, get_pot, get_unin
from PIL import Image
from pylab import *
from pathlib import Path
import copy
import shutil

class MonkeyTrail_Step():
    def __init__(self,video_path='',
                 output_path='',
                 env='light',
                 preferences='top',
                 deep_learing_mode='yolo',
                 create_dataset=False,
                 save_frame_img_i=0,
                 background_during=40,
                 crop_video=True,
                 RoI=1,
                 RoUI=1,
                 set_fps_num=2):
        if not os.path.exists('./temp'):
            os.mkdir('./temp')

        self.env = env
        self.preferences = preferences  # 'top'
        if deep_learing_mode == 'ssd':
            self.deep_mode = 1  # 1:ssd;2:yolo
        else:
            self.deep_mode = 2
        self.create_dataset_flag = create_dataset
        self.background_during = background_during  # min
        self.save_frame_img_i = save_frame_img_i
        self.set_fps_num = set_fps_num
        self.out_put_fps = set_fps_num

        self.crop_video = crop_video

        self.day_night = ''#'night'
        self.class_id = 0

        self.video_path_all = video_path
        self.output_path_all = output_path

        self.RoI_num = RoI
        self.RoUI_num = RoUI

        self.R_L_BOX_FLAG = 0


    def init_value(self):
        self.video_output = self.video_save_path + '/temp.avi'
        self.video_output_ssd = self.video_save_path + '/temp_ssd.avi'
        self.path_csv, self.path_csv1, self.path_csv2 = self.video_save_path + '/temp.csv', self.video_save_path + '/temp1.csv', self.video_save_path + '/temp2.csv'
        self.frame_path_csv = self.video_save_path + '/frame_set.csv'
        self.empty_background_path = self.video_save_path + '/background' + self.video_name
        if not os.path.exists(self.empty_background_path):
            os.mkdir(self.empty_background_path)
        clip = VideoFileClip(self.video_input)
        self.fps = clip.fps

        self.set_fps_num = self.fps
        self.out_put_fps = self.fps

        self.during = clip.duration
        [self.width, self.height] = clip.size
        self.size_th = self.width * self.height // 18
        self.reliable = np.zeros((self.height, self.width), dtype=np.uint8)
        self.reliable[1:-1, 1:-1] = 1
        self.area_delete_line = 100
        self.frame_num = 0
    def pre_processing(self):
        im_pathOut =  self.video_save_path
        video2frames(self.video_input, im_pathOut, extract_time_points=(2, 3, 4))
        special_frame = array(cv2.imread( self.video_save_path+'/frame_000001.bmp'))

        if self.crop_video:
            print('Please crop out macaques tracking area ; Then press "Enter" to confirm')
            self.CUT = cv2.selectROI(windowName="Crop roi", img=special_frame, showCrosshair=True, fromCenter=False)
            self.CUT_box = [int(self.CUT[0]),int(self.CUT[1]),int(self.CUT[0] + self.CUT[2]),int(self.CUT[1] + self.CUT[3])]
            special_frame = special_frame[self.CUT_box[1]:self.CUT_box[3],self.CUT_box[0]:self.CUT_box[2],:]
            cv2.destroyAllWindows()


        print('Please select the location of the macaque initial tracking box ; Then press "Enter" to confirm')
        self.RoI = cv2.selectROI(windowName="Animal init roi", img=special_frame, showCrosshair=True, fromCenter=False)
        self.init_box = [int(self.RoI[0]),int(self.RoI[1]),int(self.RoI[0] + self.RoI[2]),int(self.RoI[1] + self.RoI[3])]
        self.init_box_size = (self.init_box[2]-self.init_box[0])*(self.init_box[3]-self.init_box[1])

        if self.R_L_BOX_FLAG==1:
            self.RoI = cv2.selectROI(windowName="roiL", img=special_frame, showCrosshair=True, fromCenter=False)
            self.L_box = [int(self.RoI[0]),int(self.RoI[1]),int(self.RoI[0] + self.RoI[2]),int(self.RoI[1] + self.RoI[3])]
            self.RoI = cv2.selectROI(windowName="roiR", img=special_frame, showCrosshair=True, fromCenter=False)
            self.R_box = [int(self.RoI[0]),int(self.RoI[1]),int(self.RoI[0] + self.RoI[2]),int(self.RoI[1] + self.RoI[3])]
        cv2.destroyAllWindows()
        mid = get_pot(special_frame)
        self.mid_x_thL, self.mid_x_thR = mid[0], mid[0]
        self.mid_y_thT, self.mid_y_thB = mid[1], mid[1]

        [height, width] = shape(special_frame)[0:2]
        self.uninterest = get_unin(special_frame, self.RoUI_num, height, width)
        self.bv = get_unin(special_frame, 0, height, width)

        if self.crop_video:
            clip1 = VideoFileClip(self.video_input).crop(x1=self.CUT_box[0],
                                                       y1=self.CUT_box[1],
                                                       x2=self.CUT_box[2],
                                                       y2=self.CUT_box[3])
            clip1.write_videofile( self.video_save_path + self.video_name + '.mp4', codec='libx264', audio=False)
            self.video_input = self.video_save_path + self.video_name + '.mp4'
        self.init_value()

    def normal_image_process(self, image):
        image = image * self.reliable
        contours, hierarch = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area_delete = cv2.contourArea(contours[i])
            if area_delete < self.area_delete_line:
                cv2.drawContours(image, [contours[i]], -1, 0, thickness=-1)
                continue
        return image

    def image_zone_delete(self, dilated_adjust):
        area_count_flag = copy.deepcopy(dilated_adjust[0:len(self.h_grid),0:len(self.w_grid)])
        area_count_size = (self.height//len(self.h_grid))*(self.width//len(self.w_grid))*0.05
        for w in range(len(self.w_grid)):
            for h in range(len(self.h_grid)):
                d_grid = dilated_adjust[self.h_grid[h][0]:self.h_grid[h][1],self.w_grid[w][0]:self.w_grid[w][1]]
                g = d_grid[:,:]==255
                area_count_flag[h][w] = int(255) if len(d_grid[g])>area_count_size else int(0)

        area_count_flag = np.array(area_count_flag)
        contours, hierarch = cv2.findContours(area_count_flag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area_delete_all, area_i_all = [], []
        for i in range(len(contours)):
            area_delete = cv2.contourArea(contours[i])
            area_delete_all.append(area_delete)
            area_i_all.append(i)
        if len(area_delete_all)==0: return dilated_adjust
        max_i = area_i_all[area_delete_all.index(max(area_delete_all))]
        for i in range(len(contours)):
            if i != max_i:
                cv2.drawContours(area_count_flag, [contours[i]], -1, 0, thickness=-1)
        for w in range(len(self.w_grid)):
            for h in range(len(self.h_grid)):
                if area_count_flag[h][w] == 0:
                    dilated_adjust[self.h_grid[h][0]:self.h_grid[h][1], self.w_grid[w][0]:self.w_grid[w][1]]=0

        return dilated_adjust

    def short_increase_delete(self, dilated_adjust, last_data, cur_data):
        [lx1, ly1, lx2, ly2] = last_data
        [x1, y1, x2, y2] = cur_data
        flag = 0

        last_box = Po([(lx1, ly1), (lx1, ly2), (lx2, ly2), (lx2, ly1)])
        cur_box = Po([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])
        diff_area_size = cur_box.difference(last_box).area

        if y1< self.mid_y_thT and y2>self.mid_y_thT:
            g = dilated_adjust[y1:int(self.mid_y_thT), x1:x2] == 255
            area_count1 = len(dilated_adjust[y1:int(self.mid_y_thT), x1:x2][g])
            g = dilated_adjust[int(self.mid_y_thT):y2, x1:x2] == 255
            area_count2 = len(dilated_adjust[int(self.mid_y_thT):y2, x1:x2][g])
            area_count = area_count1 + area_count2
            bot_box = Po([(x1,int(self.mid_y_thT)),(x1,y2),(x2,y2),(x2,int(self.mid_y_thT))])
            if self.preferences == 'top' and y1 < self.mid_y_thT and y2 - y1 > self.height * 0.5 and area_count2 / bot_box.area < 0.05:
                flag = 1
                return last_data,flag
        else:
            g = dilated_adjust[y1:y2, x1:x2] == 255
            area_count = len(dilated_adjust[y1:y2, x1:x2][g])

        if cur_box.intersects(last_box):
            same_point = cur_box.intersection(last_box).bounds
            g = dilated_adjust[int(same_point[1]):int(same_point[3]),int(same_point[0]):int(same_point[2])] == 255
            area_diff = area_count - len(dilated_adjust[int(same_point[1]):int(same_point[3]),int(same_point[0]):int(same_point[2])][g])
        else:
            area_diff = area_count
        if area_diff>0 and diff_area_size>0:
            if area_diff/diff_area_size<0.05:
                cur_data = last_data
                flag = 1

        return cur_data,flag

    def frame_diff_track(self, image, last_frame, last_data):
        self.frame_num = self.frame_num + 1
        currentframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        currentframe = cv2.absdiff(currentframe, last_frame)
        median = cv2.medianBlur(currentframe, 3)
        th1 = cv2.threshold(median.copy(), 20, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
        dilated_adjust = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        dilated_adjust = self.normal_image_process(dilated_adjust)

        col, row = dilated_adjust.sum(0), dilated_adjust.sum(1)
        xcol, yrow = np.nonzero(col)[0], np.nonzero(row)[0]

        g = dilated_adjust[:, :] == 255
        area_count = len(dilated_adjust[g])

        if self.frame_num == 1:
            with open(self.path_csv, 'w', newline='') as f:
                f_csv = csv.writer(f)
                headers = ['frame', 'area_size', 'area_count', 'x1', 'x2', 'y1', 'y2']
                f_csv.writerow(headers)
        if xcol != [] and yrow != []:
            x1, x2 = np.nanmin(xcol), np.nanmax(xcol)
            y1, y2 = np.nanmin(yrow), np.nanmax(yrow)
        else:
            area_count, [x1, y1, x2, y2] = 0, last_data

        area_size = np.abs(x2 - x1) * 0.01 * np.abs(y2 - y1) * 0.01
        if area_size > self.size_th * 0.0001 and area_count > 100:
            last_data = [x1,x2,y1,y2]
        else:
            box_current = Po([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            [lx1,lx2,ly1,ly2] = last_data
            box_last = Po([(lx1,ly1),(lx2,ly1),(lx2,ly2),(lx1,ly2)])
            if box_last.intersects(box_current):
                x1,x2,y1,y2 = lx1,lx2,ly1,ly2
        with open(self.path_csv, 'a', newline='') as f:
            f_csv = csv.writer(f)
            row_data = [self.frame_num, area_size, area_count, x1, x2, y1, y2]
            f_csv.writerow(row_data)
        last_frame= image

        dilated_adjust = dilated_adjust
        dilated_adjust_1 = np.expand_dims(dilated_adjust, axis=2)
        img_dia = np.concatenate((dilated_adjust_1, dilated_adjust_1), axis=2)
        img_dia = np.concatenate((img_dia, dilated_adjust_1), axis=2)
        image_out = cv2.bitwise_or(image, img_dia)
        cv2.rectangle(image_out, (int(x1), int(y1)), (int(x2), int(y2)), (128, 255, 128), 2)
        cv2.putText(image_out, str(area_size)+' / '+str(self.size_th), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return image_out,last_frame,last_data

    def frame_video(self, image):
        if self.frame_num == 0:
            self.last_image = image
            self.last_data = [0, 0, 0, 0]
        image, self.last_image, self.last_data = self.frame_diff_track(image, self.last_image, self.last_data)
        return image
    def frame_diff_set(self):
        clip1 = VideoFileClip(self.video_input).set_fps(self.set_fps_num)
        video_clip1 = clip1.fl_image(self.frame_video)
        video_clip1.write_videofile(self.video_output, codec='libx264', audio=False)

        with open(self.path_csv1, 'w', newline='') as f:
            f_csv = csv.writer(f)
            headers = ['frame', 'x1', 'x2', 'y1', 'y2']
            f_csv.writerow(headers)

        with open(self.path_csv) as file:
            static_time_during = self.set_fps_num * 5
            s, e = 0, 0
            record_times = 0
            frame_csv_num = 0
            for line in file:
                line = line.split('\n')[0].split(',')
                if line[0] == 'frame': continue
                frame_csv_num = frame_csv_num + 1
                x1, x2, y1, y2 = int(line[3]), int(line[4]), int(line[5]), int(line[6])
                area_count = int(line[2])
                if area_count < 1000:
                    record_times = record_times + 1
                    if record_times == 1:
                        s, e = frame_csv_num, frame_csv_num
                    if record_times >= static_time_during:
                        e = frame_csv_num
                else:
                    if s - e < 0:
                        if x1 > self.mid_x_thR or x2 < self.mid_x_thL or y1 > self.mid_y_thB or y2 < self.mid_y_thT:
                            with open(self.path_csv1, 'a', newline='') as f:
                                f_csv = csv.writer(f)
                                headers = [int(int(line[0])/self.set_fps_num*self.fps), x1, x2, y1, y2]
                                f_csv.writerow(headers)
                    s, e = 0, 0

    def match_frame(self):
        with open( self.video_save_path+'/temp_L.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['frame_num', 'x1', 'x2', 'y1', 'y2'])
        with open( self.video_save_path+'/temp_R.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['frame_num', 'x1', 'x2', 'y1', 'y2'])
        with open(self.path_csv2) as file:
            for line in file:
                line = line.split('\n')[0].split(',')
                if line[0] == 'frame_num': continue
                x1, x2, y1, y2 = float(line[3]), float(line[4]), float(line[5]), float(line[6])
                if [x1,x2,y1,y2]==[0,0,0,0]:continue
                if x2<self.mid_x_thL:
                    with open( self.video_save_path+'/temp_L.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([float(line[0]), int(x1), int(x2), int(y1), int(y2)])
                if x1>self.mid_x_thR:
                    with open( self.video_save_path+'/temp_R.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([float(line[0]), int(x1), int(x2), int(y1), int(y2)])

        mf = pd.read_csv( self.video_save_path+'/temp_R.csv')
        mf.columns = ['frame_num', 'x1', 'x2', 'y1', 'y2']
        xm = mf[['frame_num', 'x1', 'x2', 'y1', 'y2']]
        xm = np.array(xm)

        Rxm = np.lexsort(-xm.T[:2, :])
        Rxm = xm[Rxm,:]

        mf = pd.read_csv( self.video_save_path+'/temp_L.csv')
        mf.columns = ['frame_num', 'x1', 'x2', 'y1', 'y2']
        xm = mf[['frame_num', 'x1', 'x2', 'y1', 'y2']]
        xm = np.array(xm)

        Lxm = np.lexsort(xm.T[:3,:])
        Lxm = xm[Lxm,:]
        during_th = self.background_during*60*self.fps
        match_set = []
        match_set_dict = {}
        Lxm_num, Rxm_num = len(Lxm), len(Rxm)
        i, j = 0,0
        match_set_num=0
        while i<Lxm_num:
            if len(match_set)>=10:
                break
            while j<Rxm_num:
                if abs(Lxm[i][0]-Rxm[j][0])<during_th:
                    match_set_dict[str(match_set_num)]=[Lxm[i][0], Rxm[j][0]]
                    [num1, num2] = [Lxm[i][0], Rxm[j][0]] if Lxm[i][0]<Rxm[j][0] else [Rxm[j][0], Lxm[i][0]]
                    match_set.append([num1, num2, match_set_num])
                    match_set_num = match_set_num+1
                    Rxm_num = Rxm_num-1
                    Rxm = np.delete(Rxm, j,axis=0)
                    j = 0
                    break
                j = j + 1
            j = 0
            Lxm_num = Lxm_num - 1
            Lxm = np.delete(Lxm, i,axis=0)
        return match_set,match_set_dict

    def match_video_and_emptybc(self,match_set):
        if len(match_set)==0:
            print('Please manually generate an empty background')
            match_set=self.man_generate()
        match_set_frame = match_set
        for i in range(len(match_set)):
            for j in range(i+1,len(match_set)):
                match_set[j][0:2] = self.overlapping(match_set[i][0:2],match_set[j][0:2])
        match_set_new = []
        for i in range(len(match_set)):
            if not match_set[i][0]==match_set[i][1]:
                match_set_new.append(match_set[i])

        match_set = np.array(match_set_new)
        frame_sort = np.lexsort(match_set.T[:1, :])
        frame_sort = match_set[frame_sort, :]

        for i in range(len(frame_sort)-1):
            if frame_sort[i][1]==frame_sort[i+1][0]:
                continue
            else:
                if frame_sort[i][2]<frame_sort[i+1][2]:
                    frame_sort[i][1] = frame_sort[i+1][0]
                else:
                    frame_sort[i+1][0] = frame_sort[i][1]
        frame_sort1 = np.lexsort(frame_sort.T[:3, :])
        frame_sort = frame_sort[frame_sort1, :]
        frame_sort1=[]
        for i in range(len(frame_sort)):
            if frame_sort[i][1]!=0:
                frame_sort1.append(frame_sort[i])
        frame_sort, frame_sort1 = frame_sort1, []
        frame_sort[0][0] = 1
        frame_sort[-1][1] = self.during * self.fps
        return match_set_frame, frame_sort


    def make_background_fig(self,match_set,frame_sort,match_set_dict):
        empty_bc_path=[]
        for i in range(len(frame_sort)):
            [num1, num2] = match_set_dict[str(int(frame_sort[i][2]))][0:2]
            video2frames(self.video_input, './frames2', extract_time_points=(num2//self.fps, num1//self.fps))
            tmp_img = Image.open('./frames2' + '/frame_000002.bmp')
            base_img = Image.open('./frames2' + '/frame_000001.bmp')

            box = (int((self.mid_x_thL + self.mid_x_thR) // 2), 0, self.width, self.height)
            region = tmp_img.crop(box)
            base_img.paste(region, box)
            base_img.save(self.empty_background_path+'/'+str(int(frame_sort[i][0]))+'.jpg')
            empty_bc_path.append(self.empty_background_path+'/'+str(int(frame_sort[i][0]))+'.jpg')
        print('Empty background generation has finished')
        return empty_bc_path

    def set_background(self):
        if self.deep_mode == 1:
            # SSD_M().ssd_detect(self.video_input, self.video_output_ssd)
            pass
        else:
            print('STEP 2 ----> YOLOV5 Detection')
            YOLO_M(self.video_save_path).yolo_detect(self.video_input)
        print('STEP 2 ----> Splice L and R')
        match_set, match_set_dict = self.match_frame()
        print('STEP 3 ----> Select Corresponding Empty Background')
        match_set, frame_sort = self.match_video_and_emptybc(match_set)
        self.empty_bc_path = self.make_background_fig(match_set,frame_sort,match_set_dict)

    def man_generate(self):
        empty_bc_path = []
        return empty_bc_path

    def overlapping(self,m,n):
        xs,xe=m[0],m[1]
        ys,ye=n[0],n[1]
        if xs==0 and xe==0:return [0,0]
        if ys>=xs and ye<=xe:res = [0,0]
        elif ys>=xs and ye>=xe:res = [xe,ye]
        elif ys<=xs and ye<=xe:res = [0,0]
        else:res = n
        return res

    def bc_grid(self,wide_grid_num=8,height_grid_num=8):
        if wide_grid_num==0 or height_grid_num==0:
            w_grid, h_grid = [[0, self.width]], [[0, self.height]]
        else:
            w_grid, h_grid = [], []
            w=self.width//wide_grid_num
            h=self.height//height_grid_num
            s, e = 0, 0
            for i in range(wide_grid_num):
                e=self.width if abs(e-self.width)<w else e+w
                w_grid.append([s,e])
                s = s+w
            s, e = 0, 0
            for i in range(height_grid_num):
                e = self.height if abs(e-self.height)<h else e+w
                h_grid.append([s,e])
                s = s+w
        self.w_grid, self.h_grid = w_grid, h_grid

    def create_dataset(self,img,bbox,name,area_count,flag):
        def resize_image(img, max_l,name,bbox,root_path):
            img_name = root_path + str(self.save_frame_img_i)+ name + '.jpg'
            if name=='_label':
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (128, 255, 128), 2)
            w = self.width
            h = self.height
            if w > h:
                scale = float(max_l) / float(w)
                w = max_l
                h = int(h * scale)
            else:
                scale = float(max_l) / float(h)
                h = max_l
                w = int(w * scale)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            img_w_h = [w,h]
            cv2.imwrite(img_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return scale, img_w_h
        def save_txt(bbox,scale,img_w_h,root_path):
            txt_name = root_path + str(self.save_frame_img_i) + '.txt'
            class_id = self.class_id

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # round(x, 6) 这里我设置了6位有效数字，可根据实际情况更改
            center_x = round(((xmin + xmax) / 2.0) * scale / float(img_w_h[0]), 6)
            center_y = round(((ymin + ymax) / 2.0) * scale / float(img_w_h[1]), 6)
            box_w = round(float(xmax - xmin) * scale / float(img_w_h[0]), 6)
            box_h = round(float(ymax - ymin) * scale / float(img_w_h[1]), 6)

            file_txt = open(txt_name, mode='a', encoding='utf-8')
            file_txt.write(str(class_id))
            file_txt.write(' ')
            file_txt.write(str(center_x))
            file_txt.write(' ')
            file_txt.write(str(center_y))
            file_txt.write(' ')
            file_txt.write(str(box_w))
            file_txt.write(' ')
            file_txt.write(str(box_h))
            file_txt.write('\n')
            file_txt.close()

        if not os.path.exists('./data/images/train/top/'):
            os.makedirs('./data/images/train/top/')
        if not os.path.exists('./data/labels/train/top/'):
            os.makedirs('./data/labels/train/top/')
        if not os.path.exists('./data/images/train/down/'):
            os.makedirs('./data/images/train/down/')
        if not os.path.exists('./data/labels/train/down/'):
            os.makedirs('./data/labels/train/down/')

        if not os.path.exists('./data/images/train/label/'):
            os.makedirs('./data/images/train/label/')

        self.save_frame_img_i += 1


        if area_count<1000 or flag==1:
            if self.static<5:
                self.static+=1
            else:
                return
        else:
            self.static=0

        if int(bbox[3])<self.mid_y_thT:
            img_root_path = './data/images/train/top/'
            txt_root_path = './data/labels/train/top/'
        else:
            img_root_path = './data/images/train/down/'
            txt_root_path = './data/labels/train/down/'
        scale, img_w_h = resize_image(img, 640, '',None,img_root_path)
        save_txt(bbox, scale, img_w_h, str(name),txt_root_path)
        scale, img_w_h = resize_image(img, 300, '_label', bbox,'./data/images/train/label/')


    def background_subtraction_track(self,image,image_bc,last_image,env,flag=0):
        currentframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cur_last_diff = cv2.absdiff(currentframe, last_image)
        last_image = currentframe
        cur_bc_diff = cv2.absdiff(currentframe, image_bc)
        median = cv2.medianBlur(cur_last_diff, 3)
        th1 = cv2.threshold(median.copy(), 80, 255, cv2.THRESH_TOZERO_INV)[1]
        th1 = cv2.threshold(th1.copy(), 20, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
        dilated_adjust = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        dilated_adjust_diff = self.normal_image_process(dilated_adjust)
        median = cv2.medianBlur(cur_bc_diff, 3)
        th_two = 18 if env == 'light' else 15
        th1 = cv2.threshold(median.copy(), th_two , 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        dilated_adjust_bc = self.normal_image_process(dilated)
        uninterest = self.uninterest
        for bvi in range(len(self.bv)):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            dilated_adjust_bc[self.bv[bvi][0]:self.bv[bvi][1], self.bv[bvi][2]:self.bv[bvi][3]] = cv2.morphologyEx(
                dilated_adjust_bc[self.bv[bvi][0]:self.bv[bvi][1], self.bv[bvi][2]:self.bv[bvi][3]], cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            dilated_adjust_bc[self.bv[bvi][0]:self.bv[bvi][1], self.bv[bvi][2]:self.bv[bvi][3]] = cv2.morphologyEx(
                dilated_adjust_bc[self.bv[bvi][0]:self.bv[bvi][1], self.bv[bvi][2]:self.bv[bvi][3]], cv2.MORPH_OPEN, kernel)
        g = dilated_adjust_diff[:, :] == 255
        area_count_diff = len(dilated_adjust_diff[g])
        if area_count_diff<1000:
            dilated_adjust = cv2.bitwise_or(dilated_adjust_diff, dilated_adjust_bc)
        else:
            dilated_adjust = dilated_adjust_bc
        for ui in range(len(uninterest)):
            dilated_adjust[uninterest[ui][0]:uninterest[ui][1], uninterest[ui][2]:uninterest[ui][3]] = 0
        if env == 'light': dilated_adjust = self.image_zone_delete(dilated_adjust)
        col, row = dilated_adjust.sum(0), dilated_adjust.sum(1)
        xcol, yrow = np.nonzero(col)[0], np.nonzero(row)[0]
        if xcol != [] and yrow != [] and area_count_diff>100:
            x1, x2 = np.nanmin(xcol), np.nanmax(xcol)
            y1, y2 = np.nanmin(yrow), np.nanmax(yrow)
            if env == 'light' and self.preferences == 'top':
                [x1, y1, x2, y2],flag = self.short_increase_delete(dilated_adjust, self.last_data, [x1, y1, x2, y2])
            else:
                flag = 0
            if (y2-y1)*(x2-x1) < self.init_box_size * 0.4 and y2 < self.mid_y_thT:
                if self.R_L_BOX_FLAG == 1:
                    if x1 > self.mid_x_thR: [x1, y1, x2, y2] = self.R_box
                    if x2 < self.mid_x_thL: [x1, y1, x2, y2] = self.L_box
                else:
                    [x1, y1, x2, y2] = [x1, y1, x2, y2]
            if (y2-y1>self.height*0.7 and x2-x1>self.width*0.7) or (y2-y1<self.height*0.1 and x2-x1<self.width*0.1):
                [x1, y1, x2, y2] = self.last_data
            self.last_data = [x1, y1, x2, y2]
        else:
            [x1, y1, x2, y2] = self.last_data
            flag = 1

        if self.create_dataset_flag: self.create_dataset(image, [x1, y1, x2, y2], self.frame_num,area_count_diff,flag)
        image_out = image
        cv2.rectangle(image_out, (int(x1), int(y1)), (int(x2), int(y2)), (128, 255, 128), 2)
        cv2.putText(image_out, self.video_txt, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2)

        if self.frame_num == 1:
            with open( self.output_path_all+'/'+self.video_name+'.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                headers = ['frame', 'x1', 'x2', 'y1', 'y2']
                f_csv.writerow(headers)
        with open( self.output_path_all+'/'+self.video_name+'.csv', 'a', newline='') as f:
            f_csv = csv.writer(f)
            row_data = [self.frame_num, x1, x2, y1, y2]
            f_csv.writerow(row_data)

        return image_out,last_image

    def pre_bc_video(self, image):
        image, self.last_image = self.background_subtraction_track(image, self.image_bc, self.last_image, self.env)
        self.frame_num = self.frame_num + 1
        if self.bc_num == self.frame_num:
            self.image_bc = cv2.cvtColor(np.array(Image.open(self.empty_bc_path[self.path_num])), cv2.COLOR_BGR2GRAY)
            self.path_num = self.path_num + 1
            if self.path_num < len(self.empty_bc_path):
                print(self.bc_num, self.empty_bc_path[self.path_num])
                self.bc_num = int(
                    int(self.empty_bc_path[self.path_num].split('/')[-1].split('.')[0]) / self.fps * self.out_put_fps)
        return image
    def bc_track(self):
        self.frame_num, self.last_area_count = 0, 0
        self.last_data = self.init_box
        self.image_bc = cv2.cvtColor(np.array(Image.open(self.empty_bc_path[0])), cv2.COLOR_BGR2GRAY)
        self.last_image = self.image_bc
        self.path_num = 0

        self.video_txt = ''
        self.bc_num = int(self.empty_bc_path[self.path_num].split('/')[-1].split('.')[0])
        self.static = 0

        clip1 = VideoFileClip(self.video_input)
        video_clip1 = clip1.fl_image(self.pre_bc_video).set_fps(self.out_put_fps)
        video_clip1.write_videofile( self.output_path_all+'/'+self.video_name+'.mp4', codec='libx264', audio=False)

    def monkeytrail_process(self):
        for video_input_name in glob.glob(self.video_path_all + '/*.mp4'):
            self.video_input = video_input_name
            self.video_name = Path(self.video_input).name.split('.')[0]
            self.video_save_path = './temp/' + self.video_name
            if not os.path.exists(self.video_save_path):
                os.mkdir(self.video_save_path)
            print('STEP 1 ----> Video Preprocessing')
            self.pre_processing()
            print('STEP 2 ----> FDM Tracking')
            self.frame_diff_set()
            self.set_background()
            self.bc_grid()
            print('STEP 3 and 4 ----> Image Processing and Trajectory Extraction')
            self.bc_track()
        shutil.rmtree('./temp/')
        shutil.rmtree('./frames2/')
        shutil.rmtree('./pro/runs/')