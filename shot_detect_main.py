#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import sys


class Frame:
   
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.diff == other.diff

    def __ne__(self, other):
        return not self.__eq__(other)

    def getMAXdiff(self, frame):
        temp=frame[0]
        for i in range(len(frame)):
            if(temp.diff <= frame[i].diff):
                temp=frame[i]
                
        return temp

    def find_possible_frame(self, list_frames):
       
        possible_frame = []
        window_frame = []
        window_size = 30                     
        m_suddenJudge = 3
        m_MinLengthOfShot = 8
        start_id_spot,end_id_spot = [0],[]
        
      

        length = len(list_frames)
        index = 0
        while(index < length):
            frame_item = list_frames[index]
            window_frame.append(frame_item)
            if len(window_frame) < window_size:
                index += 1
                if index == length-1:
                    window_frame.append(list_frames[index])
                else:
                    continue

            max_diff_frame = self.getMAXdiff(window_frame)
            max_diff_id = max_diff_frame.id

            if len(possible_frame) == 0:
                possible_frame.append(max_diff_frame)
              
            last_max_frame = possible_frame[-1]

            sum_start_id = last_max_frame.id + 1
            sum_end_id = max_diff_id - 1


            id_no = sum_start_id
            sum_diff = 0
            
            while True:

                sum_frame_item = list_frames[id_no]
                sum_diff += sum_frame_item.diff
                id_no += 1
                if id_no > sum_end_id:
                    break

            average_diff = sum_diff / (sum_end_id - sum_start_id + 1)
            if max_diff_frame.diff >= (m_suddenJudge * average_diff):
                possible_frame.append(max_diff_frame)
                window_frame = []
                index = possible_frame[-1].id + m_MinLengthOfShot
                continue
            else:
                index = max_diff_frame.id + 1
                window_frame = []
                continue

        for i in range(len(possible_frame)):
            start_id_spot.append(possible_frame[i].id)
            end_id_spot.append(possible_frame[i].id - 1)


        sus_last_frame = possible_frame[-1]
        last_frame = list_frames[-1]
        if sus_last_frame.id < last_frame.id:
            possible_frame.append(last_frame)
            end_id_spot.append(possible_frame[-1].id)

        return possible_frame, start_id_spot, end_id_spot

    


def main():
    videopath = ""
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
    FRAME = Frame(0, 0)
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        

        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        elif curr_frame is not None and prev_frame is None:
            diff_sum_mean = 0
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)

        prev_frame = curr_frame
        i = i + 1
        success, frame = cap.read()
    cap.release()

    frame_return, start_id_spot_old, end_id_spot_old = FRAME.find_possible_frame(frames)

    new_frame, start_id_spot, end_id_spot = FRAME.optimize_frame(frame_return, frames)

    start = np.array(start_id_spot)[np.newaxis, :]
    end = np.array(end_id_spot)[np.newaxis, :]
    return start,end
#     spot = np.concatenate((start.T, end.T), axis=1)
#     np.savetxt('./result.txt', spot, fmt='%d', delimiter='\t')


 def optimize_frame(self, tag_frames, list_frames):
     '''
         optimize the possible frame
     '''
     new_tag_frames = []
     frame_count = 10
     diff_threshold = 10
     diff_optimize = 2
     start_id_spot = []
     start_id_spot.append(0)
     end_id_spot = []

     for tag_frame in tag_frames:

         tag_id = tag_frame.id

         """

         check whether the difference of the possible frame is no less than 10.

         """
         if tag_frame.diff < diff_threshold:
             continue
         """

         check whether the difference is more than twice the average difference of 
         the previous 10 frames and the subsequent 10 frames.

         """
         #get the previous 10 frames
         pre_start_id = tag_id - frame_count
         pre_end_id = tag_id - 1
         if pre_start_id < 0:
             continue

         pre_sum_diff = 0
         check_id = pre_start_id
         while True:
             pre_frame_info = list_frames[check_id]
             pre_sum_diff += pre_frame_info.diff
             check_id += 1
             if check_id > pre_end_id:
                 break

         #get the subsequent 10 frames
         back_start_id = tag_id + 1
         back_end_id = tag_id + frame_count
         if back_end_id >= len(list_frames):
             continue

         back_sum_diff = 0
         check_id = back_start_id
         while True:
             back_frame_info = list_frames[check_id]
             back_sum_diff += back_frame_info.diff
             check_id += 1
             if check_id > back_end_id:
                 break

         # calculate the difference of the previous 10 frames and the subsequent 10 frames
         sum_diff = pre_sum_diff + back_sum_diff
         average_diff = sum_diff / (frame_count * 2)

         #check whether the requirement is met or not
         if tag_frame.diff > (diff_optimize * average_diff):
             new_tag_frames.append(tag_frame)

     """
     get the index of the first and last frame of a shot
     """

     for i in range(0,len(new_tag_frames)):
         start_id_spot.append(new_tag_frames[i].id)
         end_id_spot.append(new_tag_frames[i].id - 1)


     last_frame = list_frames[-1]
     if new_tag_frames[-1].id < last_frame.id:
         new_tag_frames.append(last_frame)

     end_id_spot.append(new_tag_frames[-1].id)


     return new_tag_frames, start_id_spot, end_id_spot



