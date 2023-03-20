import os
import sys
import pafy
import cv2
import youtube_dl
import numpy as np


def create_dir(directory):
    # checking if the destination directory exists or not
    if not os.path.exists(directory):
        # if the destination directory is not present then create it
        os.makedirs(directory)


def list_to_string(genre):
    # creating a string for the genres
    return '_'.join(genre.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(","))


def video_available(link):
    # checking if video is available
    url = 'https://www.youtube.com/watch?v=' + link

    try:
        pafy.new(url)
        return True
    except OSError:
        return False
    except:
        return False


def gray_scale_check(link):
    url = 'https://www.youtube.com/watch?v=' + link
    
    try:
        vPafy = pafy.new(url)
        play = vPafy.getbestvideo(preftype='webm')

        video = cv2.VideoCapture(play.url)

        fps = int(video.get(cv2.CAP_PROP_FPS))

        # vid_len = vPafy.length

        # we want to extract one frame from the 2 second of the video
        frame_num = int(fps * 2)

        # print('frame to extract:', frame_num)

        current_frame = 0

        while(True):
            # print(current_frame)
            # read frame
            ret, frame = video.read()

            if ret:
                if current_frame == frame_num:
                    # cv2.imshow('frame', frame)

                    # splitting b, g, r channels
                    b, g, r = cv2.split(frame)

                    # getting differences between (b,g), (r,g), (b,r) channel pixels
                    r_g = np.count_nonzero(abs(r-g))
                    r_b = np.count_nonzero(abs(r-b))
                    g_b = np.count_nonzero(abs(g-b))

                    # sum of differences
                    diff_sum = float(r_g+r_b+g_b)

                    # finding ratio of diff_sum with respect to size of image
                    ratio = diff_sum/frame.size

                    if ratio > 0.005:
                        # print("image is color")
                        ret_val = True
                    else:
                        # print("image is greyscale")
                        ret_val = False

                    break
                    # current_frame += 1
                else:
                    current_frame += 1
            else:
                ret_val = False
                break

            # if current_frame > frame_num:
            #     break

        # release VideoCapture
        video.release()

        cv2.destroyAllWindows()
        
        return ret_val
    
    except:
        return False