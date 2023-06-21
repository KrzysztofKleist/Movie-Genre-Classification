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


def check_folder(link, genre, folder_path):

    genre_str = genre.replace("'", "").replace(
        "[", "").replace("]", "").replace(" ", "").replace(",", "_")

    if os.path.exists(folder_path + "/" + genre_str + "/" + link):
        # if folder exists return False
        print(folder_path + "/" + genre_str + "/" + link + " exists")
        return False
    else:
        # if folder DOESN'T exist return True
        print(folder_path + "/" + genre_str + "/" + link + " DOESN'T exists")
        return True


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


def color_check(link):
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

        while (True):
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


        # release VideoCapture
        video.release()

        cv2.destroyAllWindows()

        return ret_val

    except:
        return False


def extract_frame(link, genre, start_frame, interval, folder_path):

    genre_str = genre.replace("'", "").replace(
        "[", "").replace("]", "").replace(" ", "").replace(",", "_")

    if os.path.exists(folder_path + "/" + genre_str + "/" + link):
        print(folder_path + "/" + genre_str + "/" + link + " exists")
        return
    else:
        create_dir(folder_path + "/" + genre_str + "/" + link)
        print("Creating folder..." + folder_path + "/" + genre_str)

        url = 'https://www.youtube.com/watch?v=' + link

        vPafy = pafy.new(url)
        play = vPafy.getbestvideo(preftype='webm')

        video = cv2.VideoCapture(play.url)

        fps = int(video.get(cv2.CAP_PROP_FPS))

        vid_len = vPafy.length

        # we want to get the first frame at start_frame seconds
        # and then each frame after the next interval seconds
        start_value = start_frame * fps
        interval_fps = interval * fps
        # we stop at the 3/4 of the video because of the fact
        # that videos usually contain some ending credits
        stop_value = int(3/4 * fps * vid_len)

        frame_numbers = [start_value]

        while frame_numbers[-1] + interval_fps < stop_value:
            frame_numbers.append(frame_numbers[-1] + interval_fps)

        current_frame = 0

        while (True):
            # read frame
            ret, frame = video.read()

            if ret:
                if current_frame in frame_numbers:
                    name = folder_path + "/" + genre_str +\
                        "/" + link + "/" + str(current_frame) + '.jpg'

                    if os.path.isfile(name):
                        print(name + "exists")
                    else:
                        print("Creating..." + name)
                        cv2.imwrite(name, frame)

                    # shutil.move(name, dest)
                    current_frame += 1
                else:
                    current_frame += 1
            else:
                break

            if current_frame > frame_numbers[-1]:
                break

        # release VideoCapture
        video.release()

        cv2.destroyAllWindows()
