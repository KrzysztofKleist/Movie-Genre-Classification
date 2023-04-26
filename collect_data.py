from utilities import extract_frame, color_check, video_available, check_folder, create_dir

from multiprocessing import Pool
import pandas as pd
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def multi_run_wrapper(args):
    return download_data(*args)


final_df = pd.read_csv(r'files\movieclips_final.tsv', sep="\t")
# final_df = final_df.sample(frac=1).reset_index(drop=True)


def download_data(start_range, end_range):
    
    print("####################################################################################")
    print("Current step:", start_range)
    print("####################################################################################")
    
    # loop through each row in the dataframe
    for index, row in final_df.iloc[start_range:end_range].iterrows():
        if check_folder(row['videoid'], row['genre'], r'D:\frames') == True:
            if video_available(row['videoid']) == True and color_check(row['videoid']) == True:
                extract_frame(row['videoid'], row['genre'],
                              2, 20, r'D:\frames')


if __name__ == '__main__':
    
    create_dir(r'D:\frames')

    start = time.time()
    n = len(final_df)
    step = 50
    intervals = [(i, i+step) for i in range(0, n, step) if i+step <= n]
    last_interval = [(n-(n % step), n)]
    intervals = intervals + last_interval
    # print(intervals)

    pool = Pool(10)

    pool.map(multi_run_wrapper, intervals)

    pool.close()
    pool.join()

    end = time.time()
    print('time:', end - start)
