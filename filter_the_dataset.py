from utilities import video_available
from multiprocessing import Pool
import pandas as pd
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def multi_run_wrapper(args):
    return remove_unavailable(*args)


all_df = pd.read_csv(r'files\movieclips_all.tsv', sep="\t")
all_df = all_df.sample(frac=1).reset_index(drop=True)


def remove_unavailable(start_range, end_range):

    df = pd.DataFrame(columns=all_df.columns)

    # loop through each row in the dataframe
    for index, row in all_df.iloc[start_range:end_range].iterrows():
        if video_available(row['videoid']) == True:
            df = df.append(row, ignore_index=True)

    return df


if __name__ == '__main__':

    start = time.time()

    n = len(all_df)
    step = 50
    intervals = [(i, i+step) for i in range(0, n, step) if i+step <= n]
    last_interval = [(n-(n % step), n)]
    intervals = intervals + last_interval
    print(intervals)

    pool = Pool()

    results = pool.map(multi_run_wrapper, intervals)

    pool.close()
    pool.join()

    print(results)

    df = pd.DataFrame(columns=all_df.columns)

    for part_df in results:
        df = df.append(part_df, ignore_index=True)

    df.to_csv(r'files\movieclips_available_only.tsv', sep="\t")

    end = time.time()
    print('time:', end - start)
