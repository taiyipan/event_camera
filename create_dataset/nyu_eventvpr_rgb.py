import pandas as pd
import glob
import geopy.distance
from time import time 
import numpy as np
from time import time 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait 
from multiprocessing import cpu_count
import os 
import cv2 as cv 

'''
This program processes raw data collected by event-rgb-gps sensors
into a dataset format compatible with VPR-Bench framework. 
This program only processes RGB camera raw data into RGB dataset.
'''

# hyperparams 
threshold = 25 # unit meters; below which is a match, else false 
query_count = None 

# dir paths: change if necessary
raw_dir = 'data'
dir = 'NYU-EventVPR-RGB'

# fixed dir paths 
csv_path = os.path.join(raw_dir, 'sensor_data_*/GPS_data_*.csv')
npy_path = os.path.join(dir, 'ground_truth_new.npy')
query_dir = os.path.join(dir, 'query')
ref_dir = os.path.join(dir, 'ref')

# create appropriate directories for new dataset 
def create_dataset_dir():
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(query_dir):
        os.mkdir(query_dir)
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)

# map image paths to indices 
def map_img_path_to_index(df: pd.DataFrame) -> dict:
    img_map = dict()
    for row in df.itertuples():
        img_map['frame_' + row.Timestamp + '.jpg'] = str(row.Index).zfill(7) + '.jpg'
    return img_map 

# move images from raw data dir to processed data dir (same indexing as loaded dataframe)
def move_imgs(df: pd.DataFrame):
    img_map = map_img_path_to_index(df)
    folder_path = glob.glob('{}/sensor_data_*/img_*'.format(raw_dir))
    print(folder_path)
    with ThreadPoolExecutor() as executor:
        print('Started Thread Pool Executor...')
        futures = list()
        for folder in folder_path:
            for img_path in os.listdir(folder):
                futures.append(
                     executor.submit(
                        img_io_thread,
                        os.path.join(folder, img_path),
                        os.path.join(query_dir, img_map.get(img_path)),
                        os.path.join(ref_dir, img_map.get(img_path))
                    )
                )
        done, not_done = wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        print('All concurrent futures have completed')
    print('Executor has shutdown')
    
def img_io_thread(input_path, output_path1, output_path2):
    try:
        img = cv.imread(input_path)
        cv.imwrite(output_path1, img)
        cv.imwrite(output_path2, img)
        print(output_path1)
    except:
        raise FileNotFoundError


# calculate distance in meters between 2 GPS coordinates 
# input: a(latitude, longitude), b(latitude, longitude)
# output: distance in meters 
def calculate_gps_distance(a: tuple, b: tuple) -> float:
    return geopy.distance.geodesic(a, b).meters

# aggregate all GPS data from csv files into pandas dataframe 
def load_csv() -> pd.DataFrame:
    file_path = glob.glob(csv_path)
    print('GPS file count: {}'.format(len(file_path)))
    df = pd.concat(map(pd.read_csv, file_path), ignore_index = True)
    print('Total entry count: {}'.format(df.shape[0]))
    print(df.head())
    return df

# initialize ground truth numpy array according to dataframe dims 
def create_gt(df) -> np.ndarray:
    gt = np.zeros((df.shape[0], 2), dtype=object)
    print(gt.shape)
    for i in range(gt.shape[0]):
        gt[i][0] = i 
        gt[i][1] = list()
    return gt 

# subprocess to be handled by an assigned CPU thread
# upon completion, return partial array, start and end indices 
def match_process(df: pd.DataFrame, gt: np.ndarray, start: int, end: int):
    for row in df.iloc[start:end].itertuples():
        print('{}/{}'.format(row.Index, df.shape[0] - 1))
        gt[row.Index][0] = row.Index
        for r in df.itertuples():
            dist = calculate_gps_distance(
                (row.Latitude, row.Longitude),
                (r.Latitude, r.Longitude)
            )
            if dist < threshold:
                gt[row.Index][1].append(r.Index)
    return gt, start, end

# generate ground truth array between query and reference images (multiprocessing)
def compute_ground_truth(df: pd.DataFrame) -> np.ndarray:
    # initialize ground truth numpy array
    gt = create_gt(df)
    
    # define multiprocessing program executor
    print('CPU core count: {}'.format(cpu_count()))
    executor = ProcessPoolExecutor()
    print('Started Process Pool Executor...')

    # divide task into equal chunks among all CPU threads 
    futures = list()
    assert df.shape[0] >= 100, 'Too few discrete tasks for {} threads'.format(cpu_count())
    prev = 0
    for idx in range(df.shape[0] // cpu_count(), df.shape[0], df.shape[0] // cpu_count()):
        futures.append(executor.submit(match_process, df, gt, prev, idx))
        prev = idx
    done, not_done = wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    print('All concurrent futures have completed')

    # merge resulting arrays based on index range
    for future in futures:
        gt[future.result()[1]:future.result()[2], :] = future.result()[0][future.result()[1]:future.result()[2], :]
    print('Concurrent futures merged')

    # shutdown executor
    executor.shutdown()
    print('Executor has shutdown')

    # iterate over remaining rows 
    match_process(df, gt, prev, df.shape[0])
    print('Remaining rows iterated')

    return gt

# aggregate raw dataset and reformat into unified dataset template format
def main():
    start = time()
    #######################################################################
    create_dataset_dir()
    df = load_csv()
    gt = compute_ground_truth(df)
    np.save(npy_path, gt)
    move_imgs(df)
    #######################################################################
    end = time()
    print('Time elapsed: {:.6f} hours'.format((end - start) / 3600.0))
    print('Time elapsed: {:.6f} minutes'.format((end - start) / 60.0))
    print('Time elapsed: {:.6f} seconds'.format((end - start) / 1.0))

if __name__ == '__main__':
    main()