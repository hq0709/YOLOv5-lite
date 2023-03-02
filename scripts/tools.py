import os
import numpy as np
import sys


def get_files(rootDir, formats='jpg png', include_dirs='', exclude_dirs=''):
    list_dirs = os.walk(rootDir, followlinks=True)
    file_paths = []
    file_names = []
    exc_dirs = exclude_dirs.split(' ')
    for root, dirs, files in list_dirs:
        flag = False
        cur_dir = root.split('\\')[-3]
        # print(cur_dir)
        # return
        if include_dirs != '' and cur_dir not in include_dirs:
            continue
        for exc_dir in exc_dirs:
            if exc_dir != '' and root.find(exc_dir) != -1:
                flag = True
                break
        if flag:
            continue
        for f in files:
            if f.split('.')[-1] in formats.split(' '):
                file_paths.append(os.path.join(root, f))
                file_names.append(f)
    return file_paths, file_names

if __name__ == '__main__':
    get_files("/nas/users/hsl/hfywork/data/srclink/headpose_gaze/2021-01-18",include_dirs="location_2337 location_2336 location_2335 location_2334 location_2333 location_2332 location_2331 location_2338")
