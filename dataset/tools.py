import os
from tqdm import tqdm

def create_dirs(new_dir = './extern'):
    traing_dirs = os.listdir('./train')
    for dir in traing_dirs:
        try:
            os.mkdir(os.path.join(new_dir,dir))
        except FileNotFoundError:
            os.mkdir(new_dir)

'''Rename the index in a dir'''
def re_index(dir,start_index = 0):
    pics_list = os.listdir(dir)
    #pics_list.sort(key = lambda x : int(x.split('.')[0]))
    new_index = start_index
    for pic in tqdm(pics_list):
        pic_path = os.path.join(dir,pic)
        os.rename(pic_path,os.path.join(dir,'{0}.jpg'.format(new_index)))
        new_index += 1

'''Rename a dataset'''
def re_index_dataset(data_dir,start_index = 20000):
    class_dir_list = os.listdir(data_dir)
    for cls in class_dir_list:
        cls_path = os.path.join(data_dir,cls)
        re_index(cls_path,start_index)

def get_tmp_index(dir):
    file_list = os.listdir(dir)
    index = []
    if len(file_list)==0:
        tmp_index = 0
    else:
        for i,name in enumerate(file_list):
            index.append(int(name.split('.')[0]))
        index.sort()
        tmp_index = index[-1] + 1
    return tmp_index

if __name__ == '__main__':
    """合并两个数据集A和B，可以先以一个很大的start_index重命名index数据集B，然后复制粘贴合并；再re_index最终的数据集。"""
    re_index_dataset(data_dir = './train_extern',start_index = 0)