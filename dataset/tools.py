import os

def create_dirs():
    traing_dirs = os.listdir('./train')
    for dir in traing_dirs:
        os.mkdir(os.path.join('./val',dir))

if __name__ == '__main__':
    create_dirs()