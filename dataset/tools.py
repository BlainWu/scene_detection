import os

def create_dirs():
    traing_dirs = os.listdir('./train')
    new_dir = './extern'
    for dir in traing_dirs:
        try:
            os.mkdir(os.path.join(new_dir,dir))
        except FileNotFoundError:
            os.mkdir(new_dir)

if __name__ == '__main__':
    create_dirs()