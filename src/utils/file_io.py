import os
import glob
import pickle

DIR_PATH = '/'.join(os.path.dirname(__file__).split('/')[:-2])

def exist_file(file_path):
    return os.path.isfile(file_path)
    
def open_files(files_path):
    path = os.path.join(DIR_PATH, files_path)
    files = glob.glob(path)
    return files

def open_file(files_path):
    try:
        path = os.path.join(DIR_PATH, files_path)
        files = glob.glob(path)
        return files[0]
    except Exception as e:
        print(f'File({files_path}) does not exist! Error: {e}')
        raise

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file(name, type):
    try:
        if type == 'txt':
            with open(name, encoding="latin-1") as f:
                lines = ''.join(f.readlines()).strip().split('\n')
                words = [line.lower().strip() for line in lines]
                return words
        if type == 'obj':
            with open(name, 'rb') as f:
                obj = pickle.load(f)
                return obj
        else:
            raise Exception
    except Exception as e:
        print(f"An error occured when trying to read text file! Error message: {e}")
        raise

def save_file(path, content):
    # make_dir()
    print('/'.join(path.split('/')[:-1]))
    filehandler = open(path, "wb")
    pickle.dump(content, filehandler)
    filehandler.close()
