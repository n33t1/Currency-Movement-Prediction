import os
import glob
import pickle

DIR_PATH = '/'.join(os.path.dirname(__file__).split('/')[:-2])

def open_files(files_path):
    path = os.path.join(DIR_PATH, files_path)
    files = glob.glob(path)
    return files

def read_file(name, type):
    try:
        if type == 'txt':
            with open(name, encoding="latin-1") as f:
                lines = ''.join(f.readlines()).strip().split('\n')
                words = [line.lower() for line in lines]
                return words
        if type == 'obj':
            with open(name, 'rb') as f:
                obj = pickle.load(f)
                return obj
        else:
            raise Exception
    except Exception as e:
        print(e)
        print("An error occured when trying to read text file! Error message: {}".format(e.message))
        raise

def save_file(filename, content):
    filehandler = open(filename, "wb")
    pickle.dump(content, filehandler)
    filehandler.close()
