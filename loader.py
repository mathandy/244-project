import os


def get_file_info(filename):
    with open(filename, 'r') as f:
        file_info = []
        for line in f.read().split('\n'):
            if not line.strip():
                continue
            fn1, junk, label = line.strip().split(',')
            file_info.append((fn1.strip(), junk.strip(), label.strip()))
    return file_info