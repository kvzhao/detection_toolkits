
import os
import sys

from shutil import copyfile

from os.path import join, basename

def get_files(path, ext=('.png', '.jpg')):
    # ext is str or tuple
    files = []
    for (dir_path, _, file_names) in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(ext):
                files.append(join(dir_path, file_name))
    return files

def main(args):

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    file_paths = get_files(args.input_dir)
    print(file_paths[0])
    print(len(file_paths))

    for i, src_path in enumerate(file_paths):

        if args.output_dir is not None:
            copyfile(src_path, join(args.output_dir, str(i).zfill(8) + '.jpg'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-id', '--input_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)