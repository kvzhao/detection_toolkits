import os
import cv2

def main(args):
    if args.video is None:
        raise ValueError('input video is not given')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    vidcap = cv2.VideoCapture(args.video)
    success, image = vidcap.read()
    count = 0
    while success:
        #if count % 5 == 0:
        cv2.imwrite("{}/frame_{count:08}.png".format(args.output_dir, count=count), image)     # save frame as JPEG file      
        success, image = vidcap.read()
        count +=  1
        if count % 100 == 0:
            print('Processed {} images'.format(count))
    print('Done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-v', '--video', type=str, default=None, help='Path to the video file')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='Path to the video file')

    args = parser.parse_args()
    main(args)