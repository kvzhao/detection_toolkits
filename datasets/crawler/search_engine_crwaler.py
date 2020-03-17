
import os
import sys

from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler


def main(args):

    if args.output_dir is None:
        raise ValueError('output dir must be assigned')

    os.makedirs(args.output_dir, exist_ok=True)

    root_dir = os.path.join(args.output_dir, args.search_keyword)

    os.makedirs(root_dir, exist_ok=True)

    crawler = None
    if args.search_engine == 'bing':
        crawler = BingImageCrawler(
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=10,
            storage={'root_dir': root_dir})

    crawler.crawl(keyword=args.search_keyword,
                  filters=None, offset=0,
                  max_num=args.number_of_image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-n', '--number_of_image', type=int, default=1000)
    parser.add_argument('-k', '--search_keyword', type=str, default=None)
    parser.add_argument('-e', '--search_engine', type=str, default='bing', help='baidu, google, bing')
    args = parser.parse_args()
    main(args)