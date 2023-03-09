import os

import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="Downloader Arguments")

    parser.add_argument("-d", "--directory",
                        help="Tree download directory.", 
                        required=False,
                        default="dataset", type=str)

    return parser.parse_args()


def main():

	args = parse_args()

	if not os.path.isdir(args.directory):
		os.mkdir(args.directory)
  
	 	# download to that directory






if __name__ == "__main__":
    main()
