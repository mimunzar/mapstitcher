import sys
import configparser
import argparse
import os
import cv2
import glob
import math
import numpy as np
import torch

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', help="bar")
    args = parser.parse_args()
    return args

def main():
    args = parse()
    print(args.foo)

if __name__ == '__main__':
    main()
