import sys
import argparse
import numpy as np

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
