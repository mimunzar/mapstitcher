def parse_command_line():
    from sys      import argv
    from argparse import ArgumentParser
    from libs import make_homography_with_downscaling


    parser = ArgumentParser()
    parser.add_argument('--debug', required = False, action   = 'store_true', help     = 'Run in debug mode')
    parser.add_argument('--input', required = True, type     = str, help     = 'Input image list')
    parser.add_argument('--output', required = False, type    = str, default='result.png', help     = 'Output image')
    parser.add_argument('--max-matches', required = False, type = int, default=200, help     = 'Maximum number of matches per pair')
    return vars(parser.parse_args(argv[1:])) # Anything but program name

def parse_list_file(file_path):
    images = {}
    homographies = []
    pairs = []

    section = None  # Tracks the current section (- images, - homographies, - pairs)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("-"):
                section = line[2:].strip().lower()
                continue

            if section == "images":
                # Parse images (format: index path)
                index, path = line.split(maxsplit=1)
                images[int(index)] = path

            elif section == "homographies":
                # Parse homographies (format: two indices)
                indices = list(map(int, line.split()))
                if len(indices) == 2:
                    homographies.append(tuple(indices))

            elif section == "pairs":
                # Parse pairs (format: two indices)
                indices = list(map(int, line.split()))
                if len(indices) == 2:
                    pairs.append(tuple(indices))

    return images, homographies, pairs

def main():
    args = parse_command_line()
    print(args)
    images, homographies, pairs = parse_list_file(args["input"])

    homography = make_homography_with_downscaling(**{  # Expects following argparse arguments.
        'max_size': 128, # Homography is computed on images of this size
        'device'  : 'cuda',
        'debug'   : True})

if __name__ == '__main__':
    main()
