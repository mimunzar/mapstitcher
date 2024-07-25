def parse_command_line():
    from sys      import argv
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device',
        required = True,
        choices  = ['cpu', 'cuda'])
    parser.add_argument('--max-size',
        required = True,
        type     = int)
    parser.add_argument('--debug',
        required = False,
        action   = 'store_true',
        help     = 'Run in debug mode')
    return vars(parser.parse_args(argv[1:])) # Anything but program name


def main():
    args = parse_command_line()
    print(args)


if __name__ == '__main__':
    main()
