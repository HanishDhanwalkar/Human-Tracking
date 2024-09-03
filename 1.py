import argparse

# Define default values
parser = argparse.ArgumentParser()
parser.add_argument('--foo', default=1, type=float, help='foo')
parser.add_argument('--bar', default=1, type=float, help='bar')

# Get the args container with default values
if __name__ == '__main__':
    args = parser.parse_args()  # get arguments from command line

    print(args.foo, args.bar)