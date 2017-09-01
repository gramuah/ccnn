#!/usr/bin/env python
"""Run this with ``python -m caffe2keras <args>``"""

import argparse

parser = argparse.ArgumentParser(
    description='Converts a Caffe model to a Keras model')
parser.add_argument(
    '--debug', action='store_true', default=False, help='enable debug mode')
parser.add_argument('prototxt', help='network definition path')
parser.add_argument('caffemodel', help='network weights path')
parser.add_argument('destination', help='path for output model')


def main():
    args = parser.parse_args()

    # lazy import so that we parse args before initialising TF/Theano
    from caffe2keras import convert

    print("Converting model...")
    model = convert.caffe_to_keras(args.prototxt, args.caffemodel, args.debug)
    print("Finished converting model")

    # Save converted model structure
    print("Storing model...")
    model.save(args.destination)
    print("Finished storing the converted model to " + args.destination)


if __name__ == '__main__':
    main()
