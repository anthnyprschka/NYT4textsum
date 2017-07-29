# coding=utf-8

"""
Count number of articles with required fields

"""

import glob
import struct
import sys
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.core.example import example_pb2
import random

def _count_features():

    # Get all the filepaths (1.8m)
    incoming_filepaths = glob.glob('./*/*/*/*.xml')
    print(len(incoming_filepaths))

    random.shuffle(incoming_filepaths)

    count_abstracts = 0
    count_full_texts = 0
    count_a_and_ft = 0
    count_xmls = 0

    for x in range(0, 10000): #len(incoming_filepaths)):
	print(incoming_filepaths[x])
        tree = ET.parse(incoming_filepaths[x])
	temp1 = 0
	temp2 = 0
        for child in tree.iter():
            if 'abstract' in child.tag:
                count_abstracts += 1
		temp1 = 1
            if 'full_text' in child.attrib.values():
                count_full_texts += 1
		temp2 = 1
	if temp1 == 1 and temp2 == 1:
	    count_a_and_ft += 1
        count_xmls += 1

    print('%d abstracts' % count_abstracts)
    print('%d full_texts' % count_full_texts)
    print('%d a_and_fts' % count_a_and_ft)
    print('%d xmls' % count_xmls)


def main(unused_argv):
    _count_features()


if __name__ == '__main__':
    tf.app.run()

