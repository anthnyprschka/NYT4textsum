# coding=utf-8

"""
Convert NYT to TextSum model data.

"""

import glob
import struct
import sys
import random

import xml.etree.ElementTree as ET
import nltk
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import tensorflow as tf
from tensorflow.core.example import example_pb2


def _extract_xml_file(filepath):
    """
    For every '.xml' file in the nyt_corpus/data directory
    """
    abstract = None
    full_text = None
    tree = ET.parse(filepath)
    for child in tree.iter():
        if 'abstract' in child.tag:
            abstract = ' '.join([x.text for x in child.iter() if x.text is not None])
            abstract = ' '.join(abstract.split())
        if 'full_text' in child.attrib.values():
            full_text = ' '.join([x.text for x in child.iter() if x.text is not None])
            full_text = ' '.join(full_text.split())
        # if 'lead_paragraph' in child.attrib.values():
        #     lead_paragraph = ' '.join([x.text for x in child.iter() if x.text is not None])
        #     lead_paragraph = ' '.join(lead_paragraph.split())
    return abstract, full_text #, lead_paragraph


def _preprocess_text(text, isAbstract=False):
    """
    Preprocess text.
    """
    # Encode in unicode
    if isinstance(text, str):
        text = text.decode('utf-8') # , 'ignore')
    # To lower
    text = text.lower()
    # Remove (s) and (m) and ''
    text = text.replace('(s)', '').replace('(m)', '').replace("''", '')
    # To sentences
    sentences = sentence_detector.tokenize(text)

    # abstract-specific edits
    if isAbstract:
        sentences = [item.strip() for sentence in sentences for item in sentence.split(';')]
        junk = ['photo', 'graph', 'chart', 'map', 'table', 'drawing']
        if sentences[-1].replace('s', '') in junk:
            sentences.pop(-1)

    sentences_tokens = []
    tuples = []
    for x in range(0, len(sentences)):

        # Tokenize and pos-tag
        tokens = nltk.word_tokenize(sentences[x])
        tagged_tokens = nltk.pos_tag(tokens)
        tuples.append(tagged_tokens)

        # Substitute numbers
        for y in range(0,len(tuples[x])):
            tuples[x][y] = list(tuples[x][y])
            if tuples[x][y][1] == 'CD':
                tuples[x][y][0] = '0'

        sentences_tokens.append([pair[0] for pair in tuples[x]])

    return sentences_tokens


def _merge_ascii_with_padding(sentArray):
    for z in range (0, len(sentArray)):
        sentArray[z] = ' '.join(sentArray[z])
    startOfSentTag = '<s> '
    endOfSentTag = ' </s> '
    sentArray[0] = startOfSentTag + sentArray[0]
    sandwich = endOfSentTag + startOfSentTag
    sent = sandwich.join(sentArray)
    sent = sent + endOfSentTag

    # Encode unicode in ascii ignoring any non-ascii characters
    if isinstance(sent, unicode):
        sent = sent.encode('ascii', 'ignore')
    return sent


def _nyt_to_binary():

    # Get all the filepaths (1.8m)
    incoming_filepaths = glob.glob('./*/*/*/*.xml')
    print(len(incoming_filepaths))

    # incoming_filepaths = ['1815742.xml', '1815742.xml']

    random.shuffle(incoming_filepaths)
    sample_count = 0

    for x in range(0, len(incoming_filepaths)):

        #print(incoming_filepaths[x])

        # Extract abstract and full_text from xml file
        abstract, full_text = _extract_xml_file(incoming_filepaths[x])
        if abstract is None or full_text is None:
            # print("Required field missing")
            # print("nada")
            continue

        try:
            # Parse abstract
            processed_abstract = _preprocess_text(abstract, isAbstract=True)
            final_abstract = _merge_ascii_with_padding(processed_abstract)

            # Parse full_text
            processed_article = _preprocess_text(full_text)
            final_article = _merge_ascii_with_padding(processed_article)
        except Exception as e:
            print(incoming_filepaths[x], e)
            continue

        # Create tf.example
        tf_example = example_pb2.Example()
        
        try:
            tf_example.features.feature['abstract'].bytes_list.value.extend([final_abstract])
        #except Exception as e:
            #print(incoming_filepaths[x], e)
            #print(final_abstract)
            #if isinstance(final_abstract, unicode):
            #    print(final_abstract.encode('ascii', 'ignore'))
            #continue
            tf_example.features.feature['article'].bytes_list.value.extend([final_article])
        except Exception as e:
            print(incoming_filepaths[x], e)
            #print(final_article)
            #if isinstance(final_article, unicode):
            #    print(final_article.encode('ascii', 'ignore'))
            continue

        tf_example_str = tf_example.SerializeToString()
        #print tf_example

        str_len = len(tf_example_str)
        writer = open(incoming_filepaths[x].replace('.xml', ''), 'wb')
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()

        #print(incoming_filepaths[x])
        sample_count += 1

	if sample_count % 200 == 0:
		print(sample_count)

    print('%d samples generated' % sample_count)


def main(unused_argv):
    _nyt_to_binary()


if __name__ == '__main__':
    tf.app.run()

