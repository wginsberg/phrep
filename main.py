#!/usr/bin/env python
import logging
logging.basicConfig()
logger = logging.getLogger('phrep')

import argparse
import sqlite3
import sys
import os
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
sqlite3.register_adapter(np.float32, float)


def verify_db_init():

    create_label = '''CREATE TABLE IF NOT EXISTS label (
                        directory_id REFERENCES directory(id),
                        tag TEXT,
                        file TEXT,
                        PRIMARY KEY (directory_id, tag, file))'''

    create_directory = '''CREATE TABLE IF NOT EXISTS directory (
                            directory_id INTEGER PRIMARY KEY,
                            path TEXT)'''

    conn = sqlite3.connect('example.db')
    conn.execute(create_directory)
    conn.execute(create_label)
    conn.commit()


def file_generator(path, batch_size=64):
    '''
    Generate file paths which need to be indexed
    Yields lists of length at most batch_size
    '''

    # Take files only at the top level of the given directory
    files = next(os.walk(path))[2]
    
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    sql = '''SELECT DISTINCT file
             FROM label NATURAL JOIN directory
             WHERE file = ? AND path = ?'''
   
    # Take only files that have yet to be indexed 
    i = 0
    while i < len(files):
        
        file_names = filter(lambda x: x.endswith('.jpg'),
                            files[i: i+batch_size])

        logger.info('Checking index for {} potential new files'.format(len(file_names)))
        existing = []
        for f in file_names:
            cursor.execute(sql, (f, path))
            result = cursor.fetchone()
            if result:
                existing.append(result[0])
        new_files =  set(file_names) - set(existing)
        if new_files:
            logger.info('Found {} new files'.format(len(new_files)))
            yield new_files
        i += batch_size


def image_generator(path, batch_size=8):

    pool = ThreadPool(8)

    def _preprocess(fname):
        
        # Lazy load imports
        from skimage.io import imread
        from skimage.transform import resize
        
        if not fname.endswith('.jpg'):
            return None

        im = imread(os.path.join(path, fname)) 
        im = resize(im, (224, 224))
        return im

    for file_batch in file_generator(path):
 
        logger.info('Preprocessing {} images ...'.format(len(file_batch)))
        images = pool.map(_preprocess, file_batch)

        images = np.array([im for im in images if im is not None]) 
        
        yield images, file_batch

def prediction_generator(path):
    '''
    Predict on all images in the given directory
    '''

    def _get_model():
        '''
        Lazy loading the model to save time
        In addition, hack to surpress write to stderr during keras initialization
        '''

        _stderr = sys.stderr
        sys.stderr = open(os.devnull, 'wb')
        from keras.applications import mobilenet
        sys.stderr.close()
        sys.stderr = _stderr
        
        return mobilenet.MobileNet(weights='imagenet')
 
    def _decode(predictions):
        '''
        Lazy loading
        '''
        from keras.applications.imagenet_utils import decode_predictions
        return decode_predictions(predictions)

    model = None
    
    for images, files in image_generator(path):
        
        if model is None:
            logger.info('Loading model')
            model = _get_model()

        predictions = model.predict(images)
        decoded = _decode(predictions)
        yield decoded, files


def new_results(query, path):

    path = os.path.normpath(path)

    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    
    # Add record for directory
    sql = 'INSERT OR IGNORE INTO directory (path) VALUES (?)'
    values = (path,)
    c.execute(sql, values)
    sql = 'SELECT directory_id FROM directory WHERE path = ?'
    values = (path,)
    c.execute(sql, values)
    directory_id = c.fetchone()[0]
    
    # Run predictions and insert records

    sql = 'INSERT OR IGNORE INTO label (tag, file, directory_id) VALUES (?, ?, ?)'

    for pred_batch, fname_batch in prediction_generator(path):
        
        logger.info('Saving ...')
        for prediction, fname in zip(pred_batch, fname_batch):
            
            tags = map(lambda (_1, tag, _2): tag, prediction)
            records = map(lambda tag: (tag, fname, directory_id), tags)
            conn.executemany(sql, records)
            conn.commit()
            
            if query in tags:
                yield fname


def existing_results(query, path):
    '''
    Query database for existing search results
    '''

    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    sql = '''SELECT file
             FROM label NATURAL JOIN directory
             WHERE tag = ? AND path = ?'''

    for file_name, in c.execute(sql, (query, path)):
        yield file_name


def all_results(query, path):

    for result in existing_results(query, path):
        yield result
    
    for result in new_results(query, path):
        yield result


def main(query, path):
    verify_db_init()
    query = query.lower().replace(' ', '_')
    for result in all_results(query, path):
        print(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='search images by their visual content')
    parser.add_argument('query')
    parser.add_argument('path')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    main(args.query, args.path)

