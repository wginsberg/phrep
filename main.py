#!/usr/bin/env python
import argparse
import sqlite3
import os
from multiprocessing.dummy import Pool as ThreadPool
from skimage.io import imread
from skimage.transform import resize
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


def image_generator(path, batch_size=8):

    # Take files at the top level of the given directory
    files = next(os.walk(path))[2]

    pool = ThreadPool(8)

    def _preprocess(fname):
        if not fname.endswith('.jpg'):
            return None        
        im = imread(os.path.join(path, fname)) 
        im = resize(im, (224, 224))
        return im

    i = 0
    while i < len(files):
        
        file_names = filter(lambda x: x.endswith('.jpg'),
                            files[i: i+batch_size])
        
        print('Preprocessing {} images ...'.format(len(file_names)))
        images = pool.map(_preprocess, file_names)

        print('Building numpy array ...')
        images = np.array([im for im in images if im is not None]) 
        
        yield images, file_names
        i += batch_size

def prediction_generator(path):
    '''
    Predict on all images in the given directory
    '''
    
    # lazy load model import
    from keras.applications.imagenet_utils import decode_predictions
    from keras.applications import mobilenet
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    for images, files in image_generator(path):
        print('Running model ...')
        predictions = mobilenet_model.predict(images)
        decoded = decode_predictions(predictions)
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

    print('Running mobilenet ...')
    for pred_batch, fname_batch in prediction_generator(path):
        
        print('Saving ...')
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

    args = parser.parse_args()

    main(args.query, args.path)

