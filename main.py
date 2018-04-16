#!/usr/bin/env python
import argparse
import sqlite3
import os
from multiprocessing.dummy import Pool as ThreadPool
from skimage.io import imread
from skimage.transform import resize
import numpy as np
sqlite3.register_adapter(np.float32, float)

def verify_database_init():
    create_label = 'CREATE TABLE IF NOT EXISTS label (          \
                        directory_id REFERENCES directory(id),  \
                        tag TEXT,                               \
                        file TEXT,                              \
                        confidence REAL)'
    create_directory = 'CREATE TABLE IF NOT EXISTS directory (  \
                            id PRIMARY KEY,                     \
                            path TEXT)'                         
    conn = sqlite3.connect('example.db')
    conn.execute(create_directory)
    conn.execute(create_label)
    conn.commit()

def image_generator(path, batch_size=8):

    # Take files at the top level of the given directory
    files = next(os.walk(path))[2]

    pool = ThreadPool(8)

    def _preprocess(fname):
        print(fname)
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

def build_index(path):
    print('Building index. This may take some time...')
    
    verify_database_init()

    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    
    # Add directory entry
    sql = 'INSERT INTO directory (path) VALUES (?)'
    values = (os.path.normpath(path),)
    c.execute(sql, values)
    directory_id = c.lastrowid

    # Run predictions and insert records

    sql = 'INSERT INTO label (tag, file, confidence, directory_id) VALUES (?, ?, ?, ?)'

    print('Running mobilenet ...')
    for pred_batch, fname_batch in prediction_generator(path):
        
        print('Saving ...')
        for prediction, fname in zip(pred_batch, fname_batch):
            records = [(tag, fname, confidence, directory_id) for _, tag, confidence in prediction]
            conn.executemany(sql, records)
    
    conn.commit()

    print('Done')

def verify_index(path):
    
    verify_database_init()
    
    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    sql = 'SELECT * FROM directory WHERE path = ?'
    pattern = os.path.normpath(path)
    c.execute(sql, (pattern,))
    if c.fetchone():
        return True
    else:
        build_index(path)
        #if input('No index found. Build one now? [y/N] ') in ('y', 'Y'):
        #    build_index(path)
    return True

def prepare_query(query):
    return query.lower().replace(' ', '_')

def search(query, path):

    query = prepare_query(query)

    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    sql = "SELECT file, confidence FROM label WHERE tag = ? ORDER BY confidence DESC"
    c.execute(sql, (query,))

    for file_name, confidence in c.fetchall():
        print(os.path.relpath(file_name, path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='search images by their visual content')
    parser.add_argument('query')
    parser.add_argument('path')

    args = parser.parse_args()

    if verify_index(args.path):
        search(args.query, args.path)

