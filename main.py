#!/usr/bin/env python
import argparse
import sqlite3
import os

def predict(dir_path):
    '''
    Predict on all images in the given directory
    '''
    
    # imports here instead of top level to save time
    import numpy as np
    sqlite3.register_adapter(np.float32, float)
    
    from keras.preprocessing.image import ImageDataGenerator 
    from keras.applications.imagenet_utils import decode_predictions
    from keras.applications import mobilenet
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    # Workaround to use the default flows provided with keras
    parent_dir, classname = os.path.split(os.path.normpath(dir_path))

    generator = ImageDataGenerator()
    flow = generator.flow_from_directory(parent_dir,
                                         classes=[classname],
                                         target_size=(224, 224))
    predictions = mobilenet_model.predict_generator(flow)
    decoded = decode_predictions(predictions)

    return decoded

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
    predictions = predict(path)

    print('Saving index ...')
    files = sorted([sorted(files) for root, _, files in os.walk(path)][0])
    for prediction, fname in zip(predictions, files):
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

