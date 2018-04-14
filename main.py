#!/usr/bin/env python
import argparse
import sqlite3
import os

def predict(filename):

    import numpy as np
    sqlite3.register_adapter(np.float32, float)
    
    from keras.applications import mobilenet
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.imagenet_utils import decode_predictions
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    # Load
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # Preprocess
    processed_image = mobilenet.preprocess_input(image_batch.copy())
     
    # Run model
    predictions = mobilenet_model.predict(processed_image)
    labels = decode_predictions(predictions)

    return labels

def verify_database_init():
    sql = 'CREATE TABLE IF NOT EXISTS label (tag TEXT, file TEXT, confidence REAL)'
    conn = sqlite3.connect('example.db')
    conn.execute(sql)
    conn.commit()

def build_index(path):
    print('Building index. This may take some time...')
    
    verify_database_init()

    conn = sqlite3.connect('example.db')
    sql = "INSERT INTO label (tag, file, confidence) VALUES (?, ?, ?)"

    for item in os.listdir(path):
        file_name = os.path.join(path, item)
        if not os.path.isfile(file_name):
            continue
        result = predict(file_name)
        records = [(tag, file_name, confidence) for _, tag, confidence in result[0]]
        conn.executemany(sql, records)
        print(item)
    conn.commit()

def verify_index(path):
    
    verify_database_init()
    
    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    sql = 'SELECT * FROM label WHERE file LIKE ?'
    pattern = os.path.join(path, '%')
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

