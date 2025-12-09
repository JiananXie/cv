import pickle
with open('places-googlenet.pickle', 'rb') as f: 
    for key in pickle.load(f, encoding='bytes').keys():
        print(key)