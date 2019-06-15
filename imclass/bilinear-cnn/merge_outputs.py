import shutil

with open('data/testing/output_file.txt','wb') as wfd:
    x = "data/testing/"
    for f in [x + 'dogs/final_labels.txt', x + 'cars/final_labels.txt', 
             x + 'flowers/final_labels.txt', x + 'aircrafts/final_labels.txt', x + 'birds/final_labels.txt']:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)