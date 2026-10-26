import numpy as np
import csv
import os
import argparse

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='train or val or test')
parser.add_argument('--raw-dir', default=os.path.join(current_dir, 'raw'))
parser.add_argument('--output-dir', default=os.path.join(current_dir, 'output_csv'))
args = parser.parse_args()
split = args.split
val_size = 1000
scale_foot2meter = 0.3048

## extract partial data from train split as validation data
if args.split == 'val':
    split = 'train'
array_data = np.load(os.path.join(args.raw_dir, 'nba_'+split+'.npy'), mmap_mode='r')

os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, 'nba_'+args.split+'.csv'), 'w', newline='') as result:
    writer = csv.writer(result)

    num_batch = array_data.shape[0]
    if args.split == 'test':
        for batch in range(num_batch):

            if batch % 100 == 0:
                print(batch, '/', num_batch)

            for frame in range(30):
                for N in range(11):
                    traj = (batch*30+frame, batch*11+N, float(array_data[batch, frame, N, 0])*scale_foot2meter, float(array_data[batch, frame, N, 1])*scale_foot2meter)
                    writer.writerow(traj)


    elif args.split == 'val':

        for batch in range(num_batch):
            
            ## for val only:
            if batch == 1000:
                exit()

            if batch % 100 == 0:
                print(batch, '/', val_size)

            for frame in range(30):
                for N in range(11):                    
                    traj = (batch*30+frame, batch*11+N, float(array_data[batch, frame, N, 0])*scale_foot2meter, float(array_data[batch, frame, N, 1])*scale_foot2meter)
                    writer.writerow(traj)
    
    else:
        for batch in range(num_batch):
            
            ## skip the validation part
            if batch < val_size:
                continue
            
            if batch % 100 == 0:
                print(batch-val_size, '/', num_batch-val_size)
                
            for frame in range(30):
                for N in range(11):                    
                    traj = (batch*30+frame, batch*11+N, float(array_data[batch, frame, N, 0])*scale_foot2meter, float(array_data[batch, frame, N, 1])*scale_foot2meter)
                    writer.writerow(traj) 
