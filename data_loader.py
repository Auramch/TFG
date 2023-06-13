import os
import jsonlines
from collections import defaultdict, OrderedDict
from pathlib import Path


def prepare_all_data(SNLI_root, SNLI_files):    
    data_dict = {}
    for data_type, filename in SNLI_files.items():          
        filepath = os.path.join(SNLI_root, filename)        
        data_list = []
        with jsonlines.open(filepath) as jsonl_file:        
            for line in jsonl_file:
                pairID = line['pairID']                     
                gold_label = line['gold_label']            
                if gold_label != '-' and pairID.find('vg_') == -1:
                    imageId = pairID[:pairID.rfind('.jpg')] 
                    line['Flickr30K_ID'] = imageId
                    line = OrderedDict(sorted(line.items()))           
                    data_list.append(line)
        data_dict[data_type] = data_list                        
    all_data = data_dict['train'] + data_dict['dev'] + data_dict['test']
    image_index_dict = defaultdict(list)
    for idx, line in enumerate(all_data):
        pairID = line['pairID']
        imageID = pairID[:pairID.find('.jpg')]
        image_index_dict[imageID].append(idx)               
    return all_data, image_index_dict

def _split_data_helper(image_list, image_index_dict):
    ordered_dict = OrderedDict()
    for imageID in image_list:
        ordered_dict[imageID] = image_index_dict[imageID]
    return ordered_dict

def split_data(all_data, image_index_dict, split_root, split_files, SNLI_VE_root, SNLI_VE_files):
    with open(os.path.join(split_root, split_files['test'])) as f:
        content = f.readlines()
        test_list = [x.strip() for x in content]
    with open(os.path.join(split_root, split_files['train_val'])) as f:
        content = f.readlines()
        train_val_list = [x.strip() for x in content]
    train_list = train_val_list[:-1000]
    dev_list = train_val_list[-1000:]   

    train_index_dict = _split_data_helper(train_list, image_index_dict)
    dev_index_dict = _split_data_helper(dev_list, image_index_dict)
    test_index_dict = _split_data_helper(test_list, image_index_dict)
    all_index_dict = {'train': train_index_dict, 'dev': dev_index_dict, 'test': test_index_dict}

    for data_type, data_index_dict in all_index_dict.items():
        print('Current processing data split : {}'.format(data_type))
        with jsonlines.open(os.path.join(SNLI_VE_root, SNLI_VE_files[data_type]), mode='w') as jsonl_writer:
            for _, index_list in data_index_dict.items():
                for idx in index_list:
                    jsonl_writer.write(all_data[idx])


def main():
    SNLI_root = '/data-fast/107-data4/amartinez/SNLI-VE/snli_1.0'
    SNLI_files = {'dev': 'snli_1.0_dev.jsonl', 'test': 'snli_1.0_test.jsonl', 'train': 'snli_1.0_train.jsonl', 'prova': 'snli_1.0_prova.jsonl'}
    user_home = Path(os.path.expanduser('~'))
    split_root = user_home/'Src/SNLI-VE/data'
    split_files = {'test': 'flickr30k_test.lst', 'train_val': 'flickr30k_train_val.lst'}
    SNLI_VE_root = '/data-fast/107-data4/amartinez/SNLI-VE/snli_1.0/snli_ve'
    SNLI_VE_files = {'dev': 'snli_ve_dev.jsonl', 'test': 'snli_ve_test.jsonl', 'train': 'snli_ve_train.jsonl', 'prova': 'snli_ve_prova.jsonl'}

    print('*** SNLI-VE Generation Start! ***')
    all_data, image_index_dict = prepare_all_data(SNLI_root, SNLI_files)              
    split_data(all_data, image_index_dict, split_root, split_files, SNLI_VE_root, SNLI_VE_files)
    print('*** SNLI-VE Generation Done! ***')

if  __name__ == '__main__':
    main() 
