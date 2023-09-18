import os
import albumentations as A
from pathlib import Path
import cv2

img_folder_path = './datasets/vehicles/images/'
annot_folder_path = './datasets/vehicles/labels/'

# main function to augment images
def transformer(img_folder, annot_folder_path):
    
    counter = 0
    
    img_paths = Path(img_folder).glob('*.jpg')   

    for img_path in img_paths:

        img_name = str(img_path).split('\\')[-1].split('.')[:-1]
        img_name = '.'.join(img_name)
        label_name = img_name + '.txt'
        
        label_path = annot_folder_path + label_name
                    
        result = horizontalFlip(str(img_path), label_path, img_name)    
        counter = counter + result
            
    
    print('{} images horizontally flipped and saved.'.format(str(counter)))
    return 

   
def horizontalFlip(img_path, label_path, save_name):  
    
    # setup transformer
    transformHorizontalFlip = A.Compose([A.HorizontalFlip(always_apply=True)], 
                      bbox_params=A.BboxParams(format='yolo'))  
    
    # annotation file to list
    with open(label_path) as f:
        lines = f.readlines()

    bbox = []
    for line in lines:
        l = line.split(' ')
        bbox.append([float(l[1]), float(l[2]), float(l[3]), float(l[4]), str(l[0])])
    
    img = cv2.imread(img_path) 
    
    try:
        trns = transformHorizontalFlip(image = img, bboxes=bbox)
    
        # transformation results
        img_transformed = trns['image']   
        boxes = trns['bboxes']
        
        # save image and annotation file in YOLO format 
        cv2.imwrite('./src/ml_data_processing/outputs/augmentation/images/'+ save_name + '_hf.jpg', img_transformed)    
        
        with open(r'src/ml_data_processing/outputs/augmentation/labels/'+ save_name + '_hf.txt' , 'w') as f:     
            for box in boxes:
                f.write(str(box[-1]))
                f.write(' '+str(box[0]))
                f.write(' '+str(box[1]))
                f.write(' '+str(box[2]))
                f.write(' '+str(box[3]))
                f.write('\n')
            
        return 1
    
    except:
        print('Failed:', save_name)
        return 0


transformer(img_folder_path, annot_folder_path)