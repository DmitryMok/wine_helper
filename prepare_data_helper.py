# -*- coding: utf-8 -*-
# ver 07.05.22.2

import json
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output

def get_json_file(file_json: str):
  """ Function for load json file
  :param: file_json - string, path to file 
  :return: dictionary 
  """
  with open(file_json) as fp:
      json_data = json.load(fp)
  return json_data

def execute(cmd, stderr=False):
    """ Function run linux command
    :param: cmd - string, command with parameters 
    :return: generator of output lines
    """
    popen = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    for stdout_line in iter([popen.stdout.readline, popen.stderr.readline][stderr], ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def get_gdrive_file_by_link(gdrive_link, new_file_name, show_progress=True):
  """
  Function copy file from shared google drive link
  :param: gdrive_link - string, link to shared file
  :param: new_file_neme - string, like data.zip
  :return: result
  """
  REG_STRING = '"s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p"'
  GDOCS_LINK = '"https://docs.google.com/'
  
  get_id_reg = '\/([^\/]*)\/view' # for regex link to get 'id' only
  # check the link (try to get id)
  try:
    gdrive_id = re.findall(get_id_reg, gdrive_link)[0]
  except:
    return print(f'Something wrong!\nCheck the link {gdrive_link}\nit must be like'
    ' "https://drive.google.com/file/d/19FOvC_xkZqmmiCJK7XS7WohJMIv3g4DQ/view?usp=sharing"')

  # form the command
  gdrive_link = (f'{GDOCS_LINK}uc?export=download&confirm=$(wget --quiet'
  f' --save-cookies /tmp/cookies.txt --keep-session-cookies'
  f' --no-check-certificate {GDOCS_LINK}uc?export=download&'
  f'id={gdrive_id}" -O- | sed -rn {REG_STRING})&id={gdrive_id}"')
  
  # print(gdrive_link)  # debug

  wget_command = f'wget --load-cookies /tmp/cookies.txt {gdrive_link} -O {new_file_name} && rm -rf /tmp/cookies.txt'
  # run and get the result
  if show_progress:
    for path in execute(wget_command, stderr=True):
        if path != '\n':
          clear_output(wait=True)
        print(path, end="")
  else:
    result = subprocess.run([wget_command], 
                            shell=True, 
                            capture_output=True,
                            # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                            text=True).stderr.strip("\n")
    
    if os.system(wget_command):
      result = f'Something wrong! \nCheck it manualy:\n"{wget_command}"'
    else:
      result = 'Successfull!'
    return print(result)

#update 08.05.4
def xy2xywh(x1,x2,y1,y2):
  x, y = (x1 + x2) / 2, (y1 + y2) / 2
  w, h = x2 - x1, y2 - y1
  return x, y, w, h

def change_file_ext(fname, ext='txt'):
  '''
  Change extention of file name 'dir/file.jpg' -> 'dir/file.ext' default - ext=txt
  :return: filename, string
  '''
  return ''.join(fname.split('.')[:-1])+'.'+ext

def convert_SAjson2yolo(json_data, 
                        original_img_dir, 
                        train_yolo_dir, 
                        train_val_dir='train',
                        mono_class=False):
  '''
  convert Super annotate json data to yolo format and save
  data must be obtained using the function get_json_file()
  :param: train_val_dir - train or validation dir to save images and labels. Default = 'train'
  :param: mono_class - True - all bbox class 0, False - original 
  '''

  # create train_data dir: train_yolo_dir/[images|labels]/train_val_dir
  if os.path.exists(train_yolo_dir+'/images/'+'train') and os.path.exists(train_yolo_dir+'/labels/'+'train'):
    print('folders', train_yolo_dir+'/[images|labels]/'+'train already exist!')
  else:
    os.makedirs(train_yolo_dir+'/images/'+train_val_dir, exist_ok=True)
    os.makedirs(train_yolo_dir+'/labels/'+train_val_dir, exist_ok=True)
    print('folders', train_yolo_dir+'/[images|labels]/'+'train created!')
  
  # convert 2 yolo
  for file_id, file_name in enumerate(json_data):
    if type(json_data[file_name]) == dict:
      img_h, img_w = cv2.imread(original_img_dir+file_name).shape[:2]
      yolo_data = []
      for points_data in json_data[file_name]['instances']:
        # convert absolute x1..y2 to relative (x,y) center and (w,h) bbox
        x,y,w,h = xy2xywh(points_data['points']['x1']/img_w,
                          points_data['points']['x2']/img_w,
                          points_data['points']['y1']/img_h,
                          points_data['points']['y2']/img_h)
        
        # print(file_name, (w,h), points_data)
        if mono_class:
          class_id = 0
        else:
          class_id = points_data['classId']

        yolo_data.append([class_id, x, y, w, h])

      # copy image and save labels txt
      file_full_name_img = train_yolo_dir+'/images/'+train_val_dir+'/'+'img_{0:05d}.'.format(file_id)+file_name.split('.')[-1]
      file_full_name_lbl = train_yolo_dir+'/labels/'+train_val_dir+'/'+'img_{0:05d}.txt'.format(file_id)

      # save txt file with labels
      np.savetxt(file_full_name_lbl, yolo_data)

      # copy image and change its file name
      os.system(f'cp "{original_img_dir}/{file_name}" {file_full_name_img}')

  # check result
  len(json_data)-1
  print(f'In json were - {len(json_data)-1} files')
  print(f'Converted - {file_id} files')
  print('Saved images -', len(os.listdir(train_yolo_dir+'/images/'+train_val_dir)), 'files')
  print('Saved labels -', len(os.listdir(train_yolo_dir+'/labels/'+train_val_dir)), 'files')

################################################################
# visualisation functions
################################################################
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
pi = 3.1415926535


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, bbox_type='bbox', bbox_text = None, font_scale = 0.5):
  '''
  Draw bbox on the image
  :param: bbox - bbox absolute coordinates
  :return: same image with bbox
  '''
  x_cntr, y_cntr, w, h = map(int, bbox) # coordinates of centers and dimensions of boxes 
  x_min, x_max, y_min, y_max = int(x_cntr - w/2), int(x_cntr + w/2), int(y_cntr - h/2), int(y_cntr + h/2)
  center_coordinates = (x_cntr, y_cntr)

  if bbox_type=='ellipse':
    a, b = int(w/2), int(h/2)
    axesLength = (a, b)
    cv2.ellipse(img, center_coordinates, axesLength, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)  #angle=0, startAngle=0, endAngle=360, 
  elif bbox_type=='circle':
    radius = int((w + h) / 4)
    cv2.circle(img, center_coordinates, radius, color=color, thickness=thickness)
    pass
  else:
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

  if bbox_text:
    text = str(bbox_text)
    ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)    
    # Prints the text.    
    img = cv2.rectangle(img, (x_cntr-text_width//2, y_cntr - int(40*font_scale)), (x_cntr + text_width//2, y_cntr), BOX_COLOR, -1)
    # img = cv2.putText(img, label, (x1, y1 - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)    
    cv2.putText(
        img,
        text=text,
        # org=(x_cntr - int(text_width/2), y_cntr + int(0.5 * text_height)),
        org=(x_cntr-text_width//2, y_cntr-int(10*font_scale)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale, 
        color=TEXT_COLOR 
        # lineType=cv2.LINE_AA,
    )
  return img  # image with bboxes


def visualize(image, bboxes, thickness=2, bbox_type='bbox', bboxes_text = [], font_scale = 0.5):
  '''
  Draw bboxes on image
  :param: bbox_type - ellipse, circle or bbox (default)
  :param: bboxes_text - list of text to draw in the center of earch bboxes, int/float/string
  :param: font_scale - font size
  '''
  img = image.copy()
  bbxs = bboxes.copy()

  # print(img.shape, len(bbxs))
  if len(bbxs):
    bbxs[:,[0,2]] = bbxs[:,[0,2]] * img.shape[1]
    bbxs[:,[1,3]] = bbxs[:,[1,3]] * img.shape[0]
  for i,bbox in enumerate(bbxs):
      if len(bboxes_text):
        bbox_text=bboxes_text[i]
      else:
        bbox_text=[]  
      img = visualize_bbox(img, bbox, thickness=thickness, bbox_type=bbox_type, bbox_text=bbox_text, font_scale=font_scale)
  return img

def resize_image(img, width=300):
  # resize image
  height = int(img.shape[0] * width/img.shape[1])
  dim = (width, height)
  
  # resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

  return resized

def draw_images_row(img_list, fig_size=22, titles=None):
  '''
  Draw images in line
  :param: img_list - list of images (numpy)
  :param: fig_size - width of line, int (default = 22)
  :param: titles - list of titles (list of string)
  '''
  fig = plt.figure(figsize=(fig_size, int(fig_size*3*img_list[0].shape[1]/img_list[0].shape[1])+1))
  plt.subplots_adjust(wspace=0, hspace=0)
  rows = 1
  columns = 3
  if not titles:  titles=[str(i+1) for i in range(len(img_list))] # if title absant, it will be num
  for i in range(0, len(img_list)):
      # print(i+1)
      fig.add_subplot(rows, columns, i+1)
      plt.axis('off')
      plt.title(titles[i])
      plt.imshow(img_list[i])
  plt.show()

#update 15.05.4
  
CROP_SIZE = 128 # размер обрезанного изображения
CROP_MARGIN = 1.3 # размер полей по отношению к max стороне bbox

# функция вырезает из исходного изображения фрагменты с keypoints/bbox
# если bb=1 значит на вход подается готовый bbox, иначе key points
def crop_fragment_with_bbox(img, kp_list, rsz=CROP_SIZE, dxy=0.2, show=1, bb=0): 
    '''
    Function cuts fragment with bbox or kp. Margin = CROP_MARGIN (default 1.3)
    :param: kp_list - int, corner coordinates (xy), 
    :param: rsz - int, resize to CROP_SIZE (0 - no resize), 
    :param: dxy - [0:1], random offset (relate max(w,h))
    :param: bb - bounding box or key point, default bb=0 (key point)
    '''
    if bb:
      bbxs = kp_list
    else:
      # координаты bbox вокруг keypoints
      x_center, y_center, w, h = (max(kp_list[:,0])+min(kp_list[:,0]))/2, (max(kp_list[:,1])+min(kp_list[:,1]))/2, max(kp_list[:,0])-min(kp_list[:,0]), max(kp_list[:,1])-min(kp_list[:,1]) 
      bbxs = np.array([x_center, y_center, w, h])[None, ...]
    if len(bbxs):
      bbxs[:,[0,2]] = bbxs[:,[0,2]] * img.shape[1]
      bbxs[:,[1,3]] = bbxs[:,[1,3]] * img.shape[0]
      # kp_list[:,0] = kp_list[:,0] * img.shape[1]
      # kp_list[:,1] = kp_list[:,1] * img.shape[0]

    x, y, w, h = bbxs[0]
    cm, dx, dy = CROP_MARGIN, 0, 0  #константы будут изменены, если делать случайное смещение

    # случайное смещение
    if dxy:
      cm = cm + (0.5-np.random.random())*dxy*2
      dx = max(w,h) * dxy
      dy = int(0.5*dx - r(dx))
      dx = int(0.5*dx - r(dx))
    # print(CROP_MARGIN, cm)
    # размер окна вокруг bbox
    dim = max(w,h) * CROP_MARGIN
    x_min, x_max, y_min, y_max = int(x - dim/2)+dx, int(x + dim/2)+dx, int(y - dim/2)+dy, int(y + dim/2)+dy
    # новые координаты keypoints
    # print(kp_list[:,0], x_min)
    kp_list[:,0], kp_list[:,1] = kp_list[:,0] - x_min, kp_list[:,1] - y_min
    # оригинальное изображение
    # print((x_min, y_min), (x_max, y_max))
    if min(x_min, y_min)<0 or img.shape[1]<x_max or img.shape[0]<y_max:
      cropped_img = crop_bbox(img, x_min, y_min, x_max, y_max)
    else:
      cropped_img = img[y_min:y_max,x_min:x_max,:][:,:,::-1]
    # print(cropped_img.shape)
    
    new_bbox = []
    for kp in kp_list:
      if bb:
        # new_bbox.append([kp[0], kp[1], kp[2], kp[3]])
        new_bbox.append([x_min, y_min, dim, dim])
      else:
        new_bbox.append([kp[0], kp[1], 0.01, 0.01]) # 0.01 осталось от legacy, надо выпилить

    new_bbox = np.array(new_bbox) #[None, ...]
    # new_bbox[:,[0,2]] = new_bbox[:,[0,2]] / cropped_img.shape[1]
    # new_bbox[:,[1,3]] = new_bbox[:,[1,3]] / cropped_img.shape[0]
    new_bbox[:,[0,2]] = new_bbox[:,[0,2]] / img.shape[1]
    new_bbox[:,[1,3]] = new_bbox[:,[1,3]] / img.shape[0]

    if rsz:
      # transformed = transform(image=cropped_img, bboxes=bbxs, category_ids=category_ids[i:i+4])
      cropped_img = cv2.resize(cropped_img, (rsz,rsz))

    if show:
      visualize(cropped_img, new_bbox)
    # cv2_imshow(cropped_img)

    return cropped_img, [new_bbox[:,:2],new_bbox[:,:4]][bb] # для kp возвращаем только массив xy, для bbox - xywh

# функция вырезает bbox и дополняет серым область, которая выходит за край изображения
# на вход подается изображение и координаты, по которым надо вырезать фрагмент
def crop_bbox(img, x_min, y_min, x_max, y_max):
  '''
  Function crops bbox from image
  :param: img - numpy array
  :param: x_min (x_max/y_min/y_max) - int, abs coordinates 
  '''
  h_cropped, w_cropped = y_max-y_min, x_max-x_min
  blank_img = np.zeros((h_cropped, w_cropped, 3))+127
  # определяем смещение за границу изображения
  y_min_offset = max(0, -y_min)
  y_max_offset = min(0, img.shape[0] - y_max)
  x_min_offset = max(0, -x_min)
  x_max_offset = min(0, img.shape[1] - x_max)
  # сдвигаем координаты
  y_min += y_min_offset
  y_max += y_max_offset
  x_min += x_min_offset
  x_max += x_max_offset

  blank_img[y_min_offset:h_cropped+y_max_offset,x_min_offset:w_cropped+x_max_offset,:] = img[y_min:y_max,x_min:x_max,:][:,:,::-1]

  return blank_img  # изображение в формате RGB

def r(val):
  '''
  Function generate random integer [0:val]
  '''
  return int(np.random.random() * val)
