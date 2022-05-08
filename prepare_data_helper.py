# -*- coding: utf-8 -*-
# ver 07.05.22.2

import json
import os
import subprocess
import re
import numpy as np
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

#update 08.05
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
      np.savetxt(file_full_name_lbl, yolo_data)

      # save txt file with labels
      with open(file_full_name_lbl, "w") as output:
        output.write(str(yolo_data))

      # copy image and change its file name
      result = os.system(f'cp "{original_img_dir}/{file_name}" {file_full_name_img}')

  # check result
  len(json_data)-1
  print(f'In json were - {len(json_data)-1} files')
  print(f'Conderted - {file_id} files')
  print('Saved images -', len(os.listdir(train_yolo_dir+'/images/'+train_val_dir)), 'files')
  print('Saved labels -', len(os.listdir(train_yolo_dir+'/labels/'+train_val_dir)), 'files')

