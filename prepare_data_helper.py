# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:07:09 2022

@author: Dmitry.Mokachev
"""

# ver 07.05.22.1
import json
import os
import subprocess
import re
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