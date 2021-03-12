# Author: Dean
# Time  : 3/12/2021  10:03
# File  : rename.py
# Info  : 该脚本可修改当前文件夹（路径可修改）内，符合条件的文件的文件名
import os
import re
path = "."  # 当前文件夹（可修改）
filesList = os.listdir(path)    # 获取path目录下的所有文件，存入filesList
for oldName in filesList:
    mat = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", oldName)    # 正则表达式匹配日期（可修改）
    if mat: # 如果该文件名符合我们要更换掉的格式
        date = mat.group()  # date为我要更换掉的oldName中的日期格式字符串
        newName = oldName.replace("_" + date, "")   # 为oldName设置newName
        # print(oldName, newName)   # 不确定是否设置正确的话，可以先打印出来
        os.rename(oldName, newName)                 # 用os模块中的rename方法对文件改名