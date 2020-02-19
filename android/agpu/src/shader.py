#!/usr/bin/python

import sys
import os

def findAllShader(path):
  cmd = "find " + path + " -name \"*.glsl\""
  print("cmd:{}".format(cmd))
  vexs = os.popen(cmd).read().split('\n')
  output = []
  for f in vexs:
    if len(f) > 1:
      output.append(f)
  return output

def getName(filePath):
    print("getName filePath:" + filePath)
    dirPath, fileName = filePath.rsplit('/', 1)
    print("getName dirPath:{} filename:{}".format(dirPath, fileName))
    s1 = fileName.replace("/", "_")
    s1 = s1.replace(".", "_")
    return s1

def generateFile(headfile, sourcefile, shaders):
  print("head file:{}".format(headfile))
  print("source file:{}".format(sourcefile))
  h = "#pragma once\n"
  cpp = "#include \"shader.h\"\n"
  for s in shaders:
    name = getName(s)
    print (name)
    h += "extern const char* " + name + ";\n";
    cpp += "const char* " + name + " = \n";
    with open(s) as f:
      lines = f.read().split("\n")
      for l in lines:
        if (len(l) < 1):
          continue
        cpp += "\"" + l + "\\n\"\n"
    cpp += ";\n"

  with open(headfile, "w") as f:
    f.write(h);
  with open(sourcefile, "w") as f:
    f.write(cpp);

if __name__ == '__main__':
  print("__file__" + __file__)
  path, filename = __file__.rsplit('/', 1)
  print("path:" + path)
  print("filename:" + filename)

  shaders = findAllShader(path + "/glsl")
  generateFile(path + "/shader.h", path + "/shader.cpp", shaders)
