#!/usr/bin/python

import sys
import os

def findAllShader(path):
  cmd = "find " + path + " -name \"*.glsl\""
  vexs = os.popen(cmd).read().split('\n')
  output = []
  for f in vexs:
    if len(f) > 1:
      output.append(f)
  return output

def getName(fileName):
    s1 = fileName.replace("/", "_")
    s1 = s1.replace(".", "_")
    return s1

def generateFile(headfile, sourcefile, shaders):
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
  shaders = findAllShader('glsl')
  generateFile("shader.h", "shader.cpp", shaders);
