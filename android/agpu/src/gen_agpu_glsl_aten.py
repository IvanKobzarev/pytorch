#!/usr/bin/python

import sys
import os
import re

def findAllShader(path):
  cmd = "find " + path + " -name \"*.glsl\""
  vexs = os.popen(cmd).read().split('\n')
  output = []
  for f in vexs:
    if len(f) > 1:
      output.append(f)
  output.sort()
  return output

def getName(filePath):
    dirPath, fileName = filePath.rsplit('/', 1)
    return fileName.replace("/", "_").replace(".", "_")

def generateFile(headfile, sourcefile, shaders):
  print("head file:{}".format(headfile))
  print("source file:{}".format(sourcefile))
  h = "#pragma once\n"
  h += "#include <ATen/native/vulkan/glsl.h>"
  nsbegin = "\nnamespace at { namespace native { namespace vulkan { namespace gl {\n"
  nsend = "\n} } } } //namespace at::native::vulkan::gl\n"

  h += nsbegin

  cpp = "#include <ATen/native/vulkan/glsl.h>"
  cpp += nsbegin

  for s in shaders:
    name = getName(s)
    h += "extern const char* " + name + ";\n"
    cpp += "const char* " + name + " = \n"
    with open(s) as f:
      lines = f.read().split("\n")
      for l in lines:
        if (len(l) < 1):
          continue
        cpp += "\"" + l + "\\n\"\n"
    cpp += ";\n"

  cpp += nsend

  h += nsend

  with open(headfile, "w") as f:
    f.write(h)
  with open(sourcefile, "w") as f:
    f.write(cpp)

if __name__ == '__main__':
  path, filename = __file__.rsplit('/', 1)
  shaders = findAllShader(path + "/glsl")
  generateFile(path + "/glsl.h", path + "/glsl.cpp", shaders)
