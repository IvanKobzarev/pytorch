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
  h += "#include \"agpu.h\""
  nsbegin = "\nnamespace agpu {\n"
  nsend = "\n} //namespace agpu\n"

  h += nsbegin

  cpp = "#include \"agpu_glsl.h\"\n"
  cpp += "#include <cassert>"
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

  h += "\n\nenum class AConv : int32_t {\n"
  i = 0
  convShaderNames = []
  convShaderNamesDict = {}

  for s in shaders:
    name = getName(s).replace("_glsl", "")
    if (re.match(r".*conv.*", s)):
      convShaderNames.append(name)

  for conv in convShaderNames:
      code = 10 * i
      convShaderNamesDict[conv] = code
      h += "  " + conv + " = " + str(code) + ",\n"
      i += 1

  enumClassName = "AConv"
  funNameEnumToFun = "aConvToFun"
  funNameCodeToEnum = "aConvByCode"

  h += "};\n"
  h += "\nusing fp_agpu_conv_t = decltype(&::agpu::conv_);"
  h += "\nfp_agpu_conv_t {}({} aconv);".format(funNameEnumToFun, enumClassName)
  h += "\n{} {}(int64_t code);".format(enumClassName, funNameCodeToEnum)

  h += nsend

  cpp += "\nfp_agpu_conv_t {}({} aconv)".format(funNameEnumToFun, enumClassName)
  cpp += "{\n  switch (aconv){"
  for conv in convShaderNames:
    cpp += "\n    case "+enumClassName+"::" + conv + ":" \
        +  "\n      return &::agpu::" + conv + ";"
  cpp += "\n  }" # switch
  cpp += "\n  assert(false);"
  cpp += "\n}" # fun


  cpp += "\n\n{} {}(int64_t code)".format(enumClassName, funNameCodeToEnum)
  cpp += "{\n  switch (code){"
  for conv in convShaderNames:
    cpp += "\n    case " + str(convShaderNamesDict[conv]) + ":" \
           +  "\n      return AConv::" + conv + ";"
  cpp += "\n  }" # switch
  cpp += "\n  assert(false);"
  cpp += "\n}" # fun


  cpp += nsend

  with open(headfile, "w") as f:
    f.write(h)
  with open(sourcefile, "w") as f:
    f.write(cpp)

if __name__ == '__main__':
  path, filename = __file__.rsplit('/', 1)
  shaders = findAllShader(path + "/glsl")
  generateFile(path + "/agpu_glsl.h", path + "/agpu_glsl.cpp", shaders)
