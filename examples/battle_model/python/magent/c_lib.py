""" some utility for call C++ code"""

from __future__ import absolute_import

import os
import ctypes
import platform
import multiprocessing

# 该库介绍如何加载一个共享库


def _load_lib():
    """ Load library in build/lib. """
    # 获取当前 Python 文件的绝对路径
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, "../../build/")

    # 根据操作系统选择合适的共享库文件
    # macOS 使用 libmagent.dylib
    if platform.system() == 'Darwin':
        path_to_so_file = os.path.join(lib_path, "libmagent.dylib")
    # Linux 使用 libmagent.so
    elif platform.system() == 'Linux':
        path_to_so_file = os.path.join(lib_path, "libmagent.so")
    else:
        raise BaseException("unsupported system: " + platform.system())

    # 从地址中加载共享库，并返回库对象
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


# 将 NumPy 数组转换为 float 类型的 C 数组
def as_float_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# 将 NumPy 数组转换为 int32 类型的 C 数组
def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# 将 NumPy 数组转换为 bool 类型的 C 数组
def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


# 将 NumPy 数组转换为 bool 类型的 C 数组
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count() // 2)
_LIB = _load_lib()
