from distutils.core import setup, Extension

setup(name='minumpy',
      version='1.0',
      packages=['minumpy'],
      ext_modules=[Extension(
        'minarray',
        ['minumpy/core/array_py.c',
         'minumpy/core/array_py_utils.c',
         'minumpy/core/array.c',
         'minumpy/core/array_dtypes.c',
         'minumpy/core/array_utils.c',
         ])
      ])
