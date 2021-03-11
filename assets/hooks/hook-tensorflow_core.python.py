from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('tensorflow_core')
datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)

## Add to 
# pyinstaller xx.spec --additional-hooks-dir=hooks
