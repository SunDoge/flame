"""
保存当前代码到指定目录
"""

import shlex
import subprocess
import zipfile
from typing import List
from zipfile import ZipFile


def git_ls_files() -> List[str]:
    """
    https://stackoverflow.com/questions/2766600/git-archive-of-repository-with-uncommitted-changes/63092214#63092214
    """
    cmd = 'git ls-files --others --exclude-standard --cached'
    output = subprocess.check_output(shlex.split(cmd))
    output_str = output.decode()
    return output_str.split('\n')[:-1]  # 字符串的最后有一个\n


def zip_files(file_list: List[str], filename: str):
    with ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as f:
        for file_path in file_list:
            f.write(file_path)


if __name__ == '__main__':
    file_list = git_ls_files()
    zip_files(file_list, 'code.zip')
