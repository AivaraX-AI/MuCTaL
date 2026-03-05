import os
import sys
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE


def remove_space_fn(pnfn):
    fn = Path(pnfn)
    last_path = fn.parts[-1].replace(' ','_')
    # print('New name:', last_path)
    # ns_pnfn = fn.parent.joinpath(last_path)
    return last_path

# def symlink_to_fn_no_space(fn,dstname):
#     fn = Path(fn)
#     last_path = fn.parts[-1].replace(' ','_')
#     print('New name:', last_path)
#     os.symlink(fn,fn.parent.joinpath(last_path))

def rclone_copy(src,dest,symlink=False):
    if symlink:
        args = ['rclone','copy',src,dest,'--copy-links']
    else:
        args = ['rclone','copy',src,dest]
    process = Popen(args, stdout = PIPE, stderr=PIPE)
    process.wait()
    stdout, stderr = process.communicate()
    # print(stdout)
    # print(stderr)
          