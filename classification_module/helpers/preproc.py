import time
import hashlib
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def sha256sum(filename):
    ''' filename: string of pathname/filename to use to generate hash
        returns: hexidecimal sha256 hash of file
    '''
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def filelist_to_hash(fn_list,
                     truncate=32,
                    ):
    start = time.time()
    hst = []
    hs = []
    
    #Generate hash code for each file in list
    for i in tqdm(range(len(fn_list))):
        fn = Path(fn_list[i])
        file_type = fn.parts[-1].split('.')[-1]      
        if fn.exists():
            h = sha256sum(str(fn))
            hs.append(h)
            hst.append(h[0:truncate] + file_type)
        else:
            hs.append('FileNotFound')
            hst.append('FileNotFound')            
    hash_dict = {'original_fn':svs[0:len(hs)],
                 'hash_fn':hst,
                 'hash_full':hs}
    hash_df = pd.DataFrame(hash_dict)
    check_dupes(hash_df)
    print(time.time() - start, 'seconds elapsed')
    return hash_df

def check_dupes(df):
    for col in df.columns:
        print('%s duplicates: %d' % (col,df.duplicated(subset=col).values.sum()))