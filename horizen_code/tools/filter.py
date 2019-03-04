# -*- coding:utf-8 -*-
import os
import sys
IN_ROOT = '/home/omnisky/TF_Codes/horizen_code/tools/txt_output/rmaskP2_Enlarge_Noconcat'
OUT_ROOT = '/home/omnisky/TF_Codes/horizen_code/tools/txt_output/rmaskP2_Enlarge_Noconcat_REAL'

def read_and_save():
    if not os.path.exists(OUT_ROOT):
        os.mkdir(OUT_ROOT)
    for a_file in os.listdir(IN_ROOT):
        out_f = open(os.path.join(OUT_ROOT, a_file), 'w')
        with open(os.path.join(IN_ROOT, a_file)) as in_f:
            for a_line in in_f:
                score = a_line.strip().split()[1]
                score = float(score)
                if score < 1e-4:
                    continue
                out_f.write(a_line)
        out_f.close()

if __name__ == '__main__':

    read_and_save()
