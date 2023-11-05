# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import numpy as np
import math
import crc8


def crc(data, mode='encode'):
    """
    encode or decode data with CRC8
    """
    if mode == 'encode':
        wm = data[0]
    else:
        wm = data[0, :-8]

    wm_str = ''
    for i in wm:
        wm_str = wm_str + str(i)
    padding = math.ceil(len(wm_str)/4)*4 - len(wm_str)
    wm_str = '0'*padding + wm_str

    wm_bin = wm_str
    wm_hex = ''
    wm_len = len(wm_bin)

    for i in range(0, wm_len, 4):
        c = hex(int(wm_bin[i:i+4], 2))[2:]
        wm_hex += c
    
    byte_data = bytearray.fromhex(wm_hex)
    
    hash = crc8.crc8()
    hash.update(byte_data)
    crccode = hash.digest()
    crccode = ''.join(format(x, '08b') for x in crccode)
    
    if mode == 'encode':
        secret = wm_bin[padding:] + crccode
        secret = np.array([int(x) for x in secret])
        return np.array([secret])

    elif mode == 'decode':
        targetcrccode = data[0, -8:]
        crccode = np.array([int(x) for x in crccode])
        if np.all(targetcrccode == crccode):
            return 1
        else:
            return 0
 