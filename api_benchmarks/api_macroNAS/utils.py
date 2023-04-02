OPS_LIST = ['I', '1', '2']

def get_hashKey(arch_int):
    hashKey = ''
    for i, ele in enumerate(arch_int):
        if i in [4, 8, 12]:
            hashKey += '|'
        if ele != 0:
            hashKey += OPS_LIST[ele]
    return hashKey