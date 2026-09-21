import struct, sys
path = sys.argv[1]

def read_value(f, vtype, depth=0):
    if depth > 4:
        raise ValueError('array too deep')
    if vtype == 1:   # INT8
        return struct.unpack('<b', f.read(1))[0]
    if vtype == 2:   # UINT16
        return struct.unpack('<H', f.read(2))[0]
    if vtype == 3:   # INT16
        return struct.unpack('<h', f.read(2))[0]
    if vtype == 4:   # UINT32
        return struct.unpack('<I', f.read(4))[0]
    if vtype == 5:   # INT32
        return struct.unpack('<i', f.read(4))[0]
    if vtype == 6:   # FLOAT32
        return struct.unpack('<f', f.read(4))[0]
    if vtype == 7:   # BOOL
        return bool(f.read(1)[0])
    if vtype == 8:   # STRING
        slen = struct.unpack('<Q', f.read(8))[0]
        if slen > 100000:
            raise ValueError('bad string len %d' % slen)
        return f.read(slen).decode('utf-8', 'replace')
    if vtype == 9:   # ARRAY
        etype = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        if count > 100000:
            raise ValueError('bad array count %d' % count)
        return [read_value(f, etype, depth + 1) for _ in range(count)]
    if vtype == 10:  # UINT64
        return struct.unpack('<Q', f.read(8))[0]
    if vtype == 11:  # INT64
        return struct.unpack('<q', f.read(8))[0]
    if vtype == 12:  # FLOAT64
        return struct.unpack('<d', f.read(8))[0]
    raise ValueError('unknown gguf type %d' % vtype)

f = open(path, 'rb')
magic = f.read(4)
version = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]
n_kv = struct.unpack('<Q', f.read(8))[0]
print('magic=%s version=%d n_tensors=%d n_kv=%d' % (magic, version, n_tensors, n_kv))
want = ('context', 'rope', 'max_pos', 'block_count', 'head_count', 'head_size',
        'embd', 'layer_count', 'nextn', 'attn')
for i in range(n_kv):
    start = f.tell()
    klen = struct.unpack('<Q', f.read(8))[0]
    if klen > 1000:
        print('key %d: BAD klen=%d at offset %d, stopping' % (i, klen, start))
        break
    key = f.read(klen).decode('utf-8', 'replace')
    vtype = struct.unpack('<I', f.read(4))[0]
    try:
        val = read_value(f, vtype)
    except Exception as e:
        print('key %d (%s) type=%d: parse error %s at offset %d, stopping' % (i, key, vtype, e, f.tell()))
        break
    if i < 20:
        print('key %2d off=%4d klen=%2d type=%d %-42s val=%r next_off=%d' % (i, start, klen, vtype, key, val, f.tell()))
    if any(s in key for s in want):
        print('  WANT: %-45s type=%d val=%r' % (key, vtype, val))
print('META_DONE')
