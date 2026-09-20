import re, struct
path = '/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf'
with open(path, 'rb') as f:
    data = f.read(64 << 20)  # first 64 MiB
def full_key(s):
    e = s
    while e < len(data) and re.match(rb'[a-z0-9._]', data[e:e + 1]):
        e += 1
    st = s
    while st > 24 and re.match(rb'[a-z0-9._]', data[st - 1:st]):
        st -= 1
    return st, e
def val_at(e):
    t, = struct.unpack_from('<I', data, e)
    b = data[e + 4:e + 20]
    if t == 0: return 'u32=' + str(struct.unpack_from('<I', data, e + 4)[0])
    if t == 1: return 'i32=' + str(struct.unpack_from('<i', data, e + 4)[0])
    if t == 2: return 'f32=' + str(struct.unpack_from('<f', data, e + 4)[0])
    if t == 3: return 'bool=' + str(struct.unpack_from('<?', data, e + 4)[0])
    if t == 4:
        l, = struct.unpack_from('<Q', data, e + 4)
        return 'str=' + data[e + 12:e + 12 + l].decode('utf-8', 'replace')
    if t == 5: return 'u64=' + str(struct.unpack_from('<Q', data, e + 4)[0])
    if t == 6: return 'i64=' + str(struct.unpack_from('<q', data, e + 4)[0])
    if t == 7: return 'f64=' + str(struct.unpack_from('<d', data, e + 4)[0])
    if t == 8:
        n, = struct.unpack_from('<Q', data, e + 4)
        et, = struct.unpack_from('<I', data, e + 12)
        return 'array(n=%d et=%d raw=%s)' % (n, et, data[e + 16:e + 40].hex())
    return 'type=%d raw=%s' % (t, b.hex())
seen = set()
for pat in [b'attention.head', b'embedding.length', b'general.architecture', b'block_count', b'ffn', b'attn_head', b'rope']:
    for m in re.finditer(pat, data):
        st, e = full_key(m.start())
        key = data[st:e].decode('utf-8', 'replace')
        if key in seen:
            continue
        seen.add(key)
        # raw window: 8 bytes before key start (len prefix) through key + type + value
        a = max(24, st - 8)
        win = data[a:e + 16]
        print(key, '| len_prefix=%d' % struct.unpack_from('<Q', data, st - 8)[0] if st >= 32 else '?',
              '| type=%d' % struct.unpack_from('<I', data, e)[0],
              '| after_key_hex=%s' % data[e + 4:e + 12].hex())
        print('    raw:', win.hex())
