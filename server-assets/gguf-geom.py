import sys
try:
    import gguf
except Exception as e:
    print('NO_GGUF_PKG', e); sys.exit(0)
r = gguf.GGUFReader(sys.argv[1])
keys = [k for k in r.fields.keys()]
want = [k for k in keys if any(s in k for s in ('attention.head','attention.key_length','attention.value_length','embedding_length','block_count','attention.head_count','ssm','linear','full_attention'))]
for k in sorted(want):
    f = r.fields[k]
    try:
        v = f.contents()
    except Exception:
        v = f.parts[-1]
    print(k, '=', v)
print('--- arch ---', r.fields.get('general.architecture').contents() if 'general.architecture' in r.fields else '?')