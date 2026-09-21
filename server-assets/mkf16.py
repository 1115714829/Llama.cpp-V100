src = open('/root/p60-ab-harness.sh').read()
dst = src.replace('--cache-type-k q8_0 --cache-type-v q8_0', '--cache-type-k f16 --cache-type-v f16')
assert dst != src, 'kv flags not found'
open('/root/p60-ab-harness-f16.sh', 'w').write(dst)
print('OK')
