import sys
path = sys.argv[1]
f = open(path, 'rb')
data = f.read(96)
# hexdump
for off in range(0, len(data), 16):
    chunk = data[off:off+16]
    hexs = ' '.join('%02x' % b for b in chunk)
    asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print('%04x  %-48s  %s' % (off, hexs, asc))
print('HEX_DONE')
