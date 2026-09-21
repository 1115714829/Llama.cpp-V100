import sys, struct
def rd(f, fmt, n=None):
    sz = struct.calcsize(fmt)
    d = f.read(sz)
    return struct.unpack(fmt, d)
def rstr(f):
    (n,) = rd(f, '<Q')
    return f.read(n).decode('utf-8', 'replace')
p = sys.argv[1]
f = open(p, 'rb')
magic = f.read(4)
(ver,) = rd(f, '<I')
(nt,) = rd(f, '<Q')
(nk,) = rd(f, '<Q')
print('magic', magic, 'ver', ver, 'n_tensors', nt, 'n_kv', nk)
TY = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
def rval(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        (et,) = rd(f, '<I'); (n,) = rd(f, '<Q')
        return [rval(f, et) for _ in range(n)]
    fm = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t]
    return rd(f, fm)[0]
kv = {}
for _ in range(nk):
    k = rstr(f); (t,) = rd(f, '<I'); v = rval(f, t)
    kv[k] = (TY.get(t, t), v)
for k in sorted(kv):
    t, v = kv[k]
    s = str(v)
    if len(s) > 120: s = s[:120] + '...'
    print('  ', k, '=', s)
tsz = {}
for _ in range(nt):
    nm = rstr(f); (nd,) = rd(f, '<I'); ds = rd(f, '<Q'*nd); (tt,) = rd(f, '<I'); (off,) = rd(f, '<Q')
    tsz[nm] = (ds, tt)
tot = 0
for nm in sorted(tsz):
    ds, tt = tsz[nm]
    n = 1
    for d in ds: n *= d
    tot += n
print('tensor_count', len(tsz), 'total_elems', tot)
import re
for nm in sorted(tsz):
    if re.search(r'blk[.]0[.]|blk[.]64[.]|token_embd|output', nm):
        print('   T', nm, tsz[nm][0], tsz[nm][1])