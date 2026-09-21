#!/usr/bin/env python3
import socket, struct, time, re, sys, os, select

OUT = "/root/sniff8000.log"
DUR = 3600.0
for a in sys.argv[1:]:
    if re.fullmatch(r'[0-9.]+', a): DUR = float(a)

rx_mark = re.compile(rb'EFFORT[-_](?:XHIGH|MEDIUM|LOW)[-_]?\d*', re.I)
rx_eff  = re.compile(rb'"reasoning_effort"\s*:\s*"([^"]{0,32})"')
rx_post = re.compile(rb'POST (/v1/\S+) HTTP/1\.1')
rx_cl   = re.compile(rb'Content-Length: (\d+)')
rx_mdl  = re.compile(rb'"model"\s*:\s*"([^"]{0,64})"')
BUFMAX  = 4000000

class State:
    def __init__(self, f):
        self.f=f; self.buf={}; self.nmark={}; self.pending=[]
    def feed(self, k, pl):
        try:
            b=self.buf.setdefault(k, bytearray()); b.extend(pl)
            if len(b)>BUFMAX: del b[:BUFMAX//2]
            bk=bytes(b)
            ms=list(rx_mark.finditer(bk))
            pn=self.nmark.get(k,0)
            if len(ms)>pn:
                for i in range(pn,len(ms)):
                    self.pending.append([k, ms[i].end(), 0])
                self.nmark[k]=len(ms)
            keep=[]
            for p in self.pending:
                if p[2]: continue
                cur=bytes(self.buf.get(p[0], b""))
                if len(cur)<=p[1]:
                    keep.append(p); continue
                tail=cur[p[1]:p[1]+4000]
                em=rx_eff.search(tail)
                if em is None:
                    keep.append(p); continue
                p[2]=1
                mk=rx_mark.search(cur, max(0,p[1]-300))
                ps=list(rx_post.finditer(cur)); cl=list(rx_cl.finditer(cur)); md=rx_mdl.search(cur)
                self.f.write("\n[%s] 来源 %s:%d\n" % (time.strftime("%H:%M:%S"), p[0][0], p[0][1]))
                self.f.write("     标记 = %s\n" % (mk.group(0).decode("latin1","replace") if mk else "?"))
                self.f.write("     >>> reasoning_effort = %s\n" % em.group(1).decode("latin1","replace"))
                self.f.write("     model=%s 本连接累计POST=%d Content-Length=%s\n" % (
                    md.group(1).decode("latin1","replace") if md else "?",
                    len(ps), cl[-1].group(1).decode() if cl else "-"))
            self.pending=keep[-40:]
        except Exception as e:
            try: self.f.write("[ERR] %r\n" % (e,))
            except Exception: pass

def selftest():
    import io
    f=io.StringIO(); st=State(f)
    body=(b'{"model":"Qwen3.8-27B-FP8","messages":[{"role":"system","content":"You are Codex, a coding agent."},'
          b'{"role":"user","content":"EFFORT-MEDIUM-222"}],"stream":true,"reasoning_effort":"medium"}')
    for i in range(0,len(body),37):
        st.feed(("127.0.0.1",12345), body[i:i+37])
    out=f.getvalue()
    ok = ("EFFORT-MEDIUM-222" in out or "EFFORT-MEDIUM" in out) and ("medium" in out)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    print(out)

def main():
    socks={}
    for iface in sorted(os.listdir('/sys/class/net')):
        try:
            s=socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32*1024*1024)
            s.bind((iface,0)); s.setblocking(False); socks[s]=iface
        except Exception: pass
    f=open(OUT,"a",buffering=1)
    st=State(f)
    f.write("=== 监听 v9 开始 %s 时长=%ss 网卡=%s ===\n"%(time.strftime("%H:%M:%S"),DUR,",".join(sorted(socks.values()))))
    recent={}; lastgc=time.time(); lastbeat=time.time()
    end=time.time()+DUR
    while time.time()<end:
        try: rl,_,_=select.select(list(socks),[],[])
        except Exception: break
        now=time.time()
        if now-lastgc>30: recent={kk:v for kk,v in recent.items() if now-v<5}; lastgc=now
        if now-lastbeat>300:
            f.write("[%s] (心跳 存活, 已配对%d条)\n"%(time.strftime("%H:%M:%S"), st.nmark and sum(st.nmark.values()) or 0)); lastbeat=now
        for s in rl:
            try: pkt=s.recv(65536)
            except Exception: continue
            if len(pkt)<34: continue
            if struct.unpack("!H",pkt[12:14])[0]!=0x0800: continue
            ip=pkt[14:]
            if len(ip)<20 or ip[9]!=6: continue
            src=".".join(str(x) for x in ip[12:16])
            ihl=(ip[0]&0x0f)*4; tcp=ip[ihl:]
            if len(tcp)<20: continue
            sport,dport=struct.unpack("!HH",tcp[0:4])
            if dport!=8000: continue
            seq=struct.unpack("!I",tcp[4:8])[0]
            doff=(tcp[12]>>4)*4; pl=tcp[doff:]
            if not pl: continue
            dk=(sport,seq,len(pl))
            if dk in recent: continue
            recent[dk]=now
            st.feed((src,sport), pl)

if __name__=="__main__":
    if "--selftest" in sys.argv: selftest()
    else: main()
