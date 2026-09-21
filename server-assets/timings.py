import json, glob, os, sys
for tag in sys.argv[1:]:
    for i in (1,2,3):
        f = '/tmp/p60-%s-p%d.json' % (tag, i)
        if not os.path.exists(f):
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            print(tag, i, 'PARSE_FAIL', e); continue
        t = d.get('timings', {})
        print(tag, 'p%d' % i,
              'pred_n=%s' % t.get('predicted_n'),
              'pred_ms=%s' % t.get('predicted_ms'),
              'tps=%.2f' % (t.get('predicted_per_second') or 0),
              'draft_n=%s' % t.get('draft_n'),
              'draft_acc=%s' % t.get('draft_n_accepted'),
              'prompt_n=%s' % t.get('prompt_n'))
