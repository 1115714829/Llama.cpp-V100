cd /root
CSV=ncu-decode.csv
echo "=== aggregate by class (total duration ns over 1500 launches) ==="
awk -F'","' '
  /"Kernel Name"/ { next }
  {
    if (NF < 15) next
    name=$5; dur=$NF
    gsub(/"/,"",name); gsub(/"/,"",dur)
    if (dur !~ /^[0-9]+(\.[0-9]+)?$/) next
    d=dur+0
    if (name ~ /mmq/) cls="MMQ(FFN)"
    else if (name ~ /gated_delta_net/) cls="GDN"
    else if (name ~ /gated_linear_attn/) cls="GLA"
    else if (name ~ /flash_attn|fattn/) cls="FA"
    else if (name ~ /mul_mat|gemv|vec_dot/) cls="other-mm"
    else cls="other"
    total[cls]+=d; cnt[cls]++
  }
  END { for (k in total) printf "%-14s %16.0f ns  (%d kernels)\n", k, total[k], cnt[k] }
' $CSV | sort -k2 -rn
echo "=== top individual kernels (single launch, ns) ==="
awk -F'","' '
  /"Kernel Name"/ { next }
  { if (NF<15) next; name=$5; dur=$NF; gsub(/"/,"",name); gsub(/"/,"",dur);
    if (dur !~ /^[0-9]+(\.[0-9]+)?$/) next; printf "%12.0f ns  %s\n", dur+0, substr(name,1,58) }
' $CSV | sort -rn | head -14
echo "PARSE_DONE"
