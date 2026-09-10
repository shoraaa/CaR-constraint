#!/bin/bash
# status.sh -- one-screen view of the training jobs on THIS machine.
#
#   ./status.sh            table + recent validation curves for every job here
#   ./status.sh 485        also print each job's row at exactly epoch 485
#   ./status.sh -n 20      show 20 curve points instead of 8
#
# Jobs are discovered from the running processes, not from a hardcoded layout:
# the arm and solver come from each process's own command line and its log_dir
# from the same place, so this stays correct however the jobs are spread across
# machines.  Run it on each machine and read a matched epoch off the outputs.
#
# Validation numbers come from the TensorBoard event files via read_val.py, NOT
# from the nohup log.  Python block-buffers stdout at 8 KB when redirected and
# this trainer prints little per epoch, so a log can sit HOURS behind the run --
# M3's had 0 "Val Score" lines after 3.5 hours of training.  The event files are
# flushed every epoch, so what you see here is current.
#
# Rate and ETA are measured from checkpoint mtimes in the current run directory,
# so they reflect what the machine is doing now rather than a stale log estimate
# carried over from a previous GPU-sharing arrangement.

cd "$(dirname "$(readlink -f "$0")")" || exit 1

N=8
if [ "$1" = "-n" ]; then N="$2"; shift 2; fi
WANT="$*"

ckpts() { ls "$1"/*/epoch-*.pt 2>/dev/null | sed 's/.*epoch-//;s/\.pt//' | sort -n; }

newest_age() {
  local f; f=$(ls -t "$1"/*/epoch-*.pt 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "-"; return; }
  local s=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if   [ $s -lt 3600 ]  ; then echo "$((s/60))m"
  elif [ $s -lt 86400 ] ; then echo "$((s/3600))h$(( (s%3600)/60 ))m"
  else echo "$((s/86400))d$(( (s%86400)/3600 ))h"; fi
}

# minutes per epoch over the LAST few checkpoints of the current run directory.
# Deliberately not the whole run: these jobs get moved between machines and
# change GPU neighbours, so a lifetime average would report a speed the machine
# is no longer running at.
rate() {
  local d; d=$(ls -td "$1"/*/ 2>/dev/null | head -1)
  [ -n "$d" ] || { echo "-"; return; }
  local list; list=$(ls "$d"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-//;s/\.pt//' | sort -n | tail -6)
  [ "$(echo "$list" | grep -c .)" -ge 2 ] || { echo "-"; return; }
  local e1 e2 t1 t2
  e1=$(echo "$list" | head -1); e2=$(echo "$list" | tail -1)
  t1=$(stat -c %Y "$d/epoch-$e1.pt"); t2=$(stat -c %Y "$d/epoch-$e2.pt")
  awk -v t1="$t1" -v t2="$t2" -v e1="$e1" -v e2="$e2" \
      'BEGIN { printf "%.1f", (t2 - t1) / 60 / (e2 - e1) }'
}

eta() {  # $1 = min/epoch, $2 = current epoch
  awk -v r="$1" -v e="$2" 'BEGIN {
    if (r == "-" || r + 0 <= 0 || e == "") { print "-"; exit }
    h = (1000 - e) * r / 60
    if (h < 48) printf "%.1fh", h; else printf "%.1fd", h / 24
  }'
}

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# ---- discover live jobs from their own command lines -------------------------
LIVE_DIRS=""
i=0
for pid in $(pgrep -f "trai[n].py" 2>/dev/null); do
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
  [ -n "$cmd" ] || continue
  repr=$(echo "$cmd" | grep -o -- '--constraint_repr [a-z]*' | head -1 | awk '{print $2}')
  imp=$( echo "$cmd" | grep -o -- '--improve_steps [0-9]*'   | head -1 | awk '{print $2}')
  ld=$(  echo "$cmd" | grep -o -- '--log_dir [^ ]*'          | head -1 | awk '{print $2}')
  [ -n "$ld" ] || continue
  solver=POMO; [ "${imp:-0}" != "0" ] && solver=CaR
  i=$((i + 1))
  python3 read_val.py "$ld" > "$TMP/job$i" 2>/dev/null
  echo "$solver ${repr:-?}|$ld|$TMP/job$i" >> "$TMP/index"
  LIVE_DIRS="$LIVE_DIRS $ld"
done

echo "=================================================================="
echo " CaR-constraint  --  $(hostname -I | awk '{print $1}')"
echo " $(date -u '+%Y-%m-%d %H:%M UTC')   live train.py: $(pgrep -f 'trai[n].py' | wc -l)   evals: $(pgrep -f 'evaluate_al[l].py' | wc -l)"
echo "=================================================================="
echo
printf " %-15s %6s %7s %8s %8s %10s %10s %9s\n" \
  job ckpt age min/ep "eta1000" "gap(mask)" "gap(cons)" infeas

if [ -f "$TMP/index" ]; then
  while IFS='|' read -r name ld f; do
    last=$(tail -1 "$f"); ce=$(ckpts "$ld" | tail -1); r=$(rate "$ld")
    printf " %-15s %6s %7s %8s %8s %9s%% %9s%% %8s%%\n" "$name" "${ce:--}" \
      "$(newest_age "$ld")" "$r" "$(eta "$r" "$ce")" \
      "$(echo "$last" | awk '{print ($2 == "" ? "-" : $2)}')" \
      "$(echo "$last" | awk '{print ($3 == "" ? "-" : $3)}')" \
      "$(echo "$last" | awk '{print ($4 == "" ? "-" : $4)}')"
    le=$(echo "$last" | awk '{print $1}')
    [ -n "$ce" ] && [ -n "$le" ] && [ "$ce" -gt "$le" ] 2>/dev/null &&
      echo "   ($name: checkpoint $ce is ahead of the last validated epoch $le)"
  done < "$TMP/index"
fi

# ---- run directories with checkpoints but no live process --------------------
for ld in results/pomo1000/train_* results/car1000/train_*; do
  [ -d "$ld" ] || continue
  case " $LIVE_DIRS " in *" $ld "*) continue ;; esac
  n=$(ckpts "$ld" | tail -1); [ -n "$n" ] || continue
  printf " %-15s %6s   -- checkpoints only, job runs elsewhere (see server.md) --\n" \
    "$(echo "$ld" | sed 's#.*/train_pomo_soft_#POMO #;s#.*/train_#CaR #')" "$n"
done

[ -f "$TMP/index" ] || { echo; echo " no training jobs running on this machine"; exit 0; }

echo
echo " validation curve  (epoch:masked-gap%)   -- masked pass is the comparable metric"
while IFS='|' read -r name ld f; do
  [ -s "$f" ] || continue
  printf "  %-15s " "$name"; tail -n "$N" "$f" | awk '{printf "%s:%s  ", $1, $2}'; echo
done < "$TMP/index"

echo
echo " soft-construction infeasibility (epoch:%)  -- pre-mask quality, not the objective"
while IFS='|' read -r name ld f; do
  [ -s "$f" ] || continue
  printf "  %-15s " "$name"; tail -n "$N" "$f" | awk '{printf "%s:%s  ", $1, $4}'; echo
done < "$TMP/index"

# ---- exact-epoch lookup for cross-machine matching ---------------------------
if [ -n "$WANT" ]; then
  echo
  echo " requested epochs:"
  for e in $WANT; do
    while IFS='|' read -r name ld f; do
      hit=$(awk -v e="$e" '$1 == e' "$f")
      if [ -n "$hit" ]; then
        echo "$hit" | awk -v j="$name" '{printf "  %-15s epoch %-5s masked %s%%  constr %s%%  infeas %s%%\n", j, $1, $2, $3, $4}'
      else
        near=$(awk -v e="$e" '$1 <= e' "$f" | tail -1 | awk '{print $1}')
        printf "  %-15s epoch %-5s no validation (interval 5); nearest below: %s\n" "$name" "$e" "${near:-none}"
      fi
      ls "$ld"/*/epoch-"$e".pt >/dev/null 2>&1 ||
        printf "  %-15s !! no checkpoint epoch-%s.pt here -- matched_eval.sh would fail\n" "" "$e"
    done < "$TMP/index"
  done
fi

echo
echo " checkpoints on disk:"
while IFS='|' read -r name ld f; do
  printf "  %-15s %s .. %s   (%d files)\n" "$name" \
    "$(ckpts "$ld" | head -1)" "$(ckpts "$ld" | tail -1)" "$(ckpts "$ld" | wc -l)"
done < "$TMP/index"
