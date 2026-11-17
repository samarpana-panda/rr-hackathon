set -e
cd /home/claude/final_nrr
t(){ echo "[$(date +%H:%M:%S)] $*"; }
t stakes;      python3 stakes_nrr.py /home/claude/data/pack /home/claude/final_run/v8/balls_pressure.parquet /home/claude/final_nrr/stakes.csv > stakes.log 2>&1
t pressure;    python3 pressure.py > pressure.log 2>&1
t consequence; mkdir -p v4/consequence; LP_OUT=v4/consequence LP_FEATS='[0,1,2,3]' python3 learn_pressure.py > consequence.log 2>&1
t difficulty;  python3 difficulty.py > difficulty.log 2>&1
t blend;       python3 blend.py '{"out":"/home/claude/final_nrr/v6"}' > blend.log 2>&1
t superover;   python3 superover.py > superover.log 2>&1
lists(){ B=$1; O=$2; X=$3
 python3 perf.py "{\"balls\":\"$B\",\"out\":\"$O/ipl\",\"groups\":[\"IPL\"],\"min_hp_balls\":60,\"min_hp_matches\":5$X}" > $O.ipl.log 2>&1 &
 python3 perf.py "{\"balls\":\"$B\",\"out\":\"$O/intl\",\"groups\":[\"T20I_full\",\"T20I_assoc\"],\"min_hp_balls\":60,\"min_hp_matches\":5$X}" > $O.intl.log 2>&1
 wait
 python3 perf.py "{\"balls\":\"$B\",\"out\":\"$O/alevel\",\"groups\":[\"A_teams\"],\"min_hp_balls\":20,\"min_hp_matches\":2,\"big_neff\":15$X}" > $O.alevel.log 2>&1; }
for v in v8 v6; do lists $v/balls_pressure.parquet $v ""; t ratings $v; done
bash run_boot.sh > boot.log 2>&1; t boot
python3 /home/claude/work/avail.py /home/claude/final_nrr /home/claude/final_nrr/final_available.csv > avail.log 2>&1; t avail
# ---- sensitivity variants (with super overs)
mkdir -p sens
for a in 0.55 0.75; do python3 blend.py "{\"alpha\":$a,\"out\":\"/home/claude/final_nrr/sens/a$a/v6\",\"report\":false}" > sens/a$a.blend.log 2>&1
  SO_IN=sens/a$a/v6 SO_OUT=sens/a$a/v8 python3 superover.py > /dev/null 2>&1; lists sens/a$a/v8/balls_pressure.parquet sens/a$a/v8 ""; t variant alpha $a; done
for g in 0.25 0.5; do python3 blend.py "{\"gamma\":$g,\"out\":\"/home/claude/final_nrr/sens/g$g/v6\",\"report\":false}" > sens/g$g.blend.log 2>&1
  SO_IN=sens/g$g/v6 SO_OUT=sens/g$g/v8 python3 superover.py > /dev/null 2>&1; lists sens/g$g/v8/balls_pressure.parquet sens/g$g/v8 ""; t variant gamma $g; done
for q in 0.8 0.7; do mkdir -p sens/q$q; lists v8/balls_pressure.parquet sens/q$q/v8 ",\"hp_q\":$q"; t variant hpq $q; done
t ALLDONE
