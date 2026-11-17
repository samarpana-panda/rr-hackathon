cd /home/claude/final_nrr
python3 reliability.py '{"dir":"/home/claude/final_nrr/v8/ipl"}' > b1.log 2>&1 &
python3 reliability.py '{"dir":"/home/claude/final_nrr/v6/ipl"}' > b2.log 2>&1
wait
python3 reliability.py '{"dir":"/home/claude/final_nrr/v8/intl"}' > b3.log 2>&1 &
python3 reliability.py '{"dir":"/home/claude/final_nrr/v6/intl"}' > b4.log 2>&1
wait
python3 reliability.py '{"dir":"/home/claude/final_nrr/v8/alevel","big_neff":15,"min_hp_balls":20,"min_hp_matches":2}' > b5.log 2>&1 &
python3 reliability.py '{"dir":"/home/claude/final_nrr/v6/alevel","big_neff":15,"min_hp_balls":20,"min_hp_matches":2}' > b6.log 2>&1
wait
echo BOOTDONE
