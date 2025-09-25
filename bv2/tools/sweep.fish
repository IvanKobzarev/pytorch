#!/usr/bin/env fish

function runit
  set -l gpu $argv[1]
  set -l port $argv[2]
  set -l name $argv[3]
  set -l extra $argv[4..-1]
  echo "Logs in /tmp/sweep-$name for run bv2.train --name $name $extra" >&2
  set -l extras (string join ' ' $extras)
  env CUDA_VISIBLE_DEVICES=$gpu with-proxy torchrun --nproc_per_node=gpu --master_port=295$port$gpu -m bv2.train --name $name $extra &> /tmp/sweep-$name &
  echo $last_pid
end

set basename (whoami)-(date +"%Y%m%d-%H%M%S")

# Use like this, will run "eagerly" i.e. all runs simultaneously:
# First number is the GPU, second one the master port index.
# runit 4 0 "$basename-test1"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.7071
# runit 5 0 "$basename-test2"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 7.0710
# runit 4 1 "$basename-.3"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.2121
# runit 5 1 "$basename-.03"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.02121

# Use like this to queue, with each `wait` being a (full) sync point.
# set -l pid4 (runit 4 0 "$basename-test1"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.7071)
# set -l pid5 (runit 5 0 "$basename-test2"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 7.0710)
# wait $pid4 $pid5
# set -l pid4 (runit 4 0 "$basename-.3"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.2121)
# set -l pid5 (runit 5 0 "$basename-.03"  --nsteps 1000 --warmup_nsteps 100 --lr 0.0001 --lbinit 0.02121)
