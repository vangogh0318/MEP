set -x

#sleep 6h

for seed in 1235 1236 1237 1238 1239
do
  for rpe in alibi t5 kerple_log
  do
    for dataset in arxiv github openwebtext2
    do
      for length in 512 1024 2048 4096 8192
      do
        echo "my_rpe:$rpe"
        echo "my_length:$length"
        echo "my_seed:$seed"
        echo "$rpe"_"$dataset"_"$seed".yml mep_configs/lengths/length_"$length".yml
        python ./deepy.py 1.5 train.py mep_configs/local_setup.yml mep_configs/ex_eval.yml mep_configs/exp_configs/"$rpe"_"$dataset"_"$seed".yml mep_configs/lengths/length_"$length".yml
      done
    done
  done
done
