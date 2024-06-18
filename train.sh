set -x

for seed in 1235 1236 1237 1238 1239
do
  for rpe in alibi rotary t5 kerple_log ali_ali05_gaussian_all003_fp32 t5_kerplelog_gau_alibi
  do
    echo $rpe
    #for dataset in arxiv github openwebtext2
    for dataset in arxiv
    do
      python ./deepy.py 1.5 train.py mep_configs/local_setup.yml mep_configs/train.yml mep_configs/exp_configs/"$rpe"_"$dataset"_"$seed".yml
    done
  done
done
