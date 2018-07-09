for SEED in 1; do
 for GF in 0.05; do # 0.1 0.25; do
  for CPROB in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
   #
   echo "${GF}, ${CPROB} ${SEED}"
   export GF CPROB SEED
   #
   sbatch wrn.sbatch
   #
  done
 done
done
