export PYTHONPATH=../../src

time=$(date)
echo "[START] ${time}"

# 1. Baseline
method=zero_shot_knn
for similarity in l2 l1 cos; do
  echo "$method $similarity"
  python3 evaluate_baseline.py --method $method --similarity $similarity > /dev/null
done

# 2. Baseline
method=zero_shot_cluster_knn
for cluster_ratio in 0.25 0.5 1.0; do
  for cluster_representative in closest-to-center closest-to-claim; do
    for similarity in l2 l1 cos; do
      echo "$method $cluster_ratio $similarity $cluster_representative"
      python3 evaluate_baseline.py --method $method --similarity $similarity --cluster_ratio=$cluster_ratio --cluster_representative=$cluster_representative  > /dev/null
    done
  done
done

time=$(date)
echo "[DONE] ${time}"