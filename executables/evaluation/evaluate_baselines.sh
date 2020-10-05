export PYTHONPATH=../../src

start_time=$(date)
echo "[START] ${start_time}"

# 0. Dumani et al.
method=dumani
for column in first512Tokens slidingWindow sentences; do
  echo "$method $column"
  python3 evaluate_baseline.py --method dumani_$method --column $column >/dev/null
done

# 1. Baseline
method=zero_shot_knn
for similarity in l2 l1 cos; do
  echo "$method $similarity"
  python3 evaluate_baseline.py --method $method --similarity $similarity >/dev/null
done

# 2. Baseline
method=zero_shot_cluster_knn
for cluster_ratio in 0.25 0.5 1.0; do
  for cluster_representative in closest-to-center closest-to-claim; do
    for similarity in l2 l1 cos; do
      echo "$method $cluster_ratio $similarity $cluster_representative"
      python3 evaluate_baseline.py --method $method --similarity $similarity --cluster_ratio $cluster_ratio --cluster_representative $cluster_representative >/dev/null
    done
  done
done

# 3. Baseline
method=learned_similarity_knn
for softmax in True False; do
  echo "$method $softmax"
  python3 evaluate_baseline.py --method $method --softmax $softmax >/dev/null
done

# 4. Baseline
method=learned_similarity_cluster_knn
for softmax in True False; do
  for cluster_ratio in 0.25 0.5 1.0; do
    echo "$method $softmax $cluster_ratio"
    python3 evaluate_baseline.py --method $method --cluster_ratio $cluster_ratio --softmax $softmax >/dev/null
  done
done

end_time=$(date)
echo "[DONE] ${start_time} (started ${end_time})"
