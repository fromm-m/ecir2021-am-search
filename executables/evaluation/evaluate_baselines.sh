export PYTHONPATH=../../src

for method in zero_shot_knn zero_shot_cluster_knn; do
  for similarity in l2 l1 cos; do
    echo "$method $similarity"
    python3 evaluate_baseline.py --method $method --similarity $similarity
  done
done
