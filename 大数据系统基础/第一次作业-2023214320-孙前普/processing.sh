i=$1        
cat wc_dataset_distributed.txt | tr -s '\n' | sort | uniq -c | sort -rnk1 | awk '{print $2,$1}' | tee log"$i".txt

