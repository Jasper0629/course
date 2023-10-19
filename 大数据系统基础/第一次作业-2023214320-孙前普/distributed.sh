for j in 3 4 5
do
    scp wc_dataset_distributed.txt thumm0"$j":~/wc_dataset_distributed.txt
done


# start=$(date +%s)
# cat wc_dataset_joint.txt | tr -s ' ' '\n' | sort | uniq -c | sort -rnk1 | awk '{print $2,$1}'
# end=$(date +%s)
# diff=$((end - start))
# echo "时间差为 $diff 秒"