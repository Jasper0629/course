
start=$(date +%s)
for i in 1 3 4 5 
do 
{   
    if (($i==1))
    then
        echo "thumm0"$i" is the master node"
        bash processing.sh $i
    else
        echo "thumm0"$i" is the slave node"
        scp wc_dataset_distributed.txt processing.sh thumm0"$i":~/ 
        ssh thumm0"$i" "bash processing.sh $i;exit" 
        scp thumm0"$i":~/log"$i".txt log"$i".txt 
    fi

}&
done
wait
cat log1.txt log3.txt log4.txt log5.txt | awk '{sum[$1]+=$2}END{for(c in sum){print c,sum[c]}}' | sort -ur | tee log.txt
end=$(date +%s)
diff=$((end - start))
echo "时间差为 $diff 秒"
