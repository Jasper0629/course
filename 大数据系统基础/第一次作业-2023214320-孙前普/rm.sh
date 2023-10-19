for i in 3 4 5 
do 
{
    ssh thumm0"$i" "rm -rf ./*;exit" 
}&
done