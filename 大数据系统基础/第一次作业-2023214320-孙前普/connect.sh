for loop in 3 4 5
do 
    sshpass -p '2023214320'  scp ~/.ssh/id_rsa.pub thumm0$loop:~/.ssh/authorized_keys
    echo "thumm0$loop done"
done