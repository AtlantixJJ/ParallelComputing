#scp -P 19917 -r * 2015011313@166.111.227.244:hw5/
tar cvfz - src script *.sh | ssh -p 8081 atlantix@166.111.17.31 "cd hw5/; tar xvfz -"
