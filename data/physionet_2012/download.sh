mkdir train
wget http://alpha.physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz -O train_a.tar.gz
wget http://alpha.physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt -O train_a_labels.txt
tar -C train --strip-components=1 -xzf train_a.tar.gz set-a/
rm train_a.tar.gz

wget http://alpha.physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz -O train_b.tar.gz
wget http://alpha.physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt -O train_b_labels.txt
tar -C train --strip-components=1 -xzf train_b.tar.gz set-b/
rm train_b.tar.gz

cat train_a_labels.txt > train_labels.txt
tail -n +2 train_b_labels.txt >> train_labels.txt
rm train_a_labels.txt train_b_labels.txt

mkdir test
wget http://alpha.physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz -O test.tar.gz
wget http://alpha.physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt -O test_listfile.csv
tar -C test --strip-components=1 -xzf test.tar.gz set-c/
rm test.tar.gz
