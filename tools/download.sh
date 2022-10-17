cd data/scienceqa/images

wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip

unzip -q train.zip
unzip -q val.zip
unzip -q test.zip

rm train.zip
rm val.zip
rm test.zip

cd ../../..
