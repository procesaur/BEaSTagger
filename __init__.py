from beast.train import train
from beast.tag import tag


#train(file_path="beast/data/training/SrpKor4Tagging", test_ratio=0.9)
train(onlytesting="testing")
#train(onlytesting="./data/output/testing", fulltest=True)
#tag(model="./data/output/final", transliterate=True)