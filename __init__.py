from train import train
from tag import tag


train(test_ratio=0.9, transliterate=True, epochs=100)
#train(onlytesting="./data/output/testing", fulltest=True)