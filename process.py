import json
from copy import deepcopy as cpy

with open("data/original/train.json","r") as f:
    data=json.load(f)
    
final_data=[]

for i in data:
    temp=cpy(i)
    temp["emotion-cause_pairs"]=[]
    
    for j in i["emotion-cause_pairs"]:
        temp_={}
        temp_["cause_id"],temp_["cause_text"]=j[1].split("_")  
        temp_["target_id"],temp_["target_temotion"]=j[0].split("_")   
        
        temp_["cause_text"]=temp_["cause_text"].lower()
    
        temp["emotion-cause_pairs"].append(temp_)
        
    for idx in range(len(temp["conversation"])):
        temp["conversation"][idx]["text"]=temp["conversation"][idx]["text"].lower()
    
    final_data.append(temp)
    
    
import random
random.shuffle(final_data)

TEST_SIZE=0.2
VAL_SIZE=0.1
TRAIN_SIZE=1-TEST_SIZE-VAL_SIZE

size=len(final_data)


train=final_data[:int(TRAIN_SIZE*size)]
val=final_data[int(TRAIN_SIZE*size):int((TRAIN_SIZE+VAL_SIZE)*size)]
test=final_data[int((TRAIN_SIZE+VAL_SIZE)*size):]

    
with open("data/train.json","w") as f:
    json.dump(train,f,indent=4)
    
with open("data/val.json","w") as f:
    json.dump(val,f,indent=4)
    
with open("data/test.json","w") as f:
    json.dump(test,f,indent=4)

