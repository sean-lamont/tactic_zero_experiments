from new_env import *
import json
import pickle
import generate_gnn_data

with open("dataset.json") as fp:
    dataset = json.load(fp)

env = HolEnv("T")
paper_goals = []

for goal in dataset:
    try:
        p_goal = env.get_polish(goal)
        paper_goals.append((p_goal[0]["polished"]['goal'], goal))
    except:
        print (goal)

with open("dep_data.json") as fp:
    deps = json.load(fp)

with open("new_db.json") as fp:
    full_db = json.load(fp)

unique_thms = list(set(deps.keys()))

paper_goals_polished = [g[0] for g in paper_goals]

#get all theorems from paper dataset compatible with current database 

exp_thms = []

for thm in unique_thms:
    if full_db[thm][2] in paper_goals_polished:
        exp_thms.append(thm)

#remove theorems the RL agent trains/tests on from those used to pretrain the GNN encoder
gnn_encoder_set = list(set(unique_thms) - set(exp_thms))
        
train_thms = exp_thms[:int(0.8 * len(exp_thms))]
test_thms = exp_thms[int(0.8 * len(exp_thms)):]


#generate gnn data from valid set excluding goals for RL agent

generate_gnn_data.generate_gnn_data(gnn_encoder_set, 0.95, 0.05, True, "")


#todo make batch_gnn similar to model class with .train, taking in data directory
import batch_gnn
batch_gnn.run(1e-3, 0, 20, 1024, 64, 2, True)



with open("paper_goals.pk", "wb") as f:
    pickle.dump(paper_goals, f)

