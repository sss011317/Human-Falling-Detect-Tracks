import os
import pickle
save_path = '../Data/Home_new-set(labelXscrw).pkl'
# save_path = '../Data/Home_new-set(labelXscrw)_jump.pkl'
save_path_in = open(save_path,"rb")
Model = pickle.load(save_path_in)
# torch.save(Model, 'Save_File_Name.')
print(Model)