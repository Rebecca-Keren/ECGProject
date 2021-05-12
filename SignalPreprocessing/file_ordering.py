import os
import data_preprocess_function

def renaming(data_path):
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        filename = filename[len("Fecg_01_"):]
        file_path_new = os.path.join(data_path, filename)
        os.renames(file_path,file_path_new)

def division_of_patient(dictionary,data_path):
    for filename in os.listdir(data_path):
        index = int(filename[2:4])
        file_path = os.path.join(data_path, filename)
        if index in dictionary.keys():
            dictionary[index].append(file_path)
        else:
            dictionary[index] = [file_path]
    return dictionary

def creation_of_graphs(dictionary,list,main_path):
    for key in dictionary.keys():
        directory_path = os.path.join(main_path,str(key))
        try:
            os.mkdir(directory_path)
        except:
            pass
        for elem in dictionary[key]:
            for type in list:
                data_preprocess_function.transformation(type, elem, directory_path)
    return






