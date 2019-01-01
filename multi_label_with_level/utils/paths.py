'''
A class used to record and edit paths
'''
class Path():
    ml_training_data_path = '../training_data/ml_training_data/'
    techone_training_data_path = '../training_data/techone_new_training_data/'

    @staticmethod
    def techone_training_data():
        return Path.techone_training_data_path
    