# About:    superclass for data preprocessor
# Author:   Jianbang LIU
# Date:     2021.01.30

import os
import pandas as pd


class Preprocessor(object):
    """
    superclass for all the trajectory data preprocessor
    those preprocessor will reformat the data in a single sequence and feed to the system or store them
    """
    def __init__(self, root_dir, algo="tnt", obs_horizon=20, obs_range=30):
        self.root_dir = root_dir    # root directory stored the dataset

        self.algo = algo            # the name of the algorithm
        self.obs_horizon = 20       # the number of timestampe for observation
        self.obs_range = 30         # the observation range

    def __len__(self):
        """ the total number of sequence in the dataset """
        raise NotImplementedError

    def generate(self):
        """ Generator function to iterating the dataset """
        raise NotImplementedError

    def select(self, dataframe: pd.DataFrame, map_feat=True):
        """
        select filter the data frame, output filtered data frame
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def encode(self, dataframe: pd.DataFrame):
        """
        encode the filtered data to specific format required by the algorithm
        :param dataframe: DataFrame, the data frame containing the filtered data
        :return: DataFrame[agent_pl_feat, agent_gt, obj_pl_feat, obj_id_mask, map_pl_feat, map_pl_id_mask]
        """
        raise NotImplementedError

    def save(self, dataframe: pd.DataFrame, set_name, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(dataframe, pd.DataFrame):
            return

        if not dir_:
            dir_ = os.path.join(self.root_dir, set_name, "processed")
        else:
            dir_ = os.path.join(dir_, set_name, "processed")
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = f"features_{file_name}.csv"
        dataframe.to_csv(os.path.join(dir_, fname))


# example of preprocessing scripts
if __name__ == "__main__":
    processor = Preprocessor("raw_data")

    for s_name, f_name, df in processor.generate():
        df = processor.select(df)
        encoded_df = processor.encode(df)
        processor.save(encoded_df, s_name, f_name)
