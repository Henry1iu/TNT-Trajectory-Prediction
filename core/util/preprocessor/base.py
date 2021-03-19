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

    def process(self, dataframe: pd.DataFrame, map_feat=True):
        """
        select filter the data frame, output filtered data frame
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """

        agent_feats, obj_feats, lane_feats = self.extract_feature(dataframe, map_feat=map_feat)
        return self.encode_feature(agent_feats, obj_feats, lane_feats)

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        """
        select and filter the data frame, output filtered frame feature
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def encode_feature(self, *feats):
        """
        encode the filtered features to specific format required by the algorithm
        :feats dataframe: DataFrame, the data frame containing the filtered data
        :return: DataFrame[POLYLINE_FEATURES, GT, TRAJ_ID_TO_MASK, LANE_ID_TO_MASK, TARJ_LEN, LANE_LEN]
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
        processor.save(processor.process(df), s_name, f_name)
