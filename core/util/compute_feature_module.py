#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
# %%

import os
from tqdm import tqdm
import pickle
# %matplotlib inline

from argoverse_loader.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse_loader.map_representation.map_api import ArgoverseMap

from core.util.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from core.util.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR


if __name__ == "__main__":
    am = ArgoverseMap()
    for folder in os.listdir(DATA_DIR):

        # FIXME: modify the target folder by hand ('val|train|sample|test')
        if "DS_Store" in folder: continue

        print(f"folder: {folder}")
        afl = ArgoverseForecastingLoader(os.path.join(DATA_DIR, folder))
        norm_center_dict = {}
        for name in tqdm(afl.seq_list):
            afl_ = afl.get(name)
            path, name = os.path.split(name)
            name, ext = os.path.splitext(name)

            agent_feature, obj_feature_ls, lane_feature_ls, norm_center = compute_feature_for_one_seq(
                afl_.seq_df, am, OBS_LEN, LANE_RADIUS, OBJ_RADIUS, viz=False, mode='nearby'
             )
            df = encoding_features(agent_feature, obj_feature_ls, lane_feature_ls)
            save_features(df, name, os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))

            norm_center_dict[name] = norm_center

        with open(os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}-norm_center_dict.pkl"), 'wb') as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)
            # print(pd.DataFrame(df['POLYLINE_FEATURES'].values[0]).describe())


