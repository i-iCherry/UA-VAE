"""
Contain some omics data preprocess functions
"""
import pandas as pd
from util import utils
import tensorflow as tf
import torch


def separate_B(B_df_single):
    """
    Separate the DNA methylation dataframe into subsets according to their targeting chromosomes

    Parameters:
        B_df_single(DataFrame) -- a dataframe that contains the single DNA methylation matrix

    Return:
        B_df_list(list) -- a list with 23 subset dataframe
        B_dim(list) -- the dims of each chromosome
    """

    anno = pd.read_csv('./anno/B_anno.csv', sep="\t", index_col=0)  # dtype={'chrom': str},
    anno_contain = anno.loc[B_df_single.index, :]
    print('Separating B.tsv according the targeting chromosome...')
    B_df_list, B_dim_list = [], []
    ch_id = list(range(1, 23))
    ch_id.append('X')
    for ch in ch_id:
        ch_index = anno_contain[anno_contain.CHR == str(ch)].index
        ch_df = B_df_single.loc[ch_index, :]

        # multi-head-attention
        d_model = len(ch_df)
        num_heads = 1
        # multihead_att = utils.MultiHeadAttention(d_model, num_heads)
        # ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # mha_out, mha_att_matrix = multihead_att(ch_df, ch_df, ch_df)
        # ch_df= ln(ch_df + mha_out)


        B_df_list.append(ch_df)
        B_dim_list.append(len(ch_df))

    return B_df_list, B_dim_list
