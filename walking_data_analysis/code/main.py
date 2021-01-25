from Data_Filter import get_Gacc_0, split_by_time, split_by_t, FFT_show, step_fre, step_len, get_speed_ML, train_speed, low_pass, plot_xyz, plot_xyz_L, test1
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def report():
    file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    # file = "sensor_data/complex_road/2020-08-0216.10.53.csv"

    df = pd.read_csv(file)
    df = split_by_t(df, 20)
    low_pass_df = df.apply(low_pass)
    FFT_show(low_pass_df, "gFy", 6, 0.15)

    test1(file)

    model = train_speed()
    df = split_by_t(df, 20)
    df = get_speed_ML(df, model)

    print("\nstep_freq", step_fre(df))
    print("step_len", step_len(df))


def stat_sample():
    file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    file1 = "sensor_data/down_hill/2020-08-0216.17.52.csv"
    file2 = "sensor_data/up_hill/2020-08-0216.14.16.csv"
    file3 = "sensor_data/complex_road/2020-08-0215.57.04.csv"
    file4 = "sensor_data/fall_down/2020-08-0215.39.32.csv"

    df = pd.read_csv(file)
    df = split_by_t(df, 20)

    df1 = pd.read_csv(file1)
    df1 = split_by_t(df1, 20)

    df2 = pd.read_csv(file2)
    df2 = split_by_t(df2, 20)

    df3 = pd.read_csv(file3)
    df3 = split_by_t(df3, 20)

    df4 = pd.read_csv(file4)
    df4 = split_by_t(df4, 20)

    df = get_Gacc_0(df)
    FFT_show(df, "UGacc", 8, 0.1)
    df1 = get_Gacc_0(df1)
    FFT_show(df1, "UGacc", 8, 0.1)
    df2 = get_Gacc_0(df2)
    FFT_show(df2, "UGacc", 8, 0.1)
    df3 = get_Gacc_0(df3)
    FFT_show(df3, "UGacc", 8, 0.1)
    df4 = get_Gacc_0(df4)
    FFT_show(df4, "UGacc", 8, 0.1)

    # Data size is different
    anova = stats.f_oneway(df["UGacc"],df1["UGacc"],df2["UGacc"])
    print(anova)

    '''
    x_data = pd.DataFrame({'x1': df["UGacc"], 'x2': df["UGacc"], 'x3': df["UGacc"]})
    x_melt = pd.melt(x_data)
    posthoc = pairwise_tukeyhsd(
        x_melt['value'], x_melt['variable'],
        alpha=0.05)
    print(posthoc)
    '''

    print("\nstep_freq", step_fre(df))
    print("step_len", step_len(df))
    print("\nstep_freq", step_fre(df1))
    print("step_len", step_len(df1))
    print("\nstep_freq", step_fre(df2))
    print("step_len", step_len(df2))
    print("\nstep_freq", step_fre(df3))
    print("step_len", step_len(df3))

    model = train_speed()
    df = get_speed_ML(df, model)
    df1 = get_speed_ML(df1, model)
    df2 = get_speed_ML(df2, model)
    anova = stats.f_oneway(df["speed_pre"],df1["speed_pre"],df2["speed_pre"])
    print(anova)

    '''
    # we can use difference of frequency in walking between up-hill and down-hill
    # to show people's gait difference

    ttest = stats.stats.ttest_ind(df1["Gacc"],df2["Gacc"])
    print(ttest)
    print(ttest.statistic)
    print(ttest.pvalue)
    '''


def main():
    report()
    # stat_sample()

if __name__ == '__main__':
    main()