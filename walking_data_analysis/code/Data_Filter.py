from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from scipy.fftpack import fft,ifft, fftfreq, fftshift
from math import sqrt
from GPS_Tracks import GPSTracker
from Data_Analysis import classifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from random import randrange


def plot_xyz(df, x, y, z):
    # file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    # df = pd.read_csv(file)
    plt.plot(df['time'], df[x], 'b.', alpha=0.5, color='red')
    plt.plot(df['time'], df[y], 'b.', alpha=0.5, color='green')
    plt.plot(df['time'], df[z], 'b.', alpha=0.5, color='blue')
    plt.xlabel('time')
    plt.title("Scatter Plot")
    plt.xlabel("time")
    plt.legend([x, y, z])
    # plt.savefig("plot/Scatter" + str(randrange(100000)), dpi=500)
    plt.show()
    plt.clf()

def plot_xyz_L(df, x, y, z):
    # file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    # df = pd.read_csv(file)
    plt.plot(df['time'], df[x], color='red')
    plt.plot(df['time'], df[y], color='green')
    plt.plot(df['time'], df[z], color='blue')
    plt.xlabel('time')
    plt.legend([x, y, z])
    plt.show()

def low_pass(noisy_signal):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, noisy_signal)

# 0hz
def low_pass_0(noisy_signal):
    b, a = signal.butter(3, 0.01, 'lowpass')
    return signal.filtfilt(b, a, noisy_signal)

def low_pass_2(noisy_signal, o, w):
    b, a = signal.butter(o, w, 'lowpass')
    return signal.filtfilt(b, a, noisy_signal)

def high_pass(noisy_signal, o, w):
    b, a = signal.butter(o, w, 'highpass')
    return signal.filtfilt(b, a, noisy_signal)

# Learned from https://stackoverflow.com/
def FFT(fre, data, x, y, col):
    L = data.size
    # Nextpower2()
    N = int(np.power(2,np.ceil(np.log2(L))))
    Fre = np.arange(int(N / 2)) * fre / N
    FFT_1 = np.abs(fft(data,N))/L*2
    FFT_1 = FFT_1[range(int(N/2))]
    plt.plot(Fre, FFT_1)
    plt.xlim((0, x))
    plt.ylim((0, y))
    plt.title("Fast Fourier Transform for " + col)
    plt.xlabel("Frequency")
    # plt.savefig("FFT", dpi=500)
    plt.show()
    return FFT_1

# Learned from https://stackoverflow.com/
def FFT2(fre, x):
    fx = fft(x)
    abs = np.abs(fx)
    abs = fftshift(abs)
    angle = np.angle(fx)
    freq = fftfreq(x.size, fre)
    # FFT = fftshift(abs)
    plt.plot(freq[0:freq.size // 2], (2 / abs.size) * abs[0:abs.size // 2])
    # plt.plot(freq, abs)
    plt.show()

# Show fourier transform plot
def FFT_show(df, col, x, y):
    # df = df.apply(low_pass)
    data = np.array(df[col])
    time = df['time'].max() - df['time'].min()
    fre = data.size/time
    w = 2 * 5 / fre
    FFT(fre, data, x, y, col)

def split_df(df, a, b):
    size = df['time'].size
    df = df[int(size * a): int(size * b)]
    return df

def split_by_time(df, a, b):
    df = df[(df['time'] > a) & (df['time'] < b)]
    return df

def split_by_t(df, a):
    df = split_by_time(df, a, df['time'].max()-a)
    return df

def get_Gacc(df):
    df_low_pass = df.apply(low_pass)
    df['ux'] = df['gFx'] - df_low_pass['gFx']
    df['uy'] = df['gFy'] - df_low_pass['gFy']
    df['uz'] = df['gFz'] - df_low_pass['gFz']
    df['Gacc'] = df_low_pass['gFx'] * df['ux'] + df_low_pass['gFy'] * df['uy'] + df_low_pass['gFz'] * df['uz']
    x = np.array(df['Gacc'])
    time = df['time'].max() - df['time'].min()
    fre = x.size / time
    wl = 2 * 5 / fre
    wh = 2 * 1 / fre
    df['Gacc'] = low_pass_2(df['Gacc'], 4, wl)
    df['Gacc'] = high_pass(df['Gacc'], 4, wh)
    return df

# Return User Acceleration in direction of Gravity
def get_Gacc_0(df):
    df_low_pass = df.apply(low_pass_0)
    df['ux'] = df['gFx'] - df_low_pass['gFx']
    df['uy'] = df['gFy'] - df_low_pass['gFy']
    df['uz'] = df['gFz'] - df_low_pass['gFz']
    df['UGacc'] = df_low_pass['gFx'] * df['ux'] + df_low_pass['gFy'] * df['uy'] + df_low_pass['gFz'] * df['uz']
    x = np.array(df['UGacc'])
    time = df['time'].max() - df['time'].min()
    fre = x.size / time
    wl = 2 * 5 / fre
    wh = 2 * 1 / fre
    df['UGacc'] = low_pass_2(df['UGacc'], 4, wl)
    df['UGacc'] = high_pass(df['UGacc'], 4, wh)
    return df

# Return number of steps
def step_count(df):
    df = get_Gacc(df)
    Gacc_f = np.array(df['Gacc'])
    peaks = signal.argrelextrema(Gacc_f, np.greater)
    # steps = np.array(peaks).size/1.71
    steps = np.array(peaks).size / 2.33
    return steps

# Return frequency of steps
def step_fre(df):
    time = df['time'].max() - df['time'].min()
    steps = step_count(df)
    return steps/time

# Return length of steps
def step_len(df):
    tracker = GPSTracker(df)
    steps = step_count(df)
    dis = tracker.get_distance()
    if dis < 10:
        print("Warning: distance is too low")
    return dis/steps

# Return ML model for speed
def train_speed():
    file = "sensor_data/ML_Train/2020-08-0216.31.22.csv"
    df = pd.read_csv(file)
    df = split_by_time(df, 130, 210)

    df_low_pass = df.apply(low_pass)
    df['lp_ax'] = df_low_pass['ax']
    df['lp_ay'] = df_low_pass['ay']
    df['lp_az'] = df_low_pass['az']

    # X = df[['lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz', 'Azimuth', 'Pitch', 'Roll']].values
    # X = df[['lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz']].values

    df = get_Gacc_0(df)
    X = df[['UGacc', 'lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz']].values
    y = df['Speed (m/s)'].values
    y = (y * 5).astype('int')
    classifier(X, y)
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=16)
    )

    model.fit(X, y)
    return model

# Return prediction of speed
def get_speed_ML(df_v, model):
    df_low_pass_v = df_v.apply(low_pass)
    df_v['lp_ax'] = df_low_pass_v['ax']
    df_v['lp_ay'] = df_low_pass_v['ay']
    df_v['lp_az'] = df_low_pass_v['az']
    # X_v = df_v[['lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz', 'Azimuth', 'Pitch', 'Roll']].values
    # X_v = df_v[['lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz']].values

    df_v = get_Gacc_0(df_v)
    X_v = df_v[['UGacc', 'lp_ax', 'lp_ay', 'lp_az', 'wx', 'wy', 'wz']].values
    df_v['speed_pre'] = model.predict(X_v)/5

    '''
    plt.plot(df_v['time'], df_v['speed_pre'], 'b.', alpha=0.5)
    plt.plot(df_v['time'], df_v['Speed (m/s)'], color='red')
    plt.title("GPS Speed Vs KNN prediction speed")
    plt.xlabel("time")
    plt.ylabel("Speed (m/s)")
    plt.legend(['KNN prediction speed', 'GPS speed'])
    # plt.savefig("plot/ML", dpi=500)
    # plt.show()
    '''
    # df_v.to_csv("sample.csv")
    return df_v


def test1(file):
    # file = "sensor_data/fall_down/2020-08-0215.39.32.csv"
    df = pd.read_csv(file)
    '''
    size = df['time'].size

    # Cut
    df = df[int(size * 0.51) : int(size * 0.53)]
    '''
    df = split_by_time(df, 170, 175)
    plot_xyz(df, "gFx", "gFy", "gFz")
    plot_xyz(df, "ax", "ay", "az")
    df_low_pass = df.apply(low_pass)
    df_low_pass_n = df.apply(low_pass_0)
    plot_xyz(df_low_pass_n, "gFx", "gFy", "gFz")
    plot_xyz(df_low_pass, "ax", "ay", "az")

    '''
    df['Lacc'] = df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2
    df['Netacc'] = df['Lacc']**0.5
    plt.plot(df['time'], df['Lacc'], 'b.', alpha=0.5)
    plt.show()
    '''

    df['ux'] = df['gFx'] - df_low_pass['gFx']
    df['uy'] = df['gFy'] - df_low_pass['gFy']
    df['uz'] = df['gFz'] - df_low_pass['gFz']
    # plot_xyz_L(df, "ux", "uy", "uz")
    df['Gacc'] = df_low_pass['gFx'] * df['ux'] + df_low_pass['gFy'] * df['uy'] + df_low_pass['gFz'] * df['uz']
    # df = df[0:4000]
    #plt.plot(df['time'], df['Gacc'])
    #plt.show()
    # df.to_csv("sample.csv")

    x = np.array(df['Gacc'])
    time = df['time'].max() - df['time'].min()
    fre = x.size/time

    wl = 2*5/fre
    wh = 2*1/fre
    plt.plot(df['time'], df['Gacc'])
    plt.xlabel("time")
    plt.title("User Acceleration in direction of Gravity")
    # plt.savefig("plot/PF1", dpi=500)
    plt.show()
    df['Gacc'] = low_pass_2(df['Gacc'], 4, wl)
    df['Gacc'] = high_pass(df['Gacc'], 4, wh)
    plt.plot(df['time'], df['Gacc'])
    plt.xlabel("time")
    plt.title("User Acceleration in direction of Gravity")
    # plt.savefig("plot/PF2", dpi=500)
    plt.show()

    final = np.array(df['Gacc'])
    y = signal.argrelextrema(final, np.greater)
    print(np.array(y).size)

def test2():
    file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    df = pd.read_csv(file)
    df = split_by_time(df, 100, 300)
    df_low_pass = df.apply(low_pass)
    df['lp_ax'] = df_low_pass['ax']
    df['lp_ay'] = df_low_pass['ay']
    df['lp_az'] = df_low_pass['az']
    plot_xyz_L(df_low_pass, "ax", "ay", "az")
    plt.plot(df['time'], df['Speed (m/s)'])
    plt.show()
    X = df[['lp_ax', 'lp_ay', 'lp_az']].values
    y = df['Speed (m/s)'].values
    y = (y*100).astype('int')

    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model_svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear')
    )
    model_svc.fit(X_train, y_train)
    '''
    classifier(X, y)

def main():
    # model = train_speed()
    file = "sensor_data/complex_road/2020-08-0216.31.22.csv"
    file1 = "sensor_data/down_hill/2020-08-0215.36.53.csv"
    file2 = "sensor_data/up_hill/2020-08-0216.14.16.csv"
    file3 = "sensor_data/stop/2020-08-0215.33.39.csv"
    file4 = "sensor_data/fall_down/2020-08-0215.39.32.csv"


    df = pd.read_csv(file)
    df = split_by_t(df, 20)

    df1 = pd.read_csv(file1)
    df1 = split_by_t(df1, 20)

    df2 = pd.read_csv(file2)
    df2 = split_by_time(df2, 20, 80)

    df3 = pd.read_csv(file3)
    df3 = split_by_time(df3, 20, 80)

    df4 = pd.read_csv(file4)
    df4 = split_by_time(df4, 20, 80)

    low_pass_df = df.apply(low_pass)
    FFT_show(low_pass_df, "gFy", 6, 0.15)





if __name__ == '__main__':
    main()